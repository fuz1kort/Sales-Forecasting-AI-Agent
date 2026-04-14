"""
Модель прогнозирования на основе CatBoost.

Использует градиентный бустинг для прогнозирования временных рядов продаж.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from typing import Optional, List
from backend.utils import find_columns, smape
from backend.config.forecast_config import forecast_config
from backend.schemas.forecast_types import ForecastResult
import logging

logger = logging.getLogger(__name__)


def catboost_forecast(
    df: pd.DataFrame,
    periods: int = 30,
    forecast_type: str = "general",
    store_ids: Optional[List[str]] = None,
):
    """
    Прогноз продаж на основе CatBoost.

    Args:
        df: Исходные данные с продажами
        periods: Количество дней прогноза
        forecast_type: "general" — общий прогноз; "by_store" — по магазинам
        store_ids: Список ID магазинов (None = все)

    Returns:
        Словарь с прогнозом, метриками и информацией о модели
    """
    date_col, sales_col, store_col, product_col = find_columns(df)
    if not sales_col:
        return {"error": f"Не найден столбец продаж. Доступные столбцы: {df.columns.tolist()}"}

    # Фильтрация по магазинам
    if forecast_type == "by_store" and store_ids and store_col:
        df = df[df[store_col].isin(store_ids)].copy()
        if df.empty:
            return {"error": f"Нет данных для магазинов {store_ids}"}

    # Агрегация по неделям (как в Colab)
    df_weekly = df.copy()
    df_weekly[date_col] = pd.to_datetime(df_weekly[date_col])
    df_weekly = df_weekly.set_index(date_col).resample('W')[sales_col].sum().reset_index()
    df_weekly.columns = ['Date', 'Revenue']

    # Создание фич (лаги, сезонность)
    df_weekly['lag_1'] = df_weekly['Revenue'].shift(1)
    df_weekly['lag_2'] = df_weekly['Revenue'].shift(2)
    df_weekly['lag_4'] = df_weekly['Revenue'].shift(4)
    df_weekly['month'] = df_weekly['Date'].dt.month
    df_weekly['year'] = df_weekly['Date'].dt.year
    df_weekly = df_weekly.dropna()

    # Разделение на train/test
    train_size = int(len(df_weekly) * 0.8)
    train = df_weekly[:train_size]
    test = df_weekly[train_size:]

    features = ['lag_1', 'lag_2', 'lag_4', 'month', 'year']
    X_train = train[features]
    y_train = train['Revenue']
    X_test = test[features]
    y_test = test['Revenue']

    # Обучение модели
    model = CatBoostRegressor(iterations=1000, learning_rate=0.04, depth=5, verbose=0)
    model.fit(X_train, y_train)

    # Прогноз на тест (для метрик)
    forecast_test = model.predict(X_test)

    # Прогноз на будущее
    last_row = df_weekly.iloc[-1].copy()
    last_date = last_row['Date']
    future_forecasts = []
    for i in range(periods):
        # Обновление фич
        last_row['lag_4'] = last_row['lag_2']
        last_row['lag_2'] = last_row['lag_1']
        last_row['lag_1'] = last_row['Revenue']
        last_row['month'] = (last_row['month'] % 12) + 1
        if last_row['month'] == 1:
            last_row['year'] += 1

        X_future = last_row[features].values.reshape(1, -1)
        pred = model.predict(X_future)[0]
        last_row['Revenue'] = pred
        
        next_date = last_date + pd.Timedelta(days=1)
        future_forecasts.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "forecast": float(pred)
        })
        last_date = next_date

    # Метрики
    mae = np.mean(np.abs(y_test - forecast_test))
    rmse = np.sqrt(np.mean((y_test - forecast_test)**2))
    mape = np.mean(np.abs((y_test - forecast_test) / y_test)) * 100

    return {
        "status": "success",
        "model": "catboost",
        "forecast": future_forecasts,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        },
        "info": f"CatBoost модель обучена на {len(train)} неделях, прогноз на {periods} недель."
    }


def ensemble_forecast_optimized(
        df: pd.DataFrame,
        periods: int = forecast_config.DEFAULT_PERIODS,
        forecast_type: str = "general",
        store_ids: Optional[List[str]] = None,
) -> ForecastResult:
    """
    Оптимизированный ансамбль из CatBoost, XGBoost, LightGBM.
    Веса подбираются минимизацией sMAPE на тестовой выборке.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback if scipy is not available
        minimize = None

    import lightgbm as lgb
    from xgboost import XGBRegressor

    # Фильтрация по магазинам
    if forecast_type == "by_store" and store_ids:
        _, _, store_col, _ = find_columns(df)
        if store_col:
            df = df[df[store_col].isin(store_ids)].copy()
            if df.empty:
                return {"error": f"Нет данных для магазинов {store_ids}"}

    # Агрегация по неделям
    date_col, sales_col, _, _ = find_columns(df)
    df_weekly = df.copy()
    df_weekly[date_col] = pd.to_datetime(df_weekly[date_col])
    df_weekly = df_weekly.set_index(date_col).resample('W')[sales_col].sum().reset_index()
    df_weekly.columns = ['Date', 'Revenue']

    # Фичи
    df_weekly['lag_1'] = df_weekly['Revenue'].shift(1)
    df_weekly['lag_2'] = df_weekly['Revenue'].shift(2)
    df_weekly['lag_4'] = df_weekly['Revenue'].shift(4)
    df_weekly['month'] = df_weekly['Date'].dt.month
    df_weekly['year'] = df_weekly['Date'].dt.year
    df_weekly = df_weekly.dropna()

    train_size = int(len(df_weekly) * 0.8)
    train = df_weekly[:train_size]
    test = df_weekly[train_size:]

    features = ['lag_1', 'lag_2', 'lag_4', 'month', 'year']
    X_train = train[features]
    y_train = train['Revenue']
    X_test = test[features]
    y_test = test['Revenue']

    # Обучение моделей
    cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.04, depth=5, verbose=0)
    cat_model.fit(X_train, y_train)
    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.45, max_depth=5)
    xgb_model.fit(X_train, y_train)
    lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.005, num_leaves=31, min_data_in_leaf=3, verbosity=-1)
    lgb_model.fit(X_train, y_train)

    def ensemble_error(weights):
        cat_pred = cat_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)
        lgb_pred = lgb_model.predict(X_test)
        forecast = weights[0] * cat_pred + weights[1] * xgb_pred + weights[2] * lgb_pred
        return smape(y_test, forecast)

    # Оптимизация весов
    if minimize is not None:
        initial_weights = [0.5, 0.3, 0.2]
        bounds = [(0, 1), (0, 1), (0, 1)]
        constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
        result = minimize(ensemble_error, initial_weights, bounds=bounds, constraints=constraints)
        optimal_weights = result.x
    else:
        # Fallback to equal weights if scipy not available
        optimal_weights = [0.33, 0.33, 0.34]

    logger.info(f"Оптимальные веса: CatBoost {optimal_weights[0]:.2f}, XGBoost {optimal_weights[1]:.2f}, LightGBM {optimal_weights[2]:.2f}")

    # Прогноз на будущее
    forecast_points = []
    last_row = df_weekly.iloc[-1].copy()
    last_date = last_row['Date']
    for i in range(periods):
        last_row['lag_4'] = last_row['lag_2']
        last_row['lag_2'] = last_row['lag_1']
        last_row['lag_1'] = last_row['Revenue']
        last_row['month'] = (last_row['month'] % 12) + 1
        if last_row['month'] == 1:
            last_row['year'] += 1

        X_future = last_row[features].values.reshape(1, -1)
        cat_pred = cat_model.predict(X_future)[0]
        xgb_pred = xgb_model.predict(X_future)[0]
        lgb_pred = lgb_model.predict(X_future)[0]
        pred = optimal_weights[0] * cat_pred + optimal_weights[1] * xgb_pred + optimal_weights[2] * lgb_pred

        next_date = last_date + pd.Timedelta(days=1)
        forecast_points.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "forecast": float(pred)
        })

        last_row['Revenue'] = pred
        last_date = next_date

    return {
        "status": "success",
        "model": "ensemble_optimized",
        "forecast": forecast_points,
        "weights": optimal_weights.tolist(),
        "info": {
            "description": f"Ансамбль оптимизирован с весами {optimal_weights.tolist()}",
            "model_used": "ensemble",
            "periods_requested": periods,
            "forecast_type": forecast_type,
        }
    }