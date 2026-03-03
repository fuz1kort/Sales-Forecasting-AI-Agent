"""
Сравнение моделей прогнозирования (backtesting).

Сравнивает эффективность различных моделей на исторических данных
используя stratified holdout validation.
"""

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import importlib

from utils import find_columns, calc_mae_mape


def backtest_models(
    df: pd.DataFrame,
    test_days: int = 30,
):
    """
    Сравнение моделей на истории (holdout validation).

    Разделяет данные: последние test_days как тест, остальное как обучение.

    Args:
        df: Исходные данные с продажами
        test_days: Количество дней для тестирования

    Returns:
        Словарь с метриками и предсказаниями обеих моделей
    """
    date_col, sales_col, _ = find_columns(df)
    if not sales_col:
        return {"error": f"Не найден столбец продаж. Доступные столбцы: {df.columns.tolist()}"}

    if not date_col:
        df = df.copy()
        df["synthetic_date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
        date_col = "synthetic_date"
    else:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    daily = _make_daily_series_filled(df, date_col, sales_col)

    if len(daily) < max(20, test_days + 5):
        return {
            "error": (
                f"Недостаточно данных для backtest. Нужно хотя бы ~{max(20, test_days + 5)} дней, "
                f"а найдено {len(daily)}."
            )
        }

    test_days = int(test_days)
    test_days = max(7, min(test_days, len(daily) // 2))

    y_train = daily.iloc[:-test_days]
    y_test = daily.iloc[-test_days:]
    test_dates = y_test.index

    # Прогнозы по моделям
    sarima_pred, sarima_metrics, sarima_error = _backtest_sarima(y_train, y_test, test_days)
    np_pred, np_metrics, np_error = _backtest_neuralprophet(y_train, y_test, test_days)

    metrics = {}
    if np_metrics is not None:
        metrics["neuralprophet"] = np_metrics
    if sarima_metrics is not None:
        metrics["sarima"] = sarima_metrics

    # Выбор лучшей модели
    best_model = None
    if metrics:
        best_model = min(
            metrics.items(),
            key=lambda kv: (kv[1].get("mape", 0.0), kv[1].get("mae", 0.0))
        )[0]

    return {
        "status": "success",
        "test_days": test_days,
        "date_range": {
            "train_start": str(y_train.index.min().date()),
            "train_end": str(y_train.index.max().date()),
            "test_start": str(test_dates.min().date()),
            "test_end": str(test_dates.max().date()),
        },
        "used_columns": {"date": date_col, "sales": sales_col},
        "metrics": metrics,
        "errors": {
            "neuralprophet": np_error,
            "sarima": sarima_error,
        },
        "best_model": best_model,
        "predictions": {
            "date": [d.strftime("%Y-%m-%d") for d in test_dates],
            "actual": [float(v) for v in y_test.values],
            "neuralprophet": [float(v) for v in (np_pred if np_pred is not None else [np.nan] * test_days)],
            "sarima": [float(v) for v in (sarima_pred if sarima_pred is not None else [np.nan] * test_days)],
        },
    }


def _make_daily_series_filled(df: pd.DataFrame, date_col: str, sales_col: str) -> pd.Series:
    """Создаёт дневной ряд с непрерывной дневной частотой. Пропуски интерполируются, а не заполняются нулями."""
    df = df[[date_col, sales_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col, sales_col])
    daily = df.groupby(date_col)[sales_col].sum().sort_index()

    # Приводим к ежедневной частоте (без искусственных нулей)
    full_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)

    if daily.isna().any():
        # Интерполяция по времени для внутренних пропусков + заполнение краёв
        daily = daily.interpolate(method="time")
        daily = daily.ffill().bfill()

    daily.index.name = "date"
    return daily.astype(float)


def _backtest_sarima(y_train, y_test, test_days):
    """Backtesting для модели SARIMA."""
    try:
        SARIMAX = importlib.import_module("statsmodels.tsa.statespace.sarimax").SARIMAX

        # Лог-преобразование для устойчивости (обрабатываем нули/отрицательные)
        y_train_pos = np.clip(y_train.to_numpy(dtype=float), a_min=0.0, a_max=None)
        y_train_log = np.log1p(y_train_pos)

        model = SARIMAX(
            y_train_log,
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 7),
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        fc_res = fitted.get_forecast(steps=test_days)
        fc_log = fc_res.predicted_mean

        # Обратное преобразование и защита от отрицательных значений
        pred = np.expm1(np.asarray(fc_log, dtype=float))
        pred = np.maximum(pred, 0.0)

        metrics = calc_mae_mape(y_test.to_numpy(dtype=float), pred)
        return pred, metrics, None
    except Exception as e:
        return None, None, f"Ошибка SARIMA: {str(e)}"


def _backtest_neuralprophet(y_train, y_test, test_days):
    """Backtesting для модели NeuralProphet."""
    try:
        df_train = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
        m = NeuralProphet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            quantiles=[0.05, 0.95],
            learning_rate=0.01,
        )
        m.fit(df_train, freq="D")
        future = m.make_future_dataframe(df_train, periods=test_days, n_historic_predictions=False)
        fcst = m.predict(future)
        yhat_col = "yhat1" if "yhat1" in fcst.columns else "yhat"
        pred = np.asarray(fcst[yhat_col].values, dtype=float)
        metrics = calc_mae_mape(y_test.to_numpy(dtype=float), pred)
        return pred, metrics, None
    except Exception as e:
        return None, None, f"Ошибка NeuralProphet: {str(e)}"

