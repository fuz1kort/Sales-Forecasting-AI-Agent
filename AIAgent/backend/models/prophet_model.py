"""
Модель прогнозирования на основе Prophet.

Использует аддитивную модель для прогнозирования временных рядов продаж.
"""

import pandas as pd
import numpy as np
from prophet import Prophet

from backend.utils import find_columns


def prophet_forecast(
    df: pd.DataFrame,
    periods: int = 30,
    forecast_type: str = "general",
    store_ids: list | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Прогноз продаж на основе Prophet.

    Args:
        df: Исходные данные с продажами
        periods: Количество дней прогноза
        forecast_type: "general" — общий прогноз; "by_store" — по магазинам
        store_ids: Список ID магазинов (None = все)
        start_date: Начальная дата прогноза (опционально)
        end_date: Конечная дата прогноза (опционально)

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

    # Обработка дат
    if not date_col:
        df = df.copy()
        df['synthetic_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        date_col = 'synthetic_date'
    else:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            print(f"Ошибка парсинга дат: {e}")
            df = df.copy()
            df['synthetic_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            date_col = 'synthetic_date'

    # Агрегация по дням
    df_agg = df.groupby(date_col)[sales_col].sum().reset_index()
    df_agg.columns = ['ds', 'y']

    # Обучение модели
    model = Prophet()
    model.fit(df_agg)

    # Создание future dataframe
    if start_date and end_date:
        # Прогноз на указанный период
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        future_dates = pd.date_range(start=start, end=end, freq='D')
        future = pd.DataFrame({'ds': future_dates})
    else:
        # Стандартный прогноз на periods дней
        future = model.make_future_dataframe(periods=periods)

    forecast = model.predict(future)

    # Извлечение прогноза
    if start_date and end_date:
        forecast_values = forecast['yhat'].values.tolist()
        forecast_dates = forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        forecast_points = [{"date": d, "forecast": f} for d, f in zip(forecast_dates, forecast_values)]
    else:
        forecast_values = forecast['yhat'].tail(periods).values.tolist()
        forecast_points = forecast_values

    # Метрики (на основе исторических данных)
    historical_forecast = model.predict(df_agg[['ds']])
    mae = np.mean(np.abs(df_agg['y'] - historical_forecast['yhat']))
    rmse = np.sqrt(np.mean((df_agg['y'] - historical_forecast['yhat'])**2))

    return {
        "status": "success",
        "model": "prophet",
        "forecast": forecast_points,
        "metrics": {
            "mae": mae,
            "rmse": rmse
        },
        "info": {
            "description": f"Prophet модель обучена на {len(df_agg)} днях, прогноз на {periods} дней.",
            "model_used": "prophet",
            "periods_requested": periods,
            "forecast_type": forecast_type,
        }
    }