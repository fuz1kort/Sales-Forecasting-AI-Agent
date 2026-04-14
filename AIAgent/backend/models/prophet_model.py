"""
Модель прогнозирования на основе Prophet.

Использует Prophet для прогнозирования временных рядов продаж.
Поддерживает:
- Праздники конкретной страны
- Дополнительные регрессоры (weekend, день месяца, месяц)
- Мультипликативную сезонность
"""

import pandas as pd
import numpy as np
import logging
from prophet import Prophet

from backend.utils import find_columns

logger = logging.getLogger(__name__)


def prophet_forecast(
    df: pd.DataFrame,
    periods: int = 30,
    forecast_type: str = "general",
    store_ids: list | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    country: str = "UK",
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
        country: Код страны для добавления праздников (UK, US, RU, etc)

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
            logger.warning(f"Ошибка парсинга дат: {e}. Используется синтетическая дата.")
            df = df.copy()
            df['synthetic_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            date_col = 'synthetic_date'

    # Агрегация по дням
    df_agg = df.groupby(date_col)[sales_col].sum().reset_index()
    df_agg.columns = ['ds', 'y']
    
    # Добавляем дополнительные регрессоры
    df_agg['is_weekend'] = df_agg['ds'].dt.weekday.isin([5, 6]).astype(int)
    df_agg['day_of_month'] = df_agg['ds'].dt.day
    df_agg['month'] = df_agg['ds'].dt.month

    # Обучение модели
    try:
        model = Prophet(
            interval_width=0.95,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.1
        )
        
        # Добавляем праздники страны
        try:
            model.add_country_holidays(country_name=country)
        except Exception as e:
            logger.warning(f"Не удалось добавить праздники {country}: {e}")
        
        # Добавляем регрессоры
        model.add_regressor('is_weekend')
        model.add_regressor('day_of_month')
        model.add_regressor('month')
        
        model.fit(df_agg)
    except Exception as e:
        logger.warning(f"Ошибка при обучении Prophet: {e}")
        return {"error": f"Ошибка обучения Prophet: {str(e)}"}

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

    # Добавляем регрессоры в future
    future['is_weekend'] = future['ds'].dt.weekday.isin([5, 6]).astype(int)
    future['day_of_month'] = future['ds'].dt.day
    future['month'] = future['ds'].dt.month

    forecast = model.predict(future)

    # Извлечение прогноза с confidence intervals
    if start_date and end_date:
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(future))
        forecast_points = [{
            "date": row['ds'].strftime('%Y-%m-%d'),
            "forecast": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        } for _, row in forecast_data.iterrows()]
    else:
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_points = [{
            "date": row['ds'].strftime('%Y-%m-%d'),
            "forecast": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        } for _, row in forecast_data.iterrows()]

    # Метрики (на основе исторических данных)
    historical_forecast = model.predict(df_agg[['ds', 'is_weekend', 'day_of_month', 'month']])
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
            "country": country,
        }
    }