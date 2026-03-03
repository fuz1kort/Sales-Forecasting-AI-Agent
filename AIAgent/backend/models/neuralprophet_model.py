"""
Модель прогнозирования на основе NeuralProphet.

Использует нейронные сети для прогнозирования временных рядов продаж
с поддержкой прогноза по отдельным магазинам.
"""

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet

from utils import find_columns


def neuralprophet_forecast(
    df: pd.DataFrame,
    periods: int = 30,
    forecast_type: str = "general",
    store_ids: list | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Прогноз продаж на основе NeuralProphet.

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
    date_col, sales_col, store_col = find_columns(df)
    if not sales_col:
        return {"error": f"Не найден столбец продаж. Доступные столбцы: {df.columns.tolist()}"}

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

    use_store = forecast_type == "by_store" and store_col and df[store_col].nunique() > 1

    # Подготовка данных
    if use_store:
        df_prepared = _prepare_store_data(df, date_col, sales_col, store_col, store_ids)
    else:
        df_prepared = _prepare_general_data(df, date_col, sales_col)

    if isinstance(df_prepared, dict) and "error" in df_prepared:
        return df_prepared

    # Коррекция горизонта прогноза
    periods = _adjust_forecast_periods(df_prepared, periods, start_date, end_date)

    # Обучение и прогноз
    try:
        model = _create_model(use_store)
        model.fit(df_prepared, freq="D")
        forecast_list, model_performance = _generate_forecast(
            model, df_prepared, periods, use_store, start_date, end_date
        )

        info = _build_info(date_col, sales_col, df_prepared, periods, use_store, store_col)

        return {
            "forecast": forecast_list,
            "model_performance": model_performance,
            "info": info
        }
    except Exception as e:
        print(f"NeuralProphet error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Ошибка NeuralProphet: {str(e)}"}


def _prepare_store_data(df, date_col, sales_col, store_col, store_ids):
    """Подготавливает данные для прогноза по магазинам."""
    cols = [date_col, sales_col, store_col]
    df_clean = df[cols].copy().dropna()

    if store_ids:
        df_clean = df_clean[df_clean[store_col].astype(str).isin([str(s) for s in store_ids])]

    if df_clean.empty:
        return {
            "error": (
                "Для указанных ID магазинов нет данных. "
                f"Столбец магазина: '{store_col}', запрошенные ID: {store_ids}."
            )
        }

    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    df_grouped = (
        df_clean
        .groupby([store_col, date_col])[sales_col]
        .sum()
        .reset_index()
        .sort_values([store_col, date_col])
    )

    return pd.DataFrame({
        'ds': df_grouped[date_col],
        'y': df_grouped[sales_col],
        'ID': df_grouped[store_col].astype(str)
    })


def _prepare_general_data(df, date_col, sales_col):
    """Подготавливает данные для общего прогноза."""
    df_clean = df[[date_col, sales_col]].copy().dropna().sort_values(date_col)
    daily_sales = df_clean.groupby(date_col)[sales_col].sum().reset_index()

    result = pd.DataFrame({'ds': daily_sales[date_col], 'y': daily_sales[sales_col]})

    if len(result) < 2:
        return {"error": "Недостаточно точек для прогноза"}

    return result


def _adjust_forecast_periods(df, periods, start_date, end_date):
    """Корректирует горизонт прогноза на основе дат."""
    end_dt = pd.to_datetime(end_date).date() if end_date else None
    last_hist_date = df["ds"].max().date()

    if end_dt and end_dt > last_hist_date:
        required_periods = (end_dt - last_hist_date).days
        if required_periods < 1:
            required_periods = 1
        if required_periods > periods:
            periods = required_periods

    return periods


def _create_model(use_store):
    """Создаёт и конфигурирует модель NeuralProphet."""
    return NeuralProphet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        quantiles=[0.05, 0.95],
        trend_global_local="local" if use_store else "global",
        season_global_local="local" if use_store else "global",
        learning_rate=0.01,
    )


def _generate_forecast(model, df, periods, use_store, start_date, end_date):
    """Генерирует прогноз и вычисляет метрики."""
    future = model.make_future_dataframe(df, periods=periods, n_historic_predictions=False)
    fcst = model.predict(future)

    yhat_col = 'yhat1' if 'yhat1' in fcst.columns else 'yhat'
    lower_col = 'yhat1 5.0%' if 'yhat1 5.0%' in fcst.columns else None
    upper_col = 'yhat1 95.0%' if 'yhat1 95.0%' in fcst.columns else None

    if 'y' in fcst.columns and fcst['y'].notna().any():
        future_only = fcst[fcst['y'].isna()]
    else:
        future_only = fcst

    if not use_store and len(future_only) > periods:
        future_only = future_only.tail(periods)

    # Фильтрация по датам
    if start_date or end_date:
        future_only = _filter_by_date_range(future_only, start_date, end_date)

    forecast_list = _build_forecast_list(future_only, yhat_col, lower_col, upper_col, use_store)

    # Метрики модели
    model_performance = _calc_model_metrics(fcst, yhat_col)

    return forecast_list, model_performance


def _filter_by_date_range(df, start_date, end_date):
    """Фильтрует DataFrame по диапазону дат."""
    start_dt = pd.to_datetime(start_date).date() if start_date else None
    end_dt = pd.to_datetime(end_date).date() if end_date else None

    ds_dates = df["ds"].dt.date
    mask = pd.Series(True, index=df.index)
    if start_dt:
        mask &= ds_dates >= start_dt
    if end_dt:
        mask &= ds_dates <= end_dt
    return df[mask]


def _build_forecast_list(df, yhat_col, lower_col, upper_col, use_store):
    """Строит список прогнозов в формате для API."""
    forecast_list = []
    for _, row in df.iterrows():
        item = {
            "date": row['ds'].strftime("%Y-%m-%d"),
            "forecast": float(row[yhat_col]) if yhat_col in row else float(row.iloc[2]),
        }
        if lower_col and lower_col in row:
            item["lower_bound"] = float(row[lower_col])
        if upper_col and upper_col in row:
            item["upper_bound"] = float(row[upper_col])
        if use_store and 'ID' in row:
            item["store_id"] = str(row['ID'])
        forecast_list.append(item)
    return forecast_list


def _calc_model_metrics(fcst, yhat_col):
    """Вычисляет метрики точности модели."""
    hist = fcst[fcst['y'].notna()] if 'y' in fcst.columns and fcst['y'].notna().any() else pd.DataFrame()

    mae = mape = 0.0
    if len(hist) > 0 and yhat_col in hist.columns:
        mae = float(np.mean(np.abs(hist[yhat_col] - hist['y'])))
        denom = hist['y'].replace(0, np.nan)
        mape = float(np.nanmean(np.abs((hist['y'] - hist[yhat_col]) / denom)) * 100) if denom.notna().any() else 0

    return {"mae": mae, "mape": mape, "training_samples": len(hist)}


def _build_info(date_col, sales_col, df, periods, use_store, store_col):
    """Строит информацию о модели и данных."""
    info = {
        "date_column": date_col,
        "sales_column": sales_col,
        "data_points": len(df),
        "forecast_periods": periods,
        "model": "NeuralProphet",
        "forecast_type": "by_store" if use_store else "general",
    }
    if use_store:
        info["store_column"] = store_col
        info["stores"] = df['ID'].unique().tolist() if 'ID' in df.columns else []
    return info

