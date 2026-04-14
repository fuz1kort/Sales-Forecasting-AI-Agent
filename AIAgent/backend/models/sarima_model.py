"""
Модель прогнозирования на основе ARIMA/SARIMA.

Использует статистические методы (сезонные авторегрессионные модели)
для прогнозирования временных рядов продаж.

По умолчанию использует Auto-ARIMA (автоматический подбор параметров).
Если pmdarima не установлена, использует SARIMA с фиксированными параметрами.
"""

import pandas as pd
import numpy as np
import importlib
import logging
from typing import Optional, List

try:
    from pmdarima import auto_arima
    HAS_AUTO_ARIMA = True
except ImportError:
    HAS_AUTO_ARIMA = False

from backend.utils import find_columns

logger = logging.getLogger(__name__)

def sarima_forecast(
    df: pd.DataFrame,
    periods: int = 30,
    forecast_type: str = "general",
    store_ids: list | None = None,
):
    """
    Прогноз продаж на основе ARIMA/SARIMA.

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

    # Подготовка даты
    if not date_col:
        df = df.copy()
        df["synthetic_date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
        date_col = "synthetic_date"
    else:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    try:
        SARIMAX = importlib.import_module("statsmodels.tsa.statespace.sarimax").SARIMAX

        info = {
            "date_column": date_col,
            "sales_column": sales_col,
            "forecast_periods": periods,
            "model": "SARIMA(1,0,0)x(1,1,1,7)",
        }

        use_store = forecast_type == "by_store" and store_col and df[store_col].nunique() > 1

        if use_store:
            forecast_list, info = _forecast_by_store(
                df, date_col, sales_col, store_col, store_ids, periods, SARIMAX, info
            )
        else:
            forecast_list, info = _forecast_general(
                df, date_col, sales_col, periods, SARIMAX, info
            )

        if "error" in info:
            return {"error": info["error"]}

        return {
            "forecast": forecast_list,
            "model_performance": {},
            "info": info,
        }
    except Exception as e:
        return {"error": f"Ошибка SARIMA: {str(e)}"}


def _forecast_by_store(df, date_col, sales_col, store_col, store_ids, periods, SARIMAX, info):
    """Генерирует прогноз по магазинам."""
    cols = [date_col, sales_col, store_col]
    df_clean = df[cols].copy().dropna()

    if store_ids:
        df_clean = df_clean[df_clean[store_col].astype(str).isin([str(s) for s in store_ids])]

    if df_clean.empty:
        return [], {
            "error": (
                "Для указанных ID магазинов нет данных. "
                f"Столбец магазина: '{store_col}', запрошенные ID: {store_ids}."
            )
        }

    forecast_list = []
    stores = sorted(df_clean[store_col].astype(str).unique())

    for sid in stores:
        df_store = df_clean[df_clean[store_col].astype(str) == sid]
        daily = (
            df_store[[date_col, sales_col]]
            .groupby(date_col)[sales_col]
            .sum()
            .sort_index()
        )

        if len(daily) < 10:
            continue

        forecast_list.extend(_fit_and_forecast_store(daily, sid, periods, SARIMAX))

    info["forecast_type"] = "by_store"
    info["store_column"] = store_col
    info["stores"] = stores

    return forecast_list, info


def _forecast_general(df, date_col, sales_col, periods, SARIMAX, info):
    """Генерирует общий прогноз по всем данным."""
    daily = (
        df[[date_col, sales_col]]
        .dropna()
        .groupby(date_col)[sales_col]
        .sum()
        .sort_index()
    )

    if len(daily) < 10:
        return [], {"error": "Недостаточно точек для прогноза SARIMA (минимум ~10 дат)."}

    forecast_list = _fit_and_forecast_general(daily, periods, SARIMAX)

    info["forecast_type"] = "general"
    info["data_points"] = len(daily)

    return forecast_list, info


def _valid_number(x):
    try:
        xf = float(x)
        if np.isfinite(xf):
            return True
    except Exception:
        pass
    return False


def _fallback_level(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return 0.0
    tail = series.tail(min(7, len(series)))
    val = float(tail.mean()) if tail.nunique() > 0 else float(series.iloc[-1])
    return val


def _fit_and_forecast_store(daily, store_id, periods, SARIMAX):
    """Обучает модель и генерирует прогноз для одного магазина."""
    # Ensure numeric series
    daily = pd.to_numeric(daily, errors="coerce").dropna()
    if daily.empty:
        return []

    # If series is (near) constant, forecast constant level
    if daily.std() < 1e-9:
        level = float(daily.iloc[-1])
        last_date = daily.index.max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")
        return [{
            "date": d.strftime("%Y-%m-%d"),
            "forecast": level,
            "store_id": str(store_id),
        } for d in future_dates]

    model = SARIMAX(
        daily,
        order=(1, 0, 0),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    fc_res = fitted.get_forecast(steps=periods)
    fc_values = fc_res.predicted_mean
    conf_int = fc_res.conf_int(alpha=0.1)

    last_date = daily.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    level_fallback = _fallback_level(daily)
    forecast_list = []
    for i, d in enumerate(future_dates):
        fc = fc_values.iloc[i] if i < len(fc_values) else np.nan
        # sanitize forecast value
        forecast_val = float(fc) if _valid_number(fc) else level_fallback

        item = {
            "date": d.strftime("%Y-%m-%d"),
            "forecast": forecast_val,
            "store_id": str(store_id),
        }
        # add bounds only if valid
        if conf_int is not None and i < len(conf_int):
            try:
                lo, hi = conf_int.iloc[i]
                if _valid_number(lo):
                    item["lower_bound"] = float(lo)
                if _valid_number(hi):
                    item["upper_bound"] = float(hi)
            except Exception:
                pass

        forecast_list.append(item)

    return forecast_list


def _fit_and_forecast_general(daily, periods, SARIMAX):
    """Обучает модель и генерирует общий прогноз."""
    # Ensure numeric series
    daily = pd.to_numeric(daily, errors="coerce").dropna()
    if daily.empty:
        return []

    # If series is (near) constant, forecast constant level
    if daily.std() < 1e-9:
        level = float(daily.iloc[-1])
        last_date = daily.index.max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")
        return [{
            "date": d.strftime("%Y-%m-%d"),
            "forecast": level,
        } for d in future_dates]

    model = SARIMAX(
        daily,
        order=(1, 0, 0),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    fc_res = fitted.get_forecast(steps=periods)
    fc_values = fc_res.predicted_mean
    conf_int = fc_res.conf_int(alpha=0.1)

    last_date = daily.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    level_fallback = _fallback_level(daily)
    forecast_list = []
    for i, d in enumerate(future_dates):
        fc = fc_values.iloc[i] if i < len(fc_values) else np.nan
        forecast_val = float(fc) if _valid_number(fc) else level_fallback
        item = {
            "date": d.strftime("%Y-%m-%d"),
            "forecast": forecast_val,
        }
        if conf_int is not None and i < len(conf_int):
            try:
                lo, hi = conf_int.iloc[i]
                if _valid_number(lo):
                    item["lower_bound"] = float(lo)
                if _valid_number(hi):
                    item["upper_bound"] = float(hi)
            except Exception:
                pass
        forecast_list.append(item)

    return forecast_list

