"""
Фасад сервиса прогнозирования продаж.

Предоставляет единый интерфейс для вызова моделей прогнозирования:
- NeuralProphet
- ARIMA (SARIMAX)
- Auto-select (по результатам backtest выбирает лучшую модель)

Единый контракт:
forecast(df, model_type, periods, forecast_type, store_ids, start_date, end_date)
возвращает словарь вида {"forecast": [...], "model_performance": {...}, "info": {...}} или {"error": str}.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import pandas as pd

from models import neuralprophet_forecast, sarima_forecast, backtest_models
from utils import find_columns


def _normalize_store_ids(store_ids: Optional[List[str | int]]) -> Optional[List[str]]:
    if store_ids is None:
        return None
    # приведение к строкам и очистка пустых
    norm = [str(s).strip() for s in store_ids if str(s).strip()]
    return norm if norm else None


def _validate_input_df(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return "Пустой DataFrame: нет данных для прогноза."
    # Проверка наличия столбца продаж
    _, sales_col, _ = find_columns(df)
    if not sales_col:
        return (
            "Не удалось определить столбец продаж. "
            f"Доступные столбцы: {df.columns.tolist()}"
        )
    return None


def get_forecast(
    df: pd.DataFrame,
    model_type: str = "neuralprophet",
    periods: int = 30,
    forecast_type: str = "general",
    store_ids: Optional[List[str | int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Единая точка входа для прогнозирования.

    Args:
        df: входной DataFrame с продажами
        model_type: "neuralprophet" | "sarima" | "auto"
        periods: горизонт прогноза (в днях)
        forecast_type: "general" или "by_store"
        store_ids: список ID магазинов для по-магазинного прогноза
        start_date: начало диапазона прогноза (��пционально)
        end_date: конец диапазона прогноза (опционально)

    Returns:
        Словарь результата модели или {"error": str}
    """
    # Валидация входных данных
    err = _validate_input_df(df)
    if err:
        return {"error": err}

    model_type = (model_type or "").strip().lower()
    store_ids_norm = _normalize_store_ids(store_ids)

    try:
        if model_type in ("neuralprophet", "neural", "np"):
            return neuralprophet_forecast(
                df=df,
                periods=int(periods),
                forecast_type=forecast_type,
                store_ids=store_ids_norm,
                start_date=start_date,
                end_date=end_date,
            )
        elif model_type in ("arima", "sarima", "sarimax"):
            return sarima_forecast(
                df=df,
                periods=int(periods),
                forecast_type=forecast_type,
                store_ids=store_ids_norm,
            )
        elif model_type in ("auto", "best"):
            return _forecast_auto(
                df=df,
                periods=int(periods),
                forecast_type=forecast_type,
                store_ids=store_ids_norm,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return {"error": f"Неизвестный тип модели: {model_type}. Ожидается neuralprophet|sarima|auto"}
    except Exception as e:
        return {"error": f"Ошибка сервиса прогноза: {e}"}


def _forecast_auto(
    df: pd.DataFrame,
    periods: int,
    forecast_type: str,
    store_ids: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, Any]:
    """
    Выбирает лучшую модель по backtest (holdout на конце ряда) и строит итоговый прогноз.
    Критерий: минимальный MAPE, затем MAE.
    """
    bt = backtest_models(df, test_days=max(14, min(60, int(periods))))
    if isinstance(bt, dict) and bt.get("error"):
        # если backtest не удался — по умолчанию NeuralProphet
        chosen = "neuralprophet"
    else:
        chosen = (bt or {}).get("best_model") or "neuralprophet"

    if chosen == "sarima":
        result = sarima_forecast(
            df=df,
            periods=periods,
            forecast_type=forecast_type,
            store_ids=store_ids,
        )
    else:
        result = neuralprophet_forecast(
            df=df,
            periods=periods,
            forecast_type=forecast_type,
            store_ids=store_ids,
            start_date=start_date,
            end_date=end_date,
        )

    # Добавляем в info информацию о выборе модели
    if isinstance(result, dict) and not result.get("error"):
        info = result.get("info", {})
        info.update({
            "model_selection": "auto",
            "selected_model": chosen,
            "backtest_summary": {
                "test_days": (bt or {}).get("test_days"),
                "metrics": (bt or {}).get("metrics"),
                "errors": (bt or {}).get("errors"),
            },
        })
        result["info"] = info
    return result
