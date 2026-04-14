"""
Сервис backtest для сравнения моделей прогнозирования.

Чистая бизнес-логика без зависимостей от фреймворков.
Используется из:
- forecast_service.py (для auto-выбора модели)
- agent/tools/forecast/backtest_tools.py (для @tool run_backtest)

Тестирует модели:
- SARIMA (с поддержкой Auto-ARIMA)
- Prophet (с праздниками и регрессорами)
- CatBoost Ensemble
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

from backend.utils import find_columns, calculate_metrics
from backend.models import sarima_forecast
from backend.models.prophet_model import prophet_forecast
from backend.models.catboost_model import ensemble_forecast_optimized

logger = logging.getLogger(__name__)


def backtest_models(
        df: pd.DataFrame,
        test_days: int = 30,
        sales_col: Optional[str] = None,
        date_col: Optional[str] = None,
) -> dict:
    """
    Сравнение моделей на исторических данных (holdout validation).

    Args:
        df: Исходные данные
        test_days: Количество дней для тестовой выборки
        sales_col: Название колонки продаж (авто-поиск, если None)
        date_col: Название колонки даты (авто-поиск, если None)

    Returns:
        dict с результатами сравнения
    """
    # Авто-поиск колонок
    if not date_col or not sales_col:
        date_col, sales_col, _, _ = find_columns(df)

    if not date_col or not sales_col:
        return {"error": "Не найдены колонки даты или продаж", "status": "error"}

    # Подготовка данных
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, sales_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if len(df) < test_days + 10:
        return {
            "error": f"Недостаточно данных: нужно ≥{test_days + 10}, есть {len(df)}",
            "status": "error"
        }

    # Разделение train/test
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()

    results = {
        "status": "success",
        "test_days": test_days,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "metrics": {},
        "best_model": None,
        "predictions": {
            "date": test_df[date_col].dt.strftime("%Y-%m-%d").tolist(),
            "actual": test_df[sales_col].tolist(),
            "sarima": [],
            "prophet": [],
            "ensemble": [],
        }
    }

    # Тестируем SARIMA
    sarima_result = _test_model(
        "sarima", sarima_forecast,
        train_df, test_df, sales_col, test_days
    )
    results["metrics"]["sarima"] = sarima_result
    if "error" not in sarima_result:
        results["predictions"]["sarima"] = sarima_result.get("predicted", [])

    # Тестируем Prophet
    prophet_result = _test_model(
        "prophet", prophet_forecast,
        train_df, test_df, sales_col, test_days
    )
    results["metrics"]["prophet"] = prophet_result
    if "error" not in prophet_result:
        results["predictions"]["prophet"] = prophet_result.get("predicted", [])

    # Тестируем Ensemble
    ensemble_result = _test_model(
        "ensemble", ensemble_forecast_optimized,
        train_df, test_df, sales_col, test_days
    )
    results["metrics"]["ensemble"] = ensemble_result
    if "error" not in ensemble_result:
        results["predictions"]["ensemble"] = ensemble_result.get("predicted", [])

    # Выбираем лучшую модель
    results["best_model"] = _select_best_by_mape(results["metrics"])

    return results


def _test_model(
        name: str,
        model_func,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sales_col: str,
        periods: int
) -> dict:
    """Тестирует одну модель и возвращает метрики."""
    try:
        result = model_func(df=train_df, periods=periods, forecast_type="general")

        if "error" in result or "forecast" not in result:
            return {"error": result.get("error", "Unknown error")}

        predicted = [p["forecast"] for p in result["forecast"][:len(test_df)]]
        actual = test_df[sales_col].values[:len(predicted)]

        metrics = calculate_metrics(actual, predicted)
        metrics["predicted"] = predicted  # для визуализации
        return metrics

    except Exception as e:
        logger.error(f"{name} test error: {e}")
        return {"error": str(e)}


def _select_best_by_mape(metrics: dict) -> str:
    """Выбирает модель с минимальным MAPE."""
    models = ["sarima", "prophet", "ensemble"]
    best_model = "sarima"  # fallback
    best_mape = float("inf")

    for model in models:
        mape = metrics.get(model, {}).get("mape", float("inf"))
        if mape < best_mape:
            best_mape = mape
            best_model = model

    return best_model