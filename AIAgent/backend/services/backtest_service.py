"""
Сервис backtest для сравнения моделей прогнозирования.

Чистая бизнес-логика без зависимостей от фреймворков.
Используется из:
- forecast_service.py (для auto-выбора модели)
- agent/tools/forecast/backtest_tools.py (для @tool run_backtest)
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

from utils import find_columns
from models import neuralprophet_forecast, sarima_forecast

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
        date_col, sales_col, _ = find_columns(df)

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
            "neuralprophet": [],
            "sarima": [],
        }
    }

    # Тестируем NeuralProphet
    np_result = _test_model(
        "neuralprophet", neuralprophet_forecast,
        train_df, test_df, sales_col, test_days
    )
    results["metrics"]["neuralprophet"] = np_result
    if "error" not in np_result:
        results["predictions"]["neuralprophet"] = np_result.get("predicted", [])

    # Тестируем SARIMA
    sarima_result = _test_model(
        "sarima", sarima_forecast,
        train_df, test_df, sales_col, test_days
    )
    results["metrics"]["sarima"] = sarima_result
    if "error" not in sarima_result:
        results["predictions"]["sarima"] = sarima_result.get("predicted", [])

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

        metrics = _calc_metrics(actual, predicted)
        metrics["predicted"] = predicted  # для визуализации
        return metrics

    except Exception as e:
        logger.error(f"{name} test error: {e}")
        return {"error": str(e)}


def _calc_metrics(actual: np.ndarray, predicted: list) -> dict:
    """Вычисляет MAE, MAPE, RMSE."""
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))

    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 0.0

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    return {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
    }


def _select_best_by_mape(metrics: dict) -> str:
    """Выбирает модель с минимальным MAPE."""
    np_mape = metrics.get("neuralprophet", {}).get("mape", float("inf"))
    sarima_mape = metrics.get("sarima", {}).get("mape", float("inf"))

    if np_mape == float("inf") and sarima_mape == float("inf"):
        return "neuralprophet"  # fallback

    return "neuralprophet" if np_mape <= sarima_mape else "sarima"