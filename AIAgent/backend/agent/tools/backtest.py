"""
Инструмент backtest для сравнения моделей прогнозирования.
Позволяет агенту автономно запускать сравнение моделей
на исторических данных.
"""
import logging
from typing import Optional
import json
import pandas as pd
import numpy as np
from smolagents import tool
from agent.state import get_session_manager
from utils import find_columns
from models.neuralprophet_model import neuralprophet_forecast
from models.sarima_model import sarima_forecast

logger = logging.getLogger(__name__)


def backtest_models(df: pd.DataFrame, test_days: int = 30) -> dict:
    """
    Сравнение моделей на исторических данных (holdout validation).

    Args:
        df: Исходные данные
        test_days: Количество дней для тестовой выборки

    Returns:
        dict с результатами сравнения моделей
    """
    date_col, sales_col, _ = find_columns(df)

    if not date_col or not sales_col:
        return {"error": "Не найдены колонки даты или продаж", "status": "error"}

    # Подготовка данных
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Разделение на train/test
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()

    if len(train_df) < 10:
        return {"error": "Недостаточно данных для обучения", "status": "error"}

    results = {
        "status": "success",
        "test_days": test_days,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "metrics": {},
        "best_model": None,
        "predictions": {
            "date": [],
            "actual": [],
            "neuralprophet": [],
            "sarima": []
        }
    }

    # Тестируем NeuralProphet
    try:
        np_result = neuralprophet_forecast(
            df=train_df,
            periods=test_days,
            forecast_type="general"
        )
        if "error" not in np_result and "forecast" in np_result:
            np_preds = [p["forecast"] for p in np_result["forecast"]]
            results["predictions"]["neuralprophet"] = np_preds
            results["metrics"]["neuralprophet"] = _calc_metrics(
                test_df[sales_col].values,
                np_preds
            )
        else:
            results["metrics"]["neuralprophet"] = {"error": np_result.get("error", "Unknown error")}
    except Exception as e:
        logger.error(f"NeuralProphet backtest error: {e}")
        results["metrics"]["neuralprophet"] = {"error": str(e)}

    # Тестируем SARIMA
    try:
        sarima_result = sarima_forecast(
            df=train_df,
            periods=test_days,
            forecast_type="general"
        )
        if "error" not in sarima_result and "forecast" in sarima_result:
            sarima_preds = [p["forecast"] for p in sarima_result["forecast"]]
            results["predictions"]["sarima"] = sarima_preds
            results["metrics"]["sarima"] = _calc_metrics(
                test_df[sales_col].values,
                sarima_preds
            )
        else:
            results["metrics"]["sarima"] = {"error": sarima_result.get("error", "Unknown error")}
    except Exception as e:
        logger.error(f"SARIMA backtest error: {e}")
        results["metrics"]["sarima"] = {"error": str(e)}

    # Заполняем фактические значения
    results["predictions"]["date"] = test_df[date_col].dt.strftime("%Y-%m-%d").tolist()
    results["predictions"]["actual"] = test_df[sales_col].tolist()

    # Определяем лучшую модель по MAPE
    best_model = _select_best_model(results["metrics"])
    results["best_model"] = best_model

    return results


def _calc_metrics(actual: np.ndarray, predicted: list) -> dict:
    """Вычисляет метрики качества (MAE, MAPE, RMSE)."""
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    # MAE
    mae = float(np.mean(np.abs(actual - predicted)))

    # MAPE (избегаем деления на 0)
    mask = actual != 0
    if mask.any():
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    else:
        mape = 0.0

    # RMSE
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    return {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "rmse": round(rmse, 2)
    }


def _select_best_model(metrics: dict) -> str:
    """Выбирает лучшую модель по минимальному MAPE."""
    np_mape = metrics.get("neuralprophet", {}).get("mape", float("inf"))
    sarima_mape = metrics.get("sarima", {}).get("mape", float("inf"))

    # Если обе модели ошиблись — возвращаем NeuralProphet по умолчанию
    if np_mape == float("inf") and sarima_mape == float("inf"):
        return "neuralprophet"

    if np_mape <= sarima_mape:
        return "neuralprophet"
    else:
        return "sarima"


@tool
def run_backtest_tool(test_days: int = 30, session_id: Optional[str] = None) -> dict:
    """
    Запустить сравнение моделей прогнозирования (backtest).
    Сравнивает NeuralProphet и SARIMA на исторических данных,
    используя последние `test_days` как тестовую выборку.

    Args:
        test_days: Количество дней для тестирования (по умолчанию: 30)
        session_id: ID сессии (опционально, если не передан — используется 'default')

    Returns:
        dict с результатом:
        - status: "success" или "error"
        - best_model: название лучшей модели
        - metrics: метрики для каждой модели (MAE, MAPE, RMSE)
        - predictions: предсказания vs факт для визуализации
        - error: сообщение об ошибке (если есть)

    Example:
        result = run_backtest_tool(test_days=14, session_id="user_123")
        if result["status"] == "success":
            print(f"🏆 Лучшая модель: {result['best_model']}")
    """
    # Получаем менеджер сессий
    session_manager = get_session_manager()

    # Если session_id не передан — используем дефолт
    if session_id is None:
        session_id = "default"
        logger.warning("⚠️ session_id не передан, используется 'default'")

    # Получаем датасет для конкретной сессии
    df = session_manager.get_dataset(session_id)
    if df is None:
        logger.warning(f"❌ Датасет не загружен для сессии {session_id[:8]}")
        return {
            "error": "Датасет не загружен. Сначала вызовите load_dataset.",
            "status": "error"
        }

    # Валидация параметров
    if not isinstance(test_days, int):
        logger.error(f"❌ test_days должен быть int, получено: {type(test_days)}")
        return {
            "error": "test_days должен быть целым числом",
            "status": "error"
        }

    # Проверка разумного диапазона
    min_days = 7
    max_days = 90
    if test_days < min_days:
        logger.warning(f"⚠️ test_days ({test_days}) ниже минимума ({min_days})")
        test_days = min_days
    elif test_days > max_days:
        logger.warning(f"⚠️ test_days ({test_days}) выше максимума ({max_days})")
        test_days = max_days

    # Проверка достаточности данных
    if len(df) < (test_days + 10):
        logger.warning(f"⚠️ Мало данных для backtest: {len(df)} строк")
        return {
            "error": f"Недостаточно данных для backtest. Нужно хотя бы {test_days + 10} строк.",
            "status": "error"
        }

    try:
        logger.info(f"🔬 Запуск backtest на {test_days} днях для сессии {session_id[:8]}...")

        # Вызываем функцию backtest
        result = backtest_models(df=df, test_days=test_days)

        # Сохраняем результат в Redis для этой сессии
        if result.get("status") == "success":
            session_manager.redis_client.set(
                f"{session_manager.BACKTEST_KEY}{session_id}",
                json.dumps(result, ensure_ascii=False)
            )
            logger.info(f"✅ Backtest завершён, лучшая модель: {result.get('best_model')}")

        return result

    except Exception as e:
        logger.error(f"Backtest tool failed: {e}", exc_info=True)
        return {
            "error": f"Ошибка при запуске backtest: {str(e)}",
            "status": "error"
        }