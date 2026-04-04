"""Инструмент backtest для сравнения моделей прогнозирования."""
import logging
from typing import Optional
import json

import pandas as pd
import numpy as np
from smolagents import tool

from backend.agent.state import get_session_manager
from backend.utils import find_columns
from backend.models import sarima_forecast

logger = logging.getLogger(__name__)

@tool
def run_backtest(
        test_days: int = 30,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        store_ids: Optional[str] = None,
        location_ids: Optional[str] = None,
        session_id: Optional[str] = None
) -> dict:
    """
    Сравнить модели прогнозирования на исторических данных.

    Вызывайте, когда нужно:
    - "какая модель лучше для моих данных?"
    - "сравни модели прогнозирования"
    - "проверь точность прогноза"

    Алгоритм:
    1. Берёт последние test_days как тестовую выборку
    2. Обучает модели на оставшихся данных
    3. Сравнивает метрики (MAPE, MAE, RMSE)
    4. Возвращает лучшую модель и детали

    Args:
        test_days: Количество дней для тестовой выборки (7..90)
        date_from: Начальная дата для обучения (YYYY-MM-DD)
        date_to: Конечная дата для обучения (YYYY-MM-DD)
        store_ids: ID магазинов через запятую (например "1,2,3")
        location_ids: Фильтр по стране/регионам (например "USA,UK")
        session_id: ID сессии (опционально)

    Returns:
        dict с результатом:
        - status: "success" | "error"
        - best_model: название лучшей модели
        - metrics: метрики для каждой модели
        - predictions: предсказания vs факт
        - error: сообщение об ошибке
    """
    session_manager = get_session_manager()
    sid = session_id or "default"

    df = session_manager.get_dataset(sid)
    if df is None:
        return {
            "status": "error",
            "error": "Датасет не загружен. Сначала вызовите load_dataset."
        }

    # Применяем фильтры
    filtered_df = df.copy()

    # Фильтр по датам
    date_col, sales_col, store_col, product_col = find_columns(df)
    if date_col and (date_from or date_to):
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
        if date_from:
            filtered_df = filtered_df[filtered_df[date_col] >= pd.to_datetime(date_from)]
        if date_to:
            filtered_df = filtered_df[filtered_df[date_col] <= pd.to_datetime(date_to)]

    # Фильтр по магазинам
    if store_ids and store_ids.lower() != "all" and store_col:
        store_list = [s.strip() for s in store_ids.split(',')]
        filtered_df = filtered_df[filtered_df[store_col].astype(str).isin(store_list)]

    # Фильтр по локации (страна/регион - дополнительный параметр)
    if location_ids and location_ids.lower() != "all" and store_col:
        location_list = [s.strip() for s in location_ids.split(',')]
        filtered_df = filtered_df[filtered_df[store_col].astype(str).isin(location_list)]

    if filtered_df.empty:
        return {
            "status": "error",
            "error": "Нет данных после применения фильтров."
        }

    # Валидация параметров
    test_days = max(7, min(90, int(test_days)))

    if len(filtered_df) < test_days + 10:
        return {
            "status": "error",
            "error": f"Недостаточно данных. Нужно ≥{test_days + 10} строк, есть {len(filtered_df)}"
        }

    if not date_col or not sales_col:
        return {"status": "error", "error": "Не найдены колонки даты или продаж"}

    try:
        logger.info(f"🔬 Запуск backtest: {test_days} тестовых дней")

        # Подготовка данных
        filtered_df = filtered_df.copy()
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
        filtered_df = filtered_df.sort_values(date_col).reset_index(drop=True)

        train_df = filtered_df.iloc[:-test_days].copy()
        test_df = filtered_df.iloc[-test_days:].copy()

        results = {
            "status": "success",
            "test_days": test_days,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "metrics": {},
            "best_model": None,
        }

        # Тестируем SARIMA
        sarima_result = _test_model(
            "sarima", sarima_forecast,
            train_df, test_df, sales_col, test_days
        )
        results["metrics"]["sarima"] = sarima_result

        # Выбираем лучшую по MAPE
        results["best_model"] = _select_best(results["metrics"])

        # Добавляем фактические значения для визуализации
        results["actual_values"] = test_df[sales_col].tolist()
        results["dates"] = test_df[date_col].dt.strftime("%Y-%m-%d").tolist()

        # Кэшируем результат
        session_manager.redis_client.set(
            f"backtest:{sid}",
            json.dumps(results, ensure_ascii=False, default=str)
        )

        logger.info(f"✅ Backtest завершён. Лучшая: {results['best_model']}")
        return results

    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Backtest error: {type(e).__name__}: {e}"
        }


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

        # Извлекаем предсказания
        predicted = [p["forecast"] for p in result["forecast"][:len(test_df)]]
        actual = test_df[sales_col].values[:len(predicted)]

        return _calc_metrics(actual, predicted)

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


def _select_best(metrics: dict) -> str:
    """Выбирает модель с минимальным MAPE."""
    np_mape = metrics.get("neuralprophet", {}).get("mape", float("inf"))
    sarima_mape = metrics.get("sarima", {}).get("mape", float("inf"))

    if np_mape == float("inf") and sarima_mape == float("inf"):
        return "neuralprophet"  # fallback

    return "neuralprophet" if np_mape <= sarima_mape else "sarima"