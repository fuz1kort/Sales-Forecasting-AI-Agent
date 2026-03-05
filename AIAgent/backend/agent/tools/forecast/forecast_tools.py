"""Инструменты прогнозирования для AI-агента.

Тонкий слой адаптации: валидация входных данных → вызов сервиса.
"""
from __future__ import annotations
import logging
from typing import Optional, Literal

from smolagents import tool

from agent.state import get_session_manager
from services.forecast_service import get_forecast
from config.forecast_config import forecast_config
from schemas.forecast_types import ForecastResult, ForecastSummary

logger = logging.getLogger(__name__)


def _get_session_id(session_id: Optional[str]) -> str:
    """Возвращает корректный session_id: переданный или дефолтный."""
    if not session_id or not session_id.strip():
        logger.debug("⚠️ session_id not provided, using default")
        return forecast_config.DEFAULT_SESSION_ID
    return session_id.strip()


@tool
def build_forecast(
        periods: int = forecast_config.DEFAULT_PERIODS,
        model_type: Literal["neuralprophet", "sarima"] = "neuralprophet",
        forecast_type: str = "general",
        store_ids: Optional[str] = None,
        session_id: Optional[str] = None,
) -> ForecastResult:
    """
    Построить прогноз продаж с возвращением полных данных.

    Этот инструмент используется, когда нужно:
    - Построить таблицу прогноза
    - Построить график прогнозных продаж
    - Получить массив точек прогноза для анализа

    ⚠️ Подсказка для агента:
    - Этот метод должен использоваться, если пользователь спрашивает про:
        - Таблицу прогноза
        - График прогнозных продаж
      он возвращает только агрегированные метрики.

    Пример использования:
    - "Спрогнозируй продажи на N дней"
    - "Построй график прогнозных продаж"
    - "Построй прогноз с помощью ARIMA"

    Args:
        periods: Горизонт прогноза в днях (7..365, по умолчанию 30)
        model_type: "neuralprophet" | "sarima" | "auto"
        forecast_type: "general" (агрегированный) | "by_store" (по магазинам)
        store_ids: Строка с ID магазинов: "store_1, store_2"
        session_id: ID сессии пользователя

    Returns:
        ForecastResult: Словарь с прогнозом, метриками или ошибкой
    """
    session_id = _get_session_id(session_id)
    session_manager = get_session_manager()

    df = session_manager.get_dataset(session_id)
    if df is None:
        return {
            "status": "error",
            "error": "Датасет не загружен. Сначала вызовите load_dataset."
        }

    # Парсим store_ids из строки в список
    store_list = None
    if store_ids and store_ids.strip():
        store_list = [s.strip() for s in store_ids.split(",") if s.strip()]

    try:
        result = get_forecast(
            df=df,
            model_type=model_type,
            periods=periods,
            forecast_type=forecast_type,
            store_ids=store_list,
        )

        # Кэшируем успешный результат
        if result.get("status") != "error" and "forecast" in result:
            if forecast_config.CACHE_FORECAST:
                session_manager.set_forecast(session_id, result)
                logger.debug(f"💾 Forecast cached for session {session_id[:8]}")

        return result

    except Exception as e:
        logger.error(f"❌ Tool build_forecast crashed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Tool error: {type(e).__name__}: {e}"
        }


@tool
def get_forecast_summary(
        session_id: Optional[str] = None
) -> ForecastSummary:
    """
    Получить краткую сводку по последнему прогнозу.

    ⚠️ Подсказка для агента:
    - Этот инструмент используется только для человекочитаемых ответов
      (total_forecast, average_daily, min/max, даты).
    - Для построения графиков и таблиц использовать build_forecast.

    Для таблиц и графиков используйте build_forecast.

    Args:
        session_id: ID сессии (опционально)

    Returns:
        ForecastSummary: Агрегированные метрики прогноза
    """
    session_id = _get_session_id(session_id)
    session_manager = get_session_manager()

    forecast = session_manager.get_forecast_by_session(session_id)

    if not forecast or "forecast" not in forecast:
        return {
            "status": "no_forecast",
            "periods": 0,
            "total_forecast": 0,
            "average_daily": 0,
            "min_daily": 0,
            "max_daily": 0,
            "first_date": "",
            "last_date": "",
        }

    fc_data = forecast["forecast"]
    if not fc_data:
        return {
            "status": "empty",
            "periods": 0,
            "total_forecast": 0,
            "average_daily": 0,
            "min_daily": 0,
            "max_daily": 0,
            "first_date": "",
            "last_date": "",
        }

    values = [r.get("forecast", 0) for r in fc_data if isinstance(r.get("forecast"), (int, float))]
    if not values:
        values = [0]

    total = sum(values)
    count = len(values)

    return {
        "status": "success",
        "periods": count,
        "total_forecast": round(total, 2),
        "average_daily": round(total / count, 2),
        "min_daily": round(min(values), 2),
        "max_daily": round(max(values), 2),
        "first_date": fc_data[0].get("date", ""),
        "last_date": fc_data[-1].get("date", ""),
    }