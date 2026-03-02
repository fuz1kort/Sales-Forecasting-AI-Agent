"""
Инструменты для прогнозирования продаж.

Используют модели NeuralProphet и ARIMA для построения
прогнозов на основе временных рядов.
"""

import logging
from typing import Optional

from smolagents import tool

from salesforecasting.app.backend.services.forecast import get_forecast

logger = logging.getLogger(__name__)


@tool
def build_forecast(
        periods: int = 30,
        model_type: str = "neuralprophet",
        forecast_type: str = "general",
        store_ids: Optional[str] = None
) -> dict:
    """
    Построить прогноз продаж на указанный горизонт.

    Поддерживает два типа моделей:
    - neuralprophet: нейросетевая модель с учётом сезонности
    - arima: классическая статистическая модель

    Args:
        periods: Горизонт прогноза в днях (по умолчанию: 30)
        model_type: Тип модели: "neuralprophet" или "arima"
        forecast_type: "general" (общий) или "by_store" (по магазинам)
        store_ids: Список ID магазинов через запятую (опционально)

    Returns:
        Словарь с результатом:
        - forecast: список точек прогноза [{"date": str, "forecast": float}, ...]
        - metrics: метрики качества модели (если доступны)
        - error: сообщение об ошибке (если есть)

    Example:
        result = build_forecast(
            periods=90,
            model_type="neuralprophet",
            forecast_type="by_store",
            store_ids="store_1,store_2"
        )
    """
    from salesforecasting.app.backend.agent.state import get_current_dataset, get_global_state

    # Получаем датасет из глобального состояния
    df = get_current_dataset()
    if df is None:
        return {
            "error": "Датасет не загружен. Сначала вызовите load_dataset.",
            "status": "error"
        }

    # Парсим список магазинов
    store_list = None
    if store_ids and store_ids.strip():
        store_list = [s.strip() for s in store_ids.split(",")]

    try:
        # Вызываем сервис прогнозирования
        result = get_forecast(
            df=df,
            model_type=model_type,
            periods=periods,
            forecast_type=forecast_type,
            store_ids=store_list
        )

        # Сохраняем результат в глобальное состояние для последующего анализа
        if "forecast" in result:
            state = get_global_state()
            state["last_forecast"] = result

        return result

    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        return {
            "error": f"Ошибка при построении прогноза: {str(e)}",
            "status": "error"
        }


@tool
def get_forecast_summary() -> dict:
    """
    Получить краткое резюме последнего построенного прогноза.

    Returns:
        Словарь с ключевыми метриками прогноза
    """
    from salesforecasting.app.backend.agent.state import get_global_state

    state = get_global_state()
    forecast = state.get("last_forecast")

    if not forecast or "forecast" not in forecast:
        return {"status": "no_forecast", "message": "Прогноз ещё не построен"}

    fc_data = forecast["forecast"]
    if not fc_data:
        return {"status": "empty", "message": "Прогноз пуст"}

    # Считаем агрегаты
    values = [r.get("forecast", 0) for r in fc_data]
    total = sum(values)
    avg = total / len(values) if values else 0
    min_val = min(values)
    max_val = max(values)

    return {
        "status": "success",
        "periods": len(fc_data),
        "total_forecast": round(total, 2),
        "average_daily": round(avg, 2),
        "min_daily": round(min_val, 2),
        "max_daily": round(max_val, 2),
        "first_date": fc_data[0].get("date"),
        "last_date": fc_data[-1].get("date")
    }