"""
Инструменты для прогнозирования продаж.

Используют модели NeuralProphet и ARIMA для построения
прогнозов на основе временных рядов.
"""
import logging
from typing import Optional

from smolagents import tool

from agent.state import get_session_manager
from services.forecast import get_forecast

logger = logging.getLogger(__name__)


@tool
def build_forecast(
        periods: int = 30,
        model_type: str = "neuralprophet",
        forecast_type: str = "general",
        store_ids: Optional[str] = None,
        session_id: Optional[str] = None
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
        session_id: ID сессии (опционально)

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
            store_ids="store_1,store_2",
            session_id="user_123"
        )
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
    errors = []

    if not isinstance(periods, int):
        errors.append("periods должен быть целым числом")
    else:
        if periods < 7:
            periods = 7
        elif periods > 365:
            periods = 365

    valid_models = ["neuralprophet", "sarima", "auto"]
    if model_type not in valid_models:
        errors.append(f"model_type должен быть одним из: {valid_models}")

    valid_types = ["general", "by_store"]
    if forecast_type not in valid_types:
        errors.append(f"forecast_type должен быть одним из: {valid_types}")

    if errors:
        return {"error": "; ".join(errors), "status": "error"}

    # Парсим список магазинов
    store_list = None
    if store_ids and store_ids.strip():
        store_list = [s.strip() for s in store_ids.split(",")]

    try:
        logger.info(f"🔮 Построение прогноза: {model_type}, {periods} дней, {forecast_type}")

        # Вызываем сервис прогнозирования
        result = get_forecast(
            df=df,
            model_type=model_type,
            periods=periods,
            forecast_type=forecast_type,
            store_ids=store_list
        )

        # Сохраняем результат в Redis для этой сессии
        if "forecast" in result and result.get("status") != "error":
            session_manager.set_forecast(session_id, result)
            logger.debug(f"💾 Прогноз сохранён для сессии {session_id[:8]}")

        return result

    except Exception as e:
        logger.error(f"Forecast failed: {e}", exc_info=True)
        return {
            "error": f"Ошибка при построении прогноза: {str(e)}",
            "status": "error"
        }


@tool
def get_forecast_summary(session_id: Optional[str] = None) -> dict:
    """
    Получить краткое резюме последнего построенного прогноза.

    Args:
        session_id: ID сессии (опционально)

    Returns:
        Словарь с ключевыми метриками прогноза
    """

    # Получаем менеджер сессий
    session_manager = get_session_manager()

    # Если session_id не передан — используем дефолт
    if session_id is None:
        session_id = "default"

    # Получаем прогноз из Redis для этой сессии
    forecast = session_manager.get_forecast(session_id)

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