"""
Сервис прогнозирования продаж.

Отвечает за:
- Валидацию входных данных
- Нормализацию параметров
- Выбор модели (включая auto-режим)
- Вызов конкретных реализаций (Prophet / SARIMA / CatBoost)
- Обработку ошибок и логирование

Это «чистый» слой: не зависит от фреймворков (FastAPI, smolagents),
поэтому легко тестируется юнит-тестами.
"""
import logging
from typing import List, Optional

import pandas as pd

from backend.services.backtest_service import backtest_models
from backend.config.forecast_config import forecast_config
from backend.models import sarima_forecast, catboost_forecast
from backend.models.prophet_model import prophet_forecast
from backend.models.catboost_model import ensemble_forecast_optimized
from backend.schemas.forecast_types import ForecastResult
from backend.utils import find_columns, smape

logger = logging.getLogger(__name__)


def _normalize_store_ids(store_ids: Optional[List[str | int]]) -> Optional[List[str]]:
    """
    Приводит список ID магазинов к единому формату: список строк.

    Args:
        store_ids: Список строк или чисел, или None

    Returns:
        Список строк или None, если вход пустой

    Example:
        >>> _normalize_store_ids([1, "store_2", " 3 "])
        ['1', 'store_2', '3']
    """
    if not store_ids:  # Пустой список, None или пустая строка
        return None

    # Фильтруем пустые значения, приводим к строке, убираем пробелы
    normalized = [str(s).strip() for s in store_ids if str(s).strip()]

    return normalized if normalized else None


def _validate_dataframe(df: pd.DataFrame) -> Optional[str]:
    """
    Проверяет, что DataFrame пригоден для прогнозирования.

    Args:
        df: Входной DataFrame с данными о продажах

    Returns:
        Строку с описанием ошибки или None, если всё ок
    """
    # Проверка на пустоту
    if df is None or df.empty:
        return "Пустой DataFrame: нет данных для прогноза."

    # Проверка наличия столбца с продажами
    # find_columns возвращает (date_col, sales_col, store_col, product_col)
    _, sales_col, _, _ = find_columns(df)

    if not sales_col:
        return (
            f"Не найден столбец продаж. "
            f"Доступные столбцы: {list(df.columns)}"
        )

    return None  # Всё в порядке


def _resolve_model_type(model_type: str) -> str:
    """
    Нормализует название модели: приводит к нижнему регистру,
    заменяет алиасы на канонические названия.

    Args:
        model_type: Название модели от пользователя

    Returns:
        Каноническое название: "prophet" | "sarima" | "auto"

    Example:
        >>> _resolve_model_type("NP")
        'neuralprophet'
        >>> _resolve_model_type("best")
        'auto'
    """
    model = model_type.strip().lower()

    # Если есть алиас — заменяем на основное название
    return forecast_config.MODEL_ALIASES.get(model, model)


def select_best_model(df: pd.DataFrame, periods: int) -> str:
    """
    Выбирает лучшую модель через backtest на исторических данных.

    Алгоритм:
    1. Берём последние N дней как тестовую выборку
    2. Запускаем backtest для всех моделей
    3. Выбираем модель с минимальным MAPE (или MAE как tie-breaker)

    Args:
        df: Исторические данные
        periods: Горизонт прогноза (влияет на размер тестовой выборки)

    Returns:
        Название выбранной модели
    """
    # Адаптивный размер тестовой выборки: чем больше горизонт — тем больше тест
    test_days = max(
        forecast_config.AUTO_BACKTEST_MIN_DAYS,
        min(forecast_config.AUTO_BACKTEST_MAX_DAYS, periods)
    )

    logger.debug(f"🔍 Running backtest with {test_days} test days")

    result = backtest_models(df, test_days=test_days)

    # Если backtest упал — фоллбэк на Prophet (обычно стабильнее)
    if isinstance(result, dict) and result.get("error"):
        logger.warning(f"⚠️ Backtest failed: {result['error']}. Fallback to prophet")
        return "prophet"

    # Извлекаем название лучшей модели
    best_model = (result or {}).get("best_model", "prophet")
    logger.info(f"🎯 Auto-selected model: {best_model}")

    return best_model


def get_forecast(
        df: pd.DataFrame,
        model_type: str = "prophet",
        periods: int = forecast_config.DEFAULT_PERIODS,
        forecast_type: str = "general",
        store_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
) -> ForecastResult:
    """
    Единая точка входа для построения прогноза.

    Это фасад: клиент вызывает эту функцию, не зная деталей
    реализации моделей.

    Args:
        df: DataFrame с историческими данными
        model_type: "prophet" | "sarima" | "auto" (с алиасами)
        periods: Горизонт прогноза в днях (автоматически ограничивается)
        forecast_type: "general" (агрегированный) или "by_store" (по магазинам)
        store_ids: Список ID магазинов для фильтрации (опционально)
        start_date: Начало диапазона прогноза (для Prophet)
        end_date: Конец диапазона прогноза (для Prophet)

    Returns:
        ForecastResult: Словарь с прогнозом, метриками или ошибкой

    Raises:
        Не выбрасывает исключения — все ошибки возвращаются в поле "error"
    """

    # === Шаг 1: Валидация входных данных ===
    error = _validate_dataframe(df)
    if error:
        return {"status": "error", "error": error}

    # === Шаг 2: Нормализация параметров ===
    model_type = _resolve_model_type(model_type)

    # Ограничиваем горизонт прогноза допустимыми значениями
    periods = max(forecast_config.MIN_PERIODS, min(forecast_config.MAX_PERIODS, periods))

    store_ids_norm = _normalize_store_ids(store_ids)

    # === Шаг 3: Выбор модели (для auto-режима) ===
    if model_type == "auto":
        model_type = select_best_model(df, periods)

    # === Шаг 4: Вызов реализации модели ===
    try:
        logger.info(
            f"🔮 Building forecast: model={model_type}, periods={periods}, "
            f"type={forecast_type}, stores={store_ids_norm}"
        )

        if model_type == "sarima":
            # SARIMA не поддерживает start_date/end_date — только periods
            result = sarima_forecast(
                df=df,
                periods=periods,
                forecast_type=forecast_type,
                store_ids=store_ids_norm,
            )
        elif model_type == "prophet":
            result = prophet_forecast(
                df=df,
                periods=periods,
                forecast_type=forecast_type,
                store_ids=store_ids_norm,
                start_date=start_date,
                end_date=end_date,
            )
        elif model_type == "ensemble":
            result = ensemble_forecast_optimized(
                df=df,
                periods=periods,
                forecast_type=forecast_type,
                store_ids=store_ids_norm,
            )
        else:
            logger.warning(f"⚠️ Unknown model_type '{model_type}', fallback to ensemble")
            result = ensemble_forecast_optimized(
                df=df,
                periods=periods,
                forecast_type=forecast_type,
                store_ids=store_ids_norm,
            )

        # === Шаг 5: Обогащение результата мета-информацией ===
        if isinstance(result, dict) and result.get("status") != "error":
            # Добавляем информацию о том, какая модель реально использовалась
            info = result.get("info", "")
            if isinstance(info, str):
                # Если info - строка, создаем словарь
                result["info"] = {
                    "description": info,
                    "model_used": model_type,
                    "periods_requested": periods,
                    "forecast_type": forecast_type,
                }
            elif isinstance(info, dict):
                # Если info - словарь, обновляем его
                info.update({
                    "model_used": model_type,
                    "periods_requested": periods,
                    "forecast_type": forecast_type,
                })
                result["info"] = info

        return result  # type: ignore[return-value]

    except Exception as e:
        # Ловим все исключения — не даём упасть сервису
        logger.error(f"❌ Forecast service failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Service error: {type(e).__name__}: {e}"
        }

