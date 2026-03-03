"""
Инструмент backtest для сравнения моделей прогнозирования.

Позволяет агенту автономно запускать сравнение моделей
на исторических данных.
"""
import logging

from smolagents import tool

from models import backtest_models

logger = logging.getLogger(__name__)


@tool
def run_backtest_tool(test_days: int = 30) -> dict:
    """
    Запустить сравнение моделей прогнозирования (backtest).

    Сравнивает NeuralProphet и SARIMA на исторических данных,
    используя последние `test_days` как тестовую выборку.

    Args:
        test_days: Количество дней для тестирования (по умолчанию: 30)

    Returns:
        dict с результатом:
        - status: "success" или "error"
        - best_model: название лучшей модели
        - metrics: метрики для каждой модели (MAE, MAPE, RMSE)
        - predictions: предсказания vs факт для визуализации
        - error: сообщение об ошибке (если есть)

    Example:
        result = run_backtest_tool(test_days=14)
        if result["status"] == "success":
            print(f"🏆 Лучшая модель: {result['best_model']}")
    """
    from agent.state import get_current_dataset

    # Получаем датасет из глобального состояния
    df = get_current_dataset()
    if df is None:
        return {
            "error": "Датасет не загружен. Сначала вызовите load_dataset.",
            "status": "error"
        }

    try:
        # Вызываем сервис backtest
        result = backtest_models(df=df, test_days=test_days)

        # Сохраняем в глобальное состояние для frontend
        if result.get("status") == "success":
            from agent.state import get_global_state
            get_global_state()["last_backtest"] = result

        return result

    except Exception as e:
        logger.error(f"Backtest tool failed: {e}")
        return {
            "error": f"Ошибка при запуске backtest: {str(e)}",
            "status": "error"
        }
