"""Инструменты для анализа стационарности временных рядов."""
from __future__ import annotations
import logging
from typing import Optional

from smolagents import tool

from backend.agent.state import get_session_manager
from backend.utils import adf_test

logger = logging.getLogger(__name__)


@tool
def analyze_stationarity_tool(session_id: Optional[str] = None) -> str:
    """
    Анализирует стационарность временного ряда с помощью теста ADF.

    Этот инструмент проверяет, является ли ряд стационарным (не имеет тренда/сезонности).
    Полезно для понимания свойств данных перед прогнозированием.

    Args:
        session_id: ID сессии пользователя

    Returns:
        Текст с результатами теста ADF
    """
    session_manager = get_session_manager()
    df = session_manager.get_dataset(session_id)

    if df is None:
        return "❌ Данные не загружены. Сначала загрузите CSV."

    from utils import find_columns
    date_col, sales_col, store_col, product_col = find_columns(df)

    if not sales_col:
        return "❌ Не найден столбец продаж."

    # Агрегация по дням
    df_agg = df.groupby(date_col)[sales_col].sum().reset_index()
    series = df_agg[sales_col]

    result = adf_test(series, title="Тест ADF для продаж")

    response = f"🧪 **Тест ADF (стационарность ряда)**\n\n"
    response += f"📊 **Результаты:**\n"
    response += f"- Test Statistic: {result['Test Statistic']:.4f}\n"
    response += f"- p-value: {result['p-value']:.4f}\n"
    response += f"- Lags Used: {result['Lags Used']}\n"
    response += f"- Number of Observations: {result['Number of Observations']}\n"
    response += f"- Critical Values: {result['Critical Values']}\n\n"
    response += f"📋 **Вывод:** Ряд **{result['Conclusion']}**\n"
    if result['p-value'] < 0.05:
        response += "✅ p-value < 0.05 → Ряд стационарный (нет тренда/сезонности).\n"
    else:
        response += "⚠️ p-value >= 0.05 → Ряд нестационарный (есть тренд/сезонность).\n"
        response += "Модели вроде Prophet/SARIMA справляются с этим автоматически."

    return response