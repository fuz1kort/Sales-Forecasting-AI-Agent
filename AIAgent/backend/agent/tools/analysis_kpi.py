"""
Анализ основных метрик и KPI продаж.
"""

import logging
import pandas as pd

from smolagents import tool
from utils import find_columns
from agent.state import get_current_dataset

logger = logging.getLogger(__name__)


@tool
def analyze_kpi_tool() -> str:
    """
    Получить дашборд ключевых показателей эффективности (KPI).

    Returns:
        Отформатированная строка с KPI метриками
    """
    df = get_current_dataset()

    if df is None:
        raise ValueError("Датасет не загружен. Сначала загрузите данные через load_dataset().")

    _, sales_col, store_col = find_columns(df)

    if not sales_col:
        raise ValueError("Не найдена колонка продаж.")

    # Базовые метрики
    total_sales = df[sales_col].sum()
    avg_sales = df[sales_col].mean()
    min_sales = df[sales_col].min()
    max_sales = df[sales_col].max()
    transactions = len(df)

    kpi_text = "📊 **ДАШБОРД KPI**\n\n"
    kpi_text += f"💰 **Продажи:**\n"
    kpi_text += f"  • Общая сумма: ${total_sales:,.2f}\n"
    kpi_text += f"  • Средний чек: ${avg_sales:,.2f}\n"
    kpi_text += f"  • Минимум: ${min_sales:,.2f}\n"
    kpi_text += f"  • Максимум: ${max_sales:,.2f}\n\n"

    kpi_text += f"📈 **Объём:**\n"
    kpi_text += f"  • Транзакций: {transactions:,}\n"

    if store_col:
        stores = df[store_col].nunique()
        kpi_text += f"  • Магазинов: {stores}\n"

    return kpi_text


@tool
def analyze_seasonality_tool() -> str:
    """
    Проанализировать сезонность продаж.

    Returns:
        Отформатированная строка с анализом сезонности
    """
    df = get_current_dataset()

    if df is None:
        raise ValueError("Датасет не загружен. Сначала загрузите данные через load_dataset().")

    date_col, sales_col, _ = find_columns(df)

    if not date_col or not sales_col:
        raise ValueError("Нужны колонки с датой и продажами для анализа сезонности.")

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Анализируем по месяцам
    df_copy['month'] = df_copy[date_col].dt.month
    monthly = df_copy.groupby('month')[sales_col].mean().round(2)

    analysis = "🌟 **АНАЛИЗ СЕЗОННОСТИ**\n\n"

    months = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
              "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

    for month_num, avg_sales in monthly.items():
        month_name = months[month_num - 1] if month_num <= 12 else f"Месяц {month_num}"
        bar_length = int((avg_sales / monthly.max()) * 20) if monthly.max() > 0 else 0
        bar = "█" * bar_length + "░" * (20 - bar_length)
        analysis += f"• {month_name}: {bar} ${avg_sales:,.2f}\n"

    # Выводы о сезонности
    peak_month = monthly.idxmax()
    low_month = monthly.idxmin()
    variance = monthly.std()

    peak_name = months[peak_month - 1] if peak_month <= 12 else f"Месяц {peak_month}"
    low_name = months[low_month - 1] if low_month <= 12 else f"Месяц {low_month}"

    analysis += f"\n💡 **Выводы:**\n"
    analysis += f"• Пик продаж: {peak_name} (${monthly.max():,.2f})\n"
    analysis += f"• Минимум: {low_name} (${monthly.min():,.2f})\n"

    if variance > monthly.mean() * 0.3:
        analysis += "• 🔴 Ярко выраженная сезонность (вариация > 30%)\n"
    elif variance > monthly.mean() * 0.15:
        analysis += "• 🟡 Умеренная сезонность (вариация 15-30%)\n"
    else:
        analysis += "• 🟢 Слабая сезонность (вариация < 15%)\n"

    return analysis

