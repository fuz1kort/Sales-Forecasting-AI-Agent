"""
Анализ товаров, трендов и общих выводов.
"""

import logging
import pandas as pd
from pandas import Timestamp, Period

from smolagents import tool
from utils import find_columns, get_product_column
from agent.state import get_current_dataset

logger = logging.getLogger(__name__)


@tool
def analyze_top_products_tool(limit: int = 10) -> str:
    """
    Проанализировать топ товаров по объёму продаж.

    Args:
        limit: Количество товаров в топе (по умолчанию: 10)

    Returns:
        Отформатированная строка с результатом анализа
    """
    df = get_current_dataset()

    if df is None:
        raise ValueError("Датасет не загружен. Сначала загрузите данные через load_dataset().")

    _, sales_col, _ = find_columns(df)
    product_col = get_product_column(df)

    if not product_col or not sales_col:
        raise ValueError(
            f"Не найдены обязательные колонки. "
            f"Найдено: sales={sales_col}, product={product_col}. "
            f"Проверьте названия столбцов в CSV."
        )

    # Агрегация по товарам
    top_products = (
        df.groupby(product_col)[sales_col]
        .agg(["sum", "count", "mean"])
        .round(2)
    )
    top_products = top_products.sort_values("sum", ascending=False).head(limit)

    analysis = f"📊 **ТОП-{limit} ТОВАРОВ ПО ПРОДАЖАМ**\n\n"
    analysis += f"На основе {len(df):,} транзакций:\n\n"

    for i, (product, data) in enumerate(top_products.iterrows(), 1):
        analysis += f"{i}. **{product}**\n"
        analysis += f"   • Общий объём: ${data['sum']:,.2f}\n"
        analysis += f"   • Заказов: {int(data['count'])}\n"
        analysis += f"   • Средний чек: ${data['mean']:,.2f}\n\n"

    total_revenue = df[sales_col].sum()
    top_revenue = top_products["sum"].sum()
    percentage = (top_revenue / total_revenue * 100) if total_revenue > 0 else 0

    analysis += "💡 **Выводы:**\n"
    analysis += f"• Топ-{limit} товаров дают ${top_revenue:,.2f} ({percentage:.1f}% выручки)\n"
    if len(top_products) > 0:
        analysis += f"• Лидер: {top_products.index[0]} (${top_products.iloc[0]['sum']:,.2f})"

    return analysis


@tool
def analyze_trends_tool(period: str = "monthly") -> str:
    """
    Проанализировать тренды продаж во времени.

    Args:
        period: Период агрегации: daily/weekly/monthly/quarterly

    Returns:
        Отформатированная строка с анализом трендов
    """
    df = get_current_dataset()

    if df is None:
        raise ValueError("Датасет не загружен. Сначала загрузите данные через load_dataset().")

    date_col, sales_col, _ = find_columns(df)
    if not date_col or not sales_col:
        raise ValueError("Нужны колонки с датой и продажами для анализа трендов.")

    # Подготовка данных
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Маппинг периодов для pandas
    period_map = {
        "daily": "D",
        "weekly": "W-MON",
        "monthly": "ME",
        "quarterly": "QE"
    }
    freq = period_map.get(period.lower(), "ME")

    # Группировка по периоду
    grouped = df_copy.groupby(pd.Grouper(key=date_col, freq=freq))[sales_col].sum()

    analysis = f"📈 **ТРЕНДЫ ПРОДАЖ ({period.upper()})**\n\n"

    # Показываем последние 12 периодов
    recent = grouped.tail(12)

    for period_label, sales in recent.items():
        label = _format_period_label(period_label, period.lower())
        analysis += f"• {label}: ${sales:,.2f}\n"

    # Анализ роста, если достаточно данных
    if len(grouped) >= 2:
        prev_val = grouped.iloc[-2]
        curr_val = grouped.iloc[-1]
        if prev_val > 0:
            growth = ((curr_val - prev_val) / prev_val) * 100
            analysis += f"\n📊 **Динамика:** {growth:+.1f}% к предыдущему периоду\n"

        best_idx = grouped.idxmax()
        worst_idx = grouped.idxmin()
        best_label = _format_period_label(best_idx, period.lower())
        worst_label = _format_period_label(worst_idx, period.lower())

        analysis += f"• Пик продаж: {best_label} (${grouped.max():,.2f})\n"
        analysis += f"• Минимум: {worst_label} (${grouped.min():,.2f})"

    return analysis


@tool
def analyze_general_tool() -> str:
    """
    Получить общие выводы по датасету.

    Returns:
        Отформатированная строка с выводами
    """
    df = get_current_dataset()

    if df is None:
        raise ValueError("Датасет не загружен. Сначала загрузите данные через load_dataset().")

    date_col, sales_col, store_col = find_columns(df)
    product_col = get_product_column(df)

    analysis = "📋 **ОБЩИЕ ВЫВОДЫ ПО ДАННЫМ**\n\n"

    # Базовая информация
    analysis += f"📊 **Размер датасета:**\n"
    analysis += f"  • Строк: {len(df):,}\n"
    analysis += f"  • Колонок: {len(df.columns)}\n\n"

    # Информация о периоде
    if date_col:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        min_date = df_copy[date_col].min()
        max_date = df_copy[date_col].max()
        if pd.notna(min_date) and pd.notna(max_date):
            days = (max_date - min_date).days
            analysis += f"📅 **Период данных:**\n"
            analysis += f"  • От: {min_date.strftime('%Y-%m-%d')}\n"
            analysis += f"  • До: {max_date.strftime('%Y-%m-%d')}\n"
            analysis += f"  • Дней: {days}\n\n"

    # Информация о продажах
    if sales_col:
        analysis += f"💰 **Продажи:**\n"
        analysis += f"  • Общая сумма: ${df[sales_col].sum():,.2f}\n"
        analysis += f"  • Средняя: ${df[sales_col].mean():,.2f}\n\n"

    # Информация о структуре
    if product_col:
        unique_products = df[product_col].nunique()
        analysis += f"🏷️ **Товары:** {unique_products:,} уникальных\n"

    if store_col:
        unique_stores = df[store_col].nunique()
        analysis += f"🏪 **Магазины:** {unique_stores:,} уникальных\n"

    analysis += "\n✅ **Статус:** Датасет готов к анализу и прогнозированию"

    return analysis


def _format_period_label(period_label, period_type: str) -> str:
    """Форматирует метку периода для отображения."""
    if isinstance(period_label, (Timestamp, Period)):
        if period_type == "daily":
            return period_label.strftime("%Y-%m-%d")
        elif period_type == "weekly":
            return f"W{period_label.isocalendar()[1]:02d}-{period_label.year}"
        elif period_type == "monthly":
            return period_label.strftime("%b %Y").lower()
        elif period_type == "quarterly":
            return f"Q{period_label.quarter} {period_label.year}"
    return str(period_label)

