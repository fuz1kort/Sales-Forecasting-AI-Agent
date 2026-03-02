"""
Инструменты аналитики продаж.

Обёртки над существующими функциями анализа для использования агентом.
Все тулзы возвращают ТОЛЬКО строку (str) — так требует smolagents.
"""

import logging
import pandas as pd
from pandas import Timestamp, Period

from smolagents import tool

# Импортируем существующие функции
from salesforecasting.app.backend import sales_analysis
from salesforecasting.app.backend.utils import find_columns
from salesforecasting.app.backend.agent.state import get_current_dataset

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

    date_col, sales_col, store_col = find_columns(df)

    # Поиск колонки товара
    product_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['product', 'item', 'товар', 'продукт', 'sku', 'name']):
            product_col = col
            break

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

    return analysis  # ← ТОЛЬКО строка, без обёрток


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


def _format_period_label(period_label, period_type: str) -> str:
    """Безопасно форматирует метку периода для отображения."""
    if isinstance(period_label, Timestamp):
        if period_type == "daily":
            return period_label.strftime("%d.%m.%Y")
        elif period_type == "weekly":
            return f"Неделя {period_label.isocalendar()[1]:02d}/{period_label.year}"
        elif period_type == "monthly":
            return period_label.strftime("%B %Y")
        elif period_type == "quarterly":
            quarter = (period_label.month - 1) // 3 + 1
            return f"Q{quarter} {period_label.year}"
        return period_label.strftime("%Y-%m")

    if isinstance(period_label, Period):
        if period_type == "monthly":
            return period_label.strftime("%B %Y")
        elif period_type == "quarterly":
            return f"Q{period_label.quarter} {period_label.year}"
        return str(period_label)

    if hasattr(period_label, "strftime"):
        try:
            return period_label.strftime("%Y-%m")
        except Exception:
            pass

    return str(period_label)


@tool
def analyze_kpi_tool() -> str:
    """
    Сформировать дашборд с ключевыми метриками (KPI).

    Returns:
        Отформатированная строка с KPI-дашбордом
    """

    df = get_current_dataset()

    if df is None:
        raise ValueError("Датасет не загружен. Сначала загрузите данные через load_dataset().")

    date_col, sales_col, store_col = find_columns(df)

    # Автопоиск дополнительных колонок
    product_col = customer_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['product', 'item', 'товар', 'sku']):
            product_col = col
        elif any(kw in col_lower for kw in ['customer', 'client', 'клиент']):
            customer_col = col

    result = sales_analysis.analyze_performance_metrics(
        df, sales_col, date_col, product_col, customer_col
    )

    # Если sales_analysis возвращает dict с "answer", извлекаем строку
    if isinstance(result, dict) and "answer" in result:
        return result["answer"]

    # Если уже строка — возвращаем как есть
    if isinstance(result, str):
        return result

    # Fallback: конвертируем в строку
    return str(result)


@tool
def analyze_seasonality_tool() -> str:
    """
    Проанализировать сезонные паттерны в продажах.

    Returns:
        Отформатированная строка с анализом сезонности
    """

    df = get_current_dataset()

    if df is None:
        raise ValueError("Датасет не загружен. Сначала загрузите данные через load_dataset().")

    date_col, sales_col, _ = find_columns(df)
    if not date_col or not sales_col:
        raise ValueError("Нужны колонки с датой и продажами для анализа сезонности.")

    result = sales_analysis.analyze_seasonality(df, sales_col, date_col)

    # Нормализация возврата (dict → str)
    if isinstance(result, dict) and "answer" in result:
        return result["answer"]
    if isinstance(result, str):
        return result
    return str(result)


@tool
def analyze_general_tool() -> str:
    """
    Сформировать общие выводы по датасету.

    Returns:
        Отформатированная строка с общими инсайтами
    """

    df = get_current_dataset()

    if df is None:
        raise ValueError("Датасет не загружен. Сначала загрузите данные через load_dataset().")

    date_col, sales_col, store_col = find_columns(df)

    product_col = customer_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['product', 'item', 'товар', 'sku']):
            product_col = col
        elif any(kw in col_lower for kw in ['customer', 'client', 'клиент']):
            customer_col = col

    result = sales_analysis.get_general_insights(
        df, sales_col, date_col, product_col, customer_col
    )

    # Нормализация возврата (dict → str)
    if isinstance(result, dict) and "answer" in result:
        return result["answer"]
    if isinstance(result, str):
        return result
    return str(result)