"""Инструменты аналитики продаж. Все возвращают ТОЛЬКО строку."""
import logging
from typing import Optional

import pandas as pd
from smolagents import tool
from agent.state import get_session_manager
from utils import find_columns

logger = logging.getLogger(__name__)

@tool
def analyze_top_products_tool(limit: int = 10, session_id: str = None) -> str:
    """Анализ топ товаров по объёму продаж.

    Args:
        limit: Максимальное количество товаров для отображения (по умолчанию 10).
        session_id: Идентификатор сессии.
    """
    df = get_current_dataset_safe(session_id)
    if df is None:
        return "❌ Датасет не загружен. Сначала загрузите данные."

    date_col, sales_col, _ = find_columns(df)

    product_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['product', 'item', 'товар', 'продукт', 'sku', 'name']):
            product_col = col
            break

    if not product_col or not sales_col:
        return "❌ Не найдены колонки товара или продаж."

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
        analysis += f"   • Средний чек: ${data['mean']:,.2f}\n"

    total_revenue = df[sales_col].sum()
    top_revenue = top_products["sum"].sum()
    percentage = (top_revenue / total_revenue * 100) if total_revenue > 0 else 0

    analysis += "\n💡 **Выводы:**\n"
    analysis += f"• Топ-{limit} товаров дают ${top_revenue:,.2f} ({percentage:.1f}% выручки)\n"
    if len(top_products) > 0:
        analysis += f"• Лидер: {top_products.index[0]} (${top_products.iloc[0]['sum']:,.2f})"

    return analysis

@tool
def analyze_trends_tool(period: str = "monthly", session_id: str = None) -> str:
    """Анализ трендов продаж во времени.

    Args:
        period: Период группировки: daily, weekly, monthly, quarterly (по умолчанию monthly).
        session_id: Идентификатор сессии.
    """
    df = get_current_dataset_safe(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, _ = find_columns(df)
    if not date_col or not sales_col:
        return "❌ Нет колонок даты или продаж."

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    period_map = {"daily": "D", "weekly": "W-MON", "monthly": "ME", "quarterly": "QE"}
    freq = period_map.get(period.lower(), "ME")

    grouped = df_copy.groupby(pd.Grouper(key=date_col, freq=freq))[sales_col].sum()

    analysis = f"📈 **ТРЕНДЫ ПРОДАЖ ({period.upper()})**\n\n"
    recent = grouped.tail(12)

    for period_label, sales in recent.items():
        label = format_period_label(period_label, period.lower())
        analysis += f"• {label}: ${sales:,.2f}\n"

    if len(grouped) >= 2:
        prev_val = grouped.iloc[-2]
        curr_val = grouped.iloc[-1]
        if prev_val > 0:
            growth = ((curr_val - prev_val) / prev_val) * 100
            analysis += f"\n📊 **Динамика:** {growth:+.1f}% к предыдущему периоду\n"

    return analysis

@tool
def analyze_kpi_tool(session_id: str = None) -> str:
    """Формируем дашборд KPI.

    Args:
        session_id: Идентификатор сессии.
    """

    df = get_current_dataset_safe(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    _, sales_col, store_col = find_columns(df)
    if not sales_col:
        return "❌ Не найдена колонка продаж."

    total_sales = df[sales_col].sum()
    avg_sales = df[sales_col].mean()
    transactions = len(df)

    kpi_text = "📊 **ДАШБОРД KPI**\n\n"
    kpi_text += f"💰 **Продажи:**\n"
    kpi_text += f"  • Общая сумма: ${total_sales:,.2f}\n"
    kpi_text += f"  • Средний чек: ${avg_sales:,.2f}\n"
    kpi_text += f"  • Транзакций: {transactions:,}\n"

    if store_col:
        stores = df[store_col].nunique()
        kpi_text += f"  • Магазинов: {stores}\n"

    return kpi_text

@tool
def analyze_seasonality_tool(session_id: str = None) -> str:
    """Анализ сезонности продаж.

    Args:
        session_id: Идентификатор сессии.
    """

    df = get_current_dataset_safe(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, _ = find_columns(df)
    if not date_col or not sales_col:
        return "❌ Нет дат или продаж."

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy['month'] = df_copy[date_col].dt.month

    monthly = df_copy.groupby('month')[sales_col].mean().round(2)

    months = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
              "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

    analysis = "🌟 **АНАЛИЗ СЕЗОННОСТИ**\n\n"

    for month_num, avg_sales in monthly.items():
        month_name = months[month_num - 1] if month_num <= 12 else f"Месяц {month_num}"
        bar_length = int((avg_sales / monthly.max()) * 20) if monthly.max() > 0 else 0
        bar = "█" * bar_length + "░" * (20 - bar_length)
        analysis += f"• {month_name}: {bar} ${avg_sales:,.2f}\n"

    peak_month = monthly.idxmax()
    low_month = monthly.idxmin()
    variance = monthly.std()
    peak_name = months[peak_month - 1] if peak_month <= 12 else f"Месяц {peak_month}"
    low_name = months[low_month - 1] if low_month <= 12 else f"Месяц {low_month}"

    analysis += f"\n💡 **Выводы:**\n"
    analysis += f"• Пик продаж: {peak_name} (${monthly.max():,.2f})\n"
    analysis += f"• Минимум: {low_name} (${monthly.min():,.2f})\n"

    if variance > monthly.mean() * 0.3:
        analysis += "• 🔴 Ярко выраженная сезонность (> 30%)\n"
    elif variance > monthly.mean() * 0.15:
        analysis += "• 🟡 Умеренная сезонность (15-30%)\n"
    else:
        analysis += "• 🟢 Слабая сезонность (< 15%)\n"

    return analysis

@tool
def analyze_general_tool(session_id: str = None) -> str:
    """Общие выводы по датасету.

    Args:
        session_id: Идентификатор сессии.
    """

    df = get_current_dataset_safe(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, _ = find_columns(df)

    analysis = "📋 **ОБЩИЕ ВЫВОДЫ ПО ДАННЫМ**\n\n"
    analysis += f"📊 **Размер датасета:**\n"
    analysis += f"  • Строк: {len(df):,}\n"
    analysis += f"  • Колонок: {len(df.columns)}\n"

    if sales_col:
        analysis += f"💰 **Продажи:**\n"
        analysis += f"  • Общая сумма: ${df[sales_col].sum():,.2f}\n"
        analysis += f"  • Средняя: ${df[sales_col].mean():,.2f}\n"

    return analysis

def get_current_dataset_safe(session_id: Optional[str] = None) -> pd.DataFrame:
    """Безопасное получение датасета."""
    if not session_id:
        logger.warning("⚠️ session_id не передан в get_current_dataset_safe")
        return None

    session_manager = get_session_manager()
    dataset = session_manager.get_dataset(session_id)
    return dataset

def format_period_label(period_label, period_type: str) -> str:
    """Форматирует метку периода для отображения."""
    from pandas import Timestamp, Period
    if isinstance(period_label, (Timestamp, Period)):
        if period_type == "daily":
            return period_label.strftime("%Y-%m-%d")
        elif period_type == "weekly":
            return f"W{period_label.isocalendar()[1]:02d}-{period_label.year}"
        elif period_type == "monthly":
            return period_label.strftime("%b %Y").lower()
        elif period_type == "quarterly":
            return f"Q{period_label.quarter} {period_label.year}"
        return period_label.strftime("%Y-%m")

    try:
        return period_label.strftime("%Y-%m")
    except Exception:
        return str(period_label)