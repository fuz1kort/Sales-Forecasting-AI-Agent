"""Инструменты для дашборда KPI и общих метрик."""
import logging
from typing import Optional

import pandas as pd
from smolagents import tool

from agent.state import get_session_manager
from utils import find_columns

logger = logging.getLogger(__name__)


@tool
def analyze_kpi(
        session_id: Optional[str] = None
) -> str:
    """
    Показать дашборд ключевых метрик (KPI).

    Вызывайте, когда пользователь спрашивает:
    - "покажи основные показатели"
    - "какие у нас KPI?"
    - "давай сводку по продажам"

    Args:
        session_id: ID сессии (опционально)

    Returns:
        Форматированная строка с дашбордом KPI
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, store_col = find_columns(df)
    if not sales_col:
        return "❌ Не найдена колонка продаж."

    # === Основные метрики ===
    total_sales = df[sales_col].sum()
    avg_sales = df[sales_col].mean()
    median_sales = df[sales_col].median()
    transactions = len(df)

    # === Дополнительные метрики (если есть данные) ===
    metrics = [
        f"💰 **Продажи:**",
        f"  • Общая сумма: ${total_sales:,.2f}",
        f"  • Средний чек: ${avg_sales:,.2f}",
        f"  • Медианный чек: ${median_sales:,.2f}",
        f"  • Транзакций: {transactions:,}",
    ]

    # Если есть колонка магазина
    if store_col and store_col in df.columns:
        stores = df[store_col].nunique()
        top_store = df.groupby(store_col)[sales_col].sum().idxmax()
        top_store_sales = df.groupby(store_col)[sales_col].sum().max()
        metrics.extend([
            f"\n🏪 **Магазины:**",
            f"  • Всего: {stores}",
            f"  • Лидер: {top_store} (${top_store_sales:,.2f})",
        ])

    # Если есть дата — добавляем период
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            date_range = f"{df[date_col].min().date()} — {df[date_col].max().date()}"
            metrics.append(f"\n📅 **Период:** {date_range}")

            # Продажи в день (если данных достаточно)
            days = (df[date_col].max() - df[date_col].min()).days + 1
            if days > 0:
                daily_avg = total_sales / days
                metrics.append(f"  • В среднем в день: ${daily_avg:,.2f}")
        except Exception:
            pass

    # === Статус-индикаторы ===
    status_lines = ["\n🎯 **Статус:**"]

    if total_sales > 0:
        if avg_sales > total_sales * 0.01:  # Средний чек > 1% от общего
            status_lines.append("  • 🟢 Здоровые показатели")
        else:
            status_lines.append("  • 🟡 Низкий средний чек — проверьте данные")
    else:
        status_lines.append("  • 🔴 Нет продаж в данных")

    return "\n".join(metrics + status_lines)


@tool
def analyze_general(
        session_id: Optional[str] = None
) -> str:
    """
    Общие выводы и мета-информация о датасете.

    Вызывайте для первичного ознакомления с данными
    или когда пользователь спрашивает "что в этих данных?".

    Args:
        session_id: ID сессии (опционально)

    Returns:
        Форматированная строка с обзором данных
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, store_col = find_columns(df)

    lines = ["📋 **ОБЗОР ДАННЫХ**\n"]

    # Размер данных
    lines.append(f"📊 **Размер:**")
    lines.append(f"  • Строк: {len(df):,}")
    lines.append(f"  • Колонок: {len(df.columns)}")
    lines.append(f"  • Размер в памяти: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Найденные ключевые колонки
    lines.append(f"\n🔍 **Авто-определённые колонки:**")
    lines.append(f"  • Дата: `{date_col or 'не найдена'}`")
    lines.append(f"  • Продажи: `{sales_col or 'не найдена'}`")
    lines.append(f"  • Магазин: `{store_col or 'не найдена'}`")

    # Статистика по продажам (если есть)
    if sales_col and sales_col in df.columns:
        sales = df[sales_col].dropna()
        if len(sales) > 0:
            lines.append(f"\n💰 **Статистика продаж:**")
            lines.append(f"  • Мин: ${sales.min():,.2f}")
            lines.append(f"  • Макс: ${sales.max():,.2f}")
            lines.append(f"  • Среднее: ${sales.mean():,.2f}")
            lines.append(f"  • Пропущено: {df[sales_col].isna().sum()}")

    # Качество данных
    missing = df.isna().sum().sum()
    total_cells = df.size
    completeness = (1 - missing / total_cells) * 100 if total_cells > 0 else 0

    lines.append(f"\n✨ **Качество данных:**")
    if completeness >= 95:
        lines.append(f"  • 🟢 Полнота: {completeness:.1f}%")
    elif completeness >= 80:
        lines.append(f"  • 🟡 Полнота: {completeness:.1f}% (есть пропуски)")
    else:
        lines.append(f"  • 🔴 Полнота: {completeness:.1f}% (много пропусков)")

    return "\n".join(lines)


# === Вспомогательные функции ===

def _get_dataset(session_id: Optional[str]) -> Optional[pd.DataFrame]:
    sid = session_id or "default"
    return get_session_manager().get_dataset(sid)