"""Инструменты для анализа по магазинам/локациям."""
import logging
from typing import Optional, Literal

import pandas as pd
from smolagents import tool

from agent.state import get_session_manager
from utils import find_columns

logger = logging.getLogger(__name__)


@tool
def analyze_store_profitability(
        session_id: Optional[str] = None,
        metric: Literal["revenue", "orders", "avg_check"] = "revenue"
) -> str:
    """
    Показать рейтинг магазинов по прибыльности.

    Вызывайте, когда пользователь спрашивает:
    - "какой магазин самый прибыльный?"
    - "покажи топ магазинов по выручке"
    - "сравни эффективность точек"
    - "где лучшие продажи?"

    Args:
        session_id: ID сессии (опционально, по умолчанию "default")
        metric: Метрика для ранжирования:
            • "revenue" — общая выручка (по умолчанию)
            • "orders" — количество заказов
            • "avg_check" — средний чек

    Returns:
        Форматированная строка с рейтингом магазинов в Markdown

    Example:
        analyze_store_profitability(session_id="user_123", metric="revenue")
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен. Сначала вызовите load_dataset."

    # Авто-поиск колонок
    date_col, sales_col, store_col = find_columns(df)

    if not store_col:
        return "❌ В данных нет колонки магазина/локации. Невозможно выполнить анализ."

    if not sales_col:
        return "❌ Не найдена колонка продаж."

    # Группировка по магазину
    stats = df.groupby(store_col)[sales_col].agg(
        total_revenue="sum",
        orders="count",
        avg_check="mean"
    ).round(2).sort_values("total_revenue", ascending=False)

    if stats.empty:
        return "⚠️ Нет данных для анализа магазинов"

    # Выбор метрики для сортировки и отображения
    sort_by = {
        "revenue": "total_revenue",
        "orders": "orders",
        "avg_check": "avg_check"
    }.get(metric, "total_revenue")

    stats = stats.sort_values(sort_by, ascending=False)
    metric_labels = {
        "revenue": "Выручка",
        "orders": "Заказы",
        "avg_check": "Средний чек"
    }

    # Формирование отчёта
    lines = [f"🏆 **РЕЙТИНГ МАГАЗИНОВ ПО {metric_labels[metric].upper()}**\n"]

    max_val = stats[sort_by].max()

    for rank, (store, row) in enumerate(stats.iterrows(), 1):
        # Визуальный бар (пропорционально значению)
        if max_val > 0:
            bar_len = min(int(row[sort_by] / max_val * 25), 25)
        else:
            bar_len = 0
        bar = "█" * bar_len + "░" * (25 - bar_len)

        lines.append(f"{rank}. **{store}**")
        lines.append(f"   {bar} ${row['total_revenue']:,.2f}")
        lines.append(f"   • Заказов: {int(row['orders'])} | Средний чек: ${row['avg_check']:,.2f}")
        lines.append("")

    # Итоговая аналитика
    total_rev = stats["total_revenue"].sum()
    top3 = stats.head(3)
    top3_share = (top3["total_revenue"].sum() / total_rev * 100) if total_rev > 0 else 0

    lines.append(f"💡 **Вывод:** Топ-3 магазина дают {top3_share:.1f}% всей выручки")

    if len(stats) > 0:
        leader = stats.index[0]
        lines.append(f"🥇 Лидер: **{leader}** (${stats.iloc[0]['total_revenue']:,.2f})")

    return "\n".join(lines)


@tool
def compare_stores(
        store_ids: str,
        session_id: Optional[str] = None
) -> str:
    """
    Сравнить выбранные магазины по ключевым метрикам.

    Вызывайте для детального сравнения 2-5 конкретных магазинов,
    когда пользователь спрашивает:
    - "сравни магазин А и магазин Б"
    - "какой из этих магазинов лучше?"
    - "покажи разницу между точками"

    Args:
        store_ids: Список ID магазинов через запятую: "store_1, store_2"
        session_id: ID сессии (опционально)

    Returns:
        Сравнительная таблица в текстовом формате

    Example:
        compare_stores("Москва_Центр, Санкт-Петербург_Невский", session_id="user_123")
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    _, sales_col, store_col = find_columns(df)
    if not store_col or not sales_col:
        return "❌ Нет колонок магазина или продаж."

    # Парсим список магазинов
    stores = [s.strip() for s in store_ids.split(",") if s.strip()]
    if not stores:
        return "❌ Не указаны магазины для сравнения."

    # Фильтруем данные
    filtered = df[df[store_col].isin(stores)]
    if filtered.empty:
        return f"❌ Ни один из магазинов {stores} не найден в данных."

    # Агрегация
    comparison = filtered.groupby(store_col)[sales_col].agg(
        revenue="sum",
        orders="count",
        avg="mean",
        median="median"
    ).round(2)

    # Формирование таблицы
    lines = [f"📊 **СРАВНЕНИЕ МАГАЗИНОВ**\n"]
    lines.append(f"{'Магазин':<25} {'Выручка':>14} {'Заказы':>10} {'Ср. чек':>12}")
    lines.append("—" * 64)

    for store, row in comparison.iterrows():
        lines.append(f"{store:<25} ${row['revenue']:>12,.2f} {int(row['orders']):>10} ${row['avg']:>11,.2f}")

    # Подсветка лидеров
    best_revenue = comparison["revenue"].idxmax()
    best_avg = comparison["avg"].idxmax()

    lines.append(f"\n🏆 Лучший по выручке: **{best_revenue}**")
    if best_avg != best_revenue:
        lines.append(f"💎 Лучший средний чек: **{best_avg}** (${comparison.loc[best_avg, 'avg']:,.2f})")

    return "\n".join(lines)

def _get_dataset(session_id: Optional[str]) -> Optional[pd.DataFrame]:
    """Безопасное получение датасета из сессии."""
    sid = session_id or "default"
    return get_session_manager().get_dataset(sid)