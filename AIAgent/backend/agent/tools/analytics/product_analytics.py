"""Инструменты для анализа товарной матрицы и ассортимента."""
import logging
from typing import Optional

import pandas as pd
from smolagents import tool

from agent.state import get_session_manager
from utils import find_columns

logger = logging.getLogger(__name__)


@tool
def analyze_top_products(
        limit: int = 10,
        min_orders: int = 1,
        session_id: Optional[str] = None
) -> str:
    """
    Показать топ товаров по объёму продаж.

    Вызывайте, когда пользователь спрашивает:
    - "какие товары продаются лучше всего?"
    - "покажи лидеров продаж"
    - "что пользуется спросом?"

    Args:
        limit: Сколько товаров показать (1..50, по умолчанию 10)
        min_orders: Минимальное количество заказов для включения в рейтинг
        session_id: ID сессии (опционально)

    Returns:
        Форматированная строка с отчётом в Markdown
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен. Сначала вызовите load_dataset."

    _, sales_col, _ = find_columns(df)
    product_col = _detect_product_column(df)

    if not product_col or not sales_col:
        return "❌ Не найдены колонки товара или продаж."

    # Ограничиваем limit разумными значениями
    limit = max(1, min(50, limit))

    # Агрегация по товарам
    stats = (
        df.groupby(product_col)[sales_col]
        .agg(total="sum", orders="count", avg="mean")
        .round(2)
        .query(f"orders >= {min_orders}")
        .sort_values("total", ascending=False)
        .head(limit)
    )

    if stats.empty:
        return f"⚠️ Нет товаров с ≥{min_orders} заказами"

    # Формирование отчёта
    lines = [f"📊 **ТОП-{len(stats)} ТОВАРОВ ПО ПРОДАЖАМ**\n"]

    for rank, (name, row) in enumerate(stats.iterrows(), 1):
        lines.append(f"{rank}. **{name}**")
        lines.append(f"   • Выручка: ${row['total']:,.2f}")
        lines.append(f"   • Заказы: {int(row['orders'])}")
        lines.append(f"   • Средний чек: ${row['avg']:,.2f}")
        lines.append("")  # пустая строка между товарами

    # Итоговая аналитика
    total_rev = df[sales_col].sum()
    top_rev = stats["total"].sum()
    share = (top_rev / total_rev * 100) if total_rev > 0 else 0

    lines.append(f"💡 **Вывод:** Топ-{len(stats)} товаров дают {share:.1f}% всей выручки")

    if len(stats) > 0:
        leader = stats.index[0]
        lines.append(f"🏆 Лидер: **{leader}** (${stats.iloc[0]['total']:,.2f})")

    return "\n".join(lines)


@tool
def analyze_product_categories(
        category_col: Optional[str] = None,
        session_id: Optional[str] = None
) -> str:
    """
    Анализ продаж по категориям товаров.

    Вызывайте, когда в данных есть колонка категории
    и пользователь хочет увидеть распределение по группам.

    Args:
        category_col: Название колонки с категорией (авто-поиск, если None)
        session_id: ID сессии (опционально)

    Returns:
        Форматированная строка с отчётом
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    _, sales_col, _ = find_columns(df)

    # Авто-поиск колонки категории
    if not category_col:
        category_col = _detect_category_column(df)

    if not category_col or category_col not in df.columns:
        return f"❌ Колонка категории не найдена. Доступные: {list(df.columns)}"

    # Агрегация по категориям
    by_category = (
        df.groupby(category_col)[sales_col]
        .agg(total="sum", items="count", avg="mean")
        .round(2)
        .sort_values("total", ascending=False)
    )

    lines = [f"📦 **ПРОДАЖИ ПО КАТЕГОРИЯМ**\n"]

    for cat, row in by_category.iterrows():
        bar = "█" * min(int(row["total"] / by_category["total"].max() * 30), 30)
        lines.append(f"• **{cat}**: {bar} ${row['total']:,.2f} ({row['items']} шт.)")

    # Топ-3 категории
    top3 = by_category.head(3)
    top3_share = (top3["total"].sum() / by_category["total"].sum() * 100)

    lines.append(f"\n💡 Топ-3 категории дают {top3_share:.1f}% выручки")

    return "\n".join(lines)


# === Вспомогательные функции (не @tool) ===

def _get_dataset(session_id: Optional[str]) -> Optional[pd.DataFrame]:
    """Безопасное получение датасета из сессии."""
    sid = session_id or "default"
    return get_session_manager().get_dataset(sid)


def _detect_product_column(df: pd.DataFrame) -> Optional[str]:
    """Эвристика для поиска колонки с названием товара."""
    keywords = ["product", "item", "товар", "продукт", "sku", "name", "название", "articul"]
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            # Исключаем колонки, которые явно не товары
            if any(ex in col_lower for ex in ["category", "категория", "type", "тип"]):
                continue
            return col
    return None


def _detect_category_column(df: pd.DataFrame) -> Optional[str]:
    """Эвристика для поиска колонки с категорией товара."""
    keywords = ["category", "категория", "type", "тип", "group", "группа", "class"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return None