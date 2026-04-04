"""Инструменты для анализа товарной матрицы и ассортимента."""
import logging
from typing import Optional

import pandas as pd
from smolagents import tool

from backend.agent.state import get_session_manager
from backend.utils import find_columns
from backend.agent.tools.data.load_tools import _detect_category_column, _get_dataset

logger = logging.getLogger(__name__)


@tool
def analyze_top_products(
        limit: int = 10,
        min_orders: int = 1,
        sort_by: str = "top",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        store_ids: Optional[str] = None,
        location_ids: Optional[str] = None,
        category_filter: Optional[str] = None,
        session_id: Optional[str] = None
) -> str:
    """
    Показать топ или худшие товары по объёму продаж.

    Вызывайте, когда пользователь спрашивает:
    - "какие товары продаются лучше всего?" (sort_by="top")
    - "покажи лидеров продаж" (sort_by="top")
    - "что пользуется спросом?" (sort_by="top")
    - "какие товары продаются хуже всего?" (sort_by="bottom")
    - "покажи аутсайдеров продаж" (sort_by="bottom")

    Args:
        limit: Сколько товаров показать (1..50, по умолчанию 10)
        min_orders: Минимальное количество заказов для включения в рейтинг
        sort_by: "top" для лучших, "bottom" для худших
        date_from: Начальная дата фильтрации (YYYY-MM-DD)
        date_to: Конечная дата фильтрации (YYYY-MM-DD)
        store_ids: ID магазинов через запятую (например "1,2,3")
        location_ids: Фильтр по стране/регионам (например "USA,UK")
        category_filter: Фильтр по категории товара
        session_id: ID сессии (опционально))

    Returns:
        Форматированная строка с отчётом в Markdown
    """
    # Конвертация параметров
    try:
        limit = int(limit) if limit is not None else 10
        min_orders = int(min_orders) if min_orders is not None else 1
    except (ValueError, TypeError):
        limit = 10
        min_orders = 1
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен. Сначала вызовите load_dataset."

    date_col, sales_col, store_col, product_col = find_columns(df)
    category_col = _detect_category_column(df)

    if not product_col or not sales_col:
        return "❌ Не найдены колонки товара или продаж."

    # Ищем колонку с названиями товаров
    product_name_col = None
    for col in df.columns:
        if any(w in col.lower() for w in ["name", "название", "description", "desc", "productname"]):
            product_name_col = col
            break

    # Создаём mapping ID -> название для отображения
    if product_name_col and product_name_col != product_col:
        name_mapping = df.drop_duplicates(product_col)[[product_col, product_name_col]].set_index(product_col)[product_name_col].to_dict()
    else:
        name_mapping = {}

    # Применяем фильтры
    filtered_df = df.copy()

    # Фильтр по датам
    if date_col and (date_from or date_to):
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
        if date_from:
            filtered_df = filtered_df[filtered_df[date_col] >= pd.to_datetime(date_from)]
        if date_to:
            filtered_df = filtered_df[filtered_df[date_col] <= pd.to_datetime(date_to)]

    # Фильтр по магазинам
    if store_ids and store_ids.lower() != "all" and store_col:
        store_list = [s.strip() for s in store_ids.split(',')]
        filtered_df = filtered_df[filtered_df[store_col].astype(str).isin(store_list)]

    # Фильтр по локации (страна/регион - дополнительный параметр)
    if location_ids and location_ids.lower() != "all" and store_col:
        location_list = [s.strip() for s in location_ids.split(',')]
        filtered_df = filtered_df[filtered_df[store_col].astype(str).isin(location_list)]

    # Фильтр по категории
    if category_filter and category_filter.lower() != "all" and category_col:
        filtered_df = filtered_df[filtered_df[category_col].str.contains(category_filter, case=False, na=False)]

    if filtered_df.empty:
        return "⚠️ Нет данных после применения фильтров"

    # Ограничиваем limit разумными значениями
    limit = max(1, min(50, limit))

    # Агрегация по товарам
    stats = (
        filtered_df.groupby(product_col)[sales_col]
        .agg(total="sum", orders="count", avg="mean")
        .round(2)
        .query(f"orders >= {min_orders}")
        .sort_values("total", ascending=(sort_by == "bottom"))
        .head(limit)
    )

    if stats.empty:
        return f"⚠️ Нет товаров с ≥{min_orders} заказами после фильтрации"

    # Формирование отчёта
    title = "ТОП" if sort_by == "top" else "ХУДШИЕ"
    lines = [f"📊 **{title}-{len(stats)} ТОВАРОВ ПО ПРОДАЖАМ**\n"]

    for rank, (name, row) in enumerate(stats.iterrows(), 1):
        display_name = name_mapping.get(name, str(name))
        # Выделяем товары с низкой выручкой
        if row['total'] < 1:
            display_name = f"**{display_name}**"
        lines.append(f"{rank}. **{display_name}**")
        lines.append(f"   • Выручка: ${row['total']:,.2f}")
        lines.append(f"   • Заказы: {int(row['orders'])}")
        lines.append(f"   • Средний чек: ${row['avg']:,.2f}")
        lines.append("")  # пустая строка между товарами

    # Итоговая аналитика
    total_rev = filtered_df[sales_col].sum()
    selected_rev = stats["total"].sum()
    share = (selected_rev / total_rev * 100) if total_rev > 0 else 0

    if sort_by == "top":
        lines.append(f"💡 **Вывод:** {title}-{len(stats)} товаров дают {share:.1f}% всей выручки")
        if len(stats) > 0:
            leader = stats.index[0]
            leader_name = name_mapping.get(leader, str(leader))
            lines.append(f"🏆 Лидер: **{leader_name}** (${stats.iloc[0]['total']:,.2f})")
    else:
        lines.append(f"💡 **Вывод:** {title}-{len(stats)} товаров дают {share:.1f}% всей выручки")
        if len(stats) > 0:
            worst = stats.index[0]
            worst_name = name_mapping.get(worst, str(worst))
            lines.append(f"📉 Аутсайдер: **{worst_name}** (${stats.iloc[0]['total']:,.2f})")

    return "\n".join(lines)


@tool
def analyze_product_categories(
        category_col: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        store_ids: Optional[str] = None,
        location_ids: Optional[str] = None,
        session_id: Optional[str] = None
) -> str:
    """
    Анализ продаж по категориям товаров.

    Вызывайте, когда в данных есть колонка категории
    и пользователь хочет увидеть распределение по группам.

    Args:
        category_col: Название колонки с категорией (авто-поиск, если None)
        date_from: Начальная дата фильтрации (YYYY-MM-DD), если None - без фильтра
        date_to: Конечная дата фильтрации (YYYY-MM-DD), если None - без фильтра
        store_ids: ID магазинов через запятую (например "1,2,3")
        location_ids: Фильтр по стране/регионам (например "USA,UK")
        session_id: ID сессии (опционально)

    Returns:
        Форматированная строка с отчётом
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, store_col, product_col = find_columns(df)

    # Авто-поиск колонки категории
    if not category_col:
        category_col = _detect_category_column(df)

    if not category_col or category_col not in df.columns:
        return f"❌ Колонка категории не найдена. Доступные: {list(df.columns)}"

    # Применяем фильтры
    filtered_df = df.copy()

    # Фильтр по датам
    if date_col and (date_from or date_to):
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
        if date_from:
            filtered_df = filtered_df[filtered_df[date_col] >= pd.to_datetime(date_from)]
        if date_to:
            filtered_df = filtered_df[filtered_df[date_col] <= pd.to_datetime(date_to)]

    # Фильтр по магазинам
    if store_ids and store_ids.lower() != "all" and store_col:
        store_list = [s.strip() for s in store_ids.split(',')]
        filtered_df = filtered_df[filtered_df[store_col].astype(str).isin(store_list)]

    # Фильтр по локации (страна/регион - дополнительный параметр)
    if location_ids and location_ids.lower() != "all" and store_col:
        location_list = [s.strip() for s in location_ids.split(',')]
        filtered_df = filtered_df[filtered_df[store_col].astype(str).isin(location_list)]

    if filtered_df.empty:
        return "⚠️ Нет данных после применения фильтров"

    # Агрегация по категориям
    by_category = (
        filtered_df.groupby(category_col)[sales_col]
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


@tool
def analyze_product_by_name(
        product_name: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        session_id: Optional[str] = None
) -> str:
    """
    Показать детальную информацию о продажах конкретного товара по названию.

    Вызывайте, когда пользователь спрашивает о конкретном товаре:
    - "расскажи про продажи PEN, 10 COLOURS"
    - "какие продажи у MAGIC SLATE?"
    - "сколько продано REGENCY CAKESTAND?"

    Args:
        product_name: Название товара (полное или начало названия)
        date_from: Начальная дата фильтрации (YYYY-MM-DD)
        date_to: Конечная дата фильтрации (YYYY-MM-DD)
        session_id: ID сессии (опционально)

    Returns:
        Форматированная строка с отчётом о товаре
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен. Сначала вызовите load_dataset."

    date_col, sales_col, store_col, product_col = find_columns(df)

    if not product_col or not sales_col:
        return "❌ Не найдены колонки товара или продаж."

    # Ищем кол онку с названиями товаров
    product_name_col = None
    for col in df.columns:
        if any(w in col.lower() for w in ["name", "название", "description", "desc", "productname"]):
            product_name_col = col
            break

    # Поиск товара по названию (с поддержкой подстрок)
    matching_products = []
    if product_name_col:
        matching_products = df[df[product_name_col].str.contains(product_name, case=False, na=False)][product_col].unique().tolist()
    
    if not matching_products:
        # Пробуем поиск по ID товара
        matching_products = df[df[product_col].astype(str).str.contains(product_name, case=False, na=False)][product_col].unique().tolist()

    if not matching_products:
        return f"❌ Товар '{product_name}' не найден в базе данных"

    # Если найдено несколько товаров, берём первый (или самый продаваемый)
    if len(matching_products) > 1:
        # Берём товар с максимальной выручкой
        product_id = df[df[product_col].isin(matching_products)].groupby(product_col)[sales_col].sum().idxmax()
    else:
        product_id = matching_products[0]

    # Получаем всю информацию о товаре
    product_data = df[df[product_col] == product_id].copy()

    # Фильтруем по датам если нужно
    if date_col and (date_from or date_to):
        product_data[date_col] = pd.to_datetime(product_data[date_col], errors='coerce')
        if date_from:
            product_data = product_data[product_data[date_col] >= pd.to_datetime(date_from)]
        if date_to:
            product_data = product_data[product_data[date_col] <= pd.to_datetime(date_to)]

    if product_data.empty:
        return f"⚠️ Нет данных о товаре '{product_name}' в указанном периоде"

    # Вычисляем статистику
    total_revenue = product_data[sales_col].sum()
    total_orders = len(product_data)
    avg_revenue = product_data[sales_col].mean()
    max_revenue = product_data[sales_col].max()
    min_revenue = product_data[sales_col].min()

    # Получаем название товара
    display_name = product_name
    if product_name_col:
        names = product_data[product_name_col].unique()
        if len(names) > 0:
            display_name = names[0]

    # Формируем отчёт
    lines = [f"📦 **АНАЛИЗ ТОВАРА: {display_name}**\n"]
    
    lines.append(f"💰 **Выручка**: ${total_revenue:,.2f}")
    lines.append(f"📊 **Среднее за период**: ${avg_revenue:,.2f}")
    lines.append(f"📈 **Максимум**: ${max_revenue:,.2f}")
    lines.append(f"📉 **Минимум**: ${min_revenue:,.2f}")
    lines.append(f"📦 **Количество записей**: {total_orders}")

    # Доля от общей выручки
    if date_col:
        period_total = df.copy()
        if date_from:
            period_total[date_col] = pd.to_datetime(period_total[date_col], errors='coerce')
            period_total = period_total[period_total[date_col] >= pd.to_datetime(date_from)]
        if date_to:
            period_total[date_col] = pd.to_datetime(period_total[date_col], errors='coerce')
            period_total = period_total[period_total[date_col] <= pd.to_datetime(date_to)]
        
        period_total_revenue = period_total[sales_col].sum()
        share = (total_revenue / period_total_revenue * 100) if period_total_revenue > 0 else 0
        lines.append(f"🎯 **Доля рынка**: {share:.2f}%")

    # Если найдено несколько товаров, указываем
    if len(matching_products) > 1:
        lines.append(f"\n⚠️ *Найдено {len(matching_products)} товаров по названию. Показана информация по товару с максимальной выручкой.*")

    return "\n".join(lines)