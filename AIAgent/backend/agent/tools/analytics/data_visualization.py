"""Инструменты для визуализации данных и аналитики."""
import logging
from typing import Optional
import json

import pandas as pd
import numpy as np
from smolagents import tool

from backend.agent.state import get_session_manager
from backend.utils import find_columns
from backend.agent.tools.data.load_tools import _get_dataset

logger = logging.getLogger(__name__)


@tool
def visualize_correlations(session_id: Optional[str] = None) -> str:
    """
    Показать матрицу корреляций числовых переменных.

    Полезно для понимания связей между Quantity, Price, Revenue.

    Args:
        session_id: ID сессии

    Returns:
        JSON с данными для heatmap
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    numeric_columns = ['Quantity', 'Price', 'Revenue']
    available_numeric = [col for col in numeric_columns if col in df.columns]

    if len(available_numeric) < 2:
        return "❌ Недостаточно числовых колонок для корреляций."

    corr_matrix = df[available_numeric].corr()

    # Преобразуем в формат для графика
    chart_data = {
        "type": "heatmap",
        "title": "Матрица корреляций числовых переменных",
        "x": available_numeric,
        "y": available_numeric,
        "z": corr_matrix.values.tolist(),
        "text": [[f"{val:.2f}" for val in row] for row in corr_matrix.values]
    }

    response = "📊 **Матрица корреляций**\n\n"
    response += "Показывает связи между переменными (от -1 до 1).\n\n"
    response += f"```json\n{json.dumps(chart_data, indent=2)}\n```"

    return response


@tool
def visualize_distributions(session_id: Optional[str] = None) -> str:
    """
    Показать распределения Revenue (гистограмма и boxplot).

    Args:
        session_id: ID сессии

    Returns:
        JSON с данными для графиков
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    if 'Revenue' not in df.columns:
        return "❌ Колонка Revenue не найдена."

    revenue = df['Revenue'].dropna()
    revenue_log = np.log1p(revenue)

    # Гистограмма
    hist_data = {
        "type": "histogram",
        "title": "Распределение Revenue (логарифм)",
        "x": revenue_log.tolist(),
        "nbinsx": 50
    }

    # Boxplot
    box_data = {
        "type": "box",
        "title": "Boxplot выручки",
        "y": revenue.tolist()
    }

    response = "📊 **Распределения Revenue**\n\n"
    response += "Гистограмма (логарифм) и boxplot для анализа выбросов.\n\n"
    response += f"**Гистограмма:** ```json\n{json.dumps(hist_data, indent=2)}\n```\n\n"
    response += f"**Boxplot:** ```json\n{json.dumps(box_data, indent=2)}\n```"

    return response


@tool
def visualize_time_series(session_id: Optional[str] = None) -> str:
    """
    Показать временной ряд продаж (еженедельно, с заполнением пропусков).

    Args:
        session_id: ID сессии

    Returns:
        JSON с данными для графика
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, store_col, product_col = find_columns(df)
    if not date_col or not sales_col:
        return "❌ Нет колонок даты или продаж."

    # Агрегация по неделям
    df_weekly = df.copy()
    df_weekly[date_col] = pd.to_datetime(df_weekly[date_col])
    df_weekly = df_weekly.set_index(date_col).resample('W')[sales_col].sum().reset_index()
    df_weekly.columns = ['Date', 'Revenue']

    # Заполнение пропусков
    df_weekly['Revenue'] = df_weekly['Revenue'].interpolate(method='linear')

    chart_data = {
        "type": "line",
        "title": "Временной ряд продаж (еженедельно)",
        "x": df_weekly['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "y": df_weekly['Revenue'].tolist()
    }

    response = "📈 **Временной ряд продаж**\n\n"
    response += "Еженедельная агрегация с заполнением пропусков.\n\n"
    response += f"```json\n{json.dumps(chart_data, indent=2)}\n```"

    return response


@tool
def visualize_top_products(limit: int = 10, session_id: Optional[str] = None) -> str:
    """
    Показать топ-продуктов по выручке.

    Args:
        limit: Количество топ-продуктов
        session_id: ID сессии

    Returns:
        JSON с данными для bar chart
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    if 'ProductName' not in df.columns or 'Revenue' not in df.columns:
        return "❌ Колонки ProductName или Revenue не найдены."

    # Группировка по продуктам
    product_analysis = df.groupby('ProductName').agg({
        'Revenue': 'sum',
        'Quantity': 'sum'
    }).reset_index()

    top_products = product_analysis.nlargest(limit, 'Revenue')

    chart_data = {
        "type": "bar",
        "title": f"Топ-{limit} продуктов по выручке",
        "x": top_products['ProductName'].tolist(),
        "y": top_products['Revenue'].tolist()
    }

    response = f"📦 **Топ-{limit} продуктов по выручке**\n\n"
    response += f"```json\n{json.dumps(chart_data, indent=2)}\n```"

    return response


@tool
def visualize_abc_analysis(session_id: Optional[str] = None) -> str:
    """
    ABC-анализ продуктов (A: 80% дохода, B: 15%, C: 5%).

    Args:
        session_id: ID сессии

    Returns:
        Текст с анализом и данными для pie chart
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    if 'ProductName' not in df.columns or 'Revenue' not in df.columns:
        return "❌ Колонки ProductName или Revenue не найдены."

    product_analysis = df.groupby('ProductName')['Revenue'].sum().reset_index()
    total_revenue = product_analysis['Revenue'].sum()
    product_analysis['Revenue_share'] = product_analysis['Revenue'] / total_revenue * 100
    product_analysis = product_analysis.sort_values('Revenue', ascending=False)
    product_analysis['Cumulative_share'] = product_analysis['Revenue_share'].cumsum()

    def assign_abc(cum_share):
        if cum_share <= 80:
            return 'A'
        elif cum_share <= 95:
            return 'B'
        else:
            return 'C'

    product_analysis['ABC_class'] = product_analysis['Cumulative_share'].apply(assign_abc)

    abc_counts = product_analysis['ABC_class'].value_counts()

    chart_data = {
        "type": "pie",
        "title": "ABC-анализ продуктов",
        "labels": abc_counts.index.tolist(),
        "values": abc_counts.values.tolist()
    }

    response = "🔤 **ABC-анализ продуктов**\n\n"
    response += "- **A-класс**: 20% продуктов, 80% дохода\n"
    response += "- **B-класс**: 30% продуктов, 15% дохода\n"
    response += "- **C-класс**: 50% продуктов, 5% дохода\n\n"
    response += f"Распределение: {dict(abc_counts)}\n\n"
    response += f"```json\n{json.dumps(chart_data, indent=2)}\n```"

    return response