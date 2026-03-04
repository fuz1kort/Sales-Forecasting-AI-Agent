"""Инструменты для анализа временных трендов и сезонности."""
import logging
from typing import Optional, Literal

import pandas as pd
from smolagents import tool

from agent.state import get_session_manager
from utils import find_columns

logger = logging.getLogger(__name__)

PeriodType = Literal["daily", "weekly", "monthly", "quarterly"]


@tool
def analyze_trends(
        period: PeriodType = "monthly",
        session_id: Optional[str] = None
) -> str:
    """
    Показать динамику продаж во времени.

    Вызывайте, когда пользователь спрашивает:
    - "как менялись продажи?"
    - "покажи график продаж по месяцам"
    - "есть ли рост или падение?"

    Args:
        period: Период группировки: daily | weekly | monthly | quarterly
        session_id: ID сессии (опционально)

    Returns:
        Форматированная строка с отчётом о трендах
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, _ = find_columns(df)
    if not date_col or not sales_col:
        return "❌ Нет колонок даты или продаж."

    # Подготовка данных
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.sort_values(date_col)

    # Маппинг периодов для pandas Grouper
    freq_map = {
        "daily": "D",
        "weekly": "W-MON",
        "monthly": "ME",  # Month End (pandas >= 2.2)
        "quarterly": "QE"  # Quarter End
    }
    freq = freq_map.get(period, "ME")

    # Группировка
    grouped = df_copy.groupby(pd.Grouper(key=date_col, freq=freq))[sales_col].agg(["sum", "mean"]).round(2)

    if grouped.empty:
        return "⚠️ Недостаточно данных для анализа трендов"

    # Формирование отчёта
    lines = [f"📈 **ТРЕНДЫ ПРОДАЖ ({period.upper()})**\n"]

    # Показываем последние 12 периодов (или все, если меньше)
    recent = grouped.tail(12)

    for period_label, row in recent.iterrows():
        label = _format_period_label(period_label, period)
        # Визуальный бар
        bar_len = min(int(row["sum"] / grouped["sum"].max() * 25), 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        lines.append(f"• {label}: {bar} ${row['sum']:,.2f}")

    # Расчёт динамики
    if len(grouped) >= 2:
        prev_val = grouped["sum"].iloc[-2]
        curr_val = grouped["sum"].iloc[-1]

        if prev_val > 0:
            growth = ((curr_val - prev_val) / prev_val) * 100
            trend_emoji = "📈" if growth > 0 else "📉" if growth < 0 else "➡️"
            lines.append(f"\n{trend_emoji} **Динамика:** {growth:+.1f}% к предыдущему периоду")

            # Простая классификация тренда
            if abs(growth) < 5:
                lines.append("• Статус: 🟢 Стабильно")
            elif growth > 20:
                lines.append("• Статус: 🚀 Быстрый рост")
            elif growth > 5:
                lines.append("• Статус: 📈 Умеренный рост")
            elif growth < -20:
                lines.append("• Статус: 🔴 Резкое падение")
            else:
                lines.append("• Статус: 📉 Небольшое снижение")

    return "\n".join(lines)


@tool
def analyze_seasonality(
        session_id: Optional[str] = None
) -> str:
    """
    Анализ сезонности продаж по месяцам.

    Вызывайте, когда пользователь интересуется:
    - "в какие месяцы продажи выше?"
    - "есть ли сезонность?"
    - "когда ждать пик продаж?"

    Args:
        session_id: ID сессии (опционально)

    Returns:
        Форматированная строка с анализом сезонности
    """
    df = _get_dataset(session_id)
    if df is None:
        return "❌ Датасет не загружен."

    date_col, sales_col, _ = find_columns(df)
    if not date_col or not sales_col:
        return "❌ Нет колонок даты или продаж."

    # Подготовка
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy['month'] = df_copy[date_col].dt.month
    df_copy['month_name'] = df_copy[date_col].dt.month_name(locale='ru_RU')

    # Средние продажи по месяцам
    monthly = df_copy.groupby('month')[sales_col].agg(['mean', 'std', 'count']).round(2)

    if len(monthly) < 3:
        return "⚠️ Недостаточно данных для анализа сезонности (нужно ≥3 разных месяца)"

    # Названия месяцев на русском
    months_ru = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                 "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

    lines = ["🌟 **АНАЛИЗ СЕЗОННОСТИ**\n"]

    # Визуализация барами
    max_val = monthly['mean'].max()
    for month_num, row in monthly.iterrows():
        month_name = months_ru[month_num - 1] if 1 <= month_num <= 12 else f"М{month_num}"
        bar_len = int((row['mean'] / max_val) * 20) if max_val > 0 else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"• {month_name}: {bar} ${row['mean']:,.2f}")

    # Статистика
    peak_month = monthly['mean'].idxmax()
    low_month = monthly['mean'].idxmin()
    cv = monthly['mean'].std() / monthly['mean'].mean() * 100  # Коэффициент вариации

    peak_name = months_ru[peak_month - 1] if 1 <= peak_month <= 12 else f"М{peak_month}"
    low_name = months_ru[low_month - 1] if 1 <= low_month <= 12 else f"М{low_month}"

    lines.append(f"\n💡 **Выводы:**")
    lines.append(f"• Пик: {peak_name} (${monthly.loc[peak_month, 'mean']:,.2f})")
    lines.append(f"• Минимум: {low_name} (${monthly.loc[low_month, 'mean']:,.2f})")

    # Интерпретация сезонности
    if cv > 30:
        lines.append("• 🔴 Ярко выраженная сезонность (CV > 30%)")
        lines.append("  → Планируйте запасы и маркетинг под пики")
    elif cv > 15:
        lines.append("• 🟡 Умеренная сезонность (CV 15-30%)")
        lines.append("  → Учитывайте при прогнозировании")
    else:
        lines.append("• 🟢 Слабая сезонность (CV < 15%)")
        lines.append("  → Продажи стабильны в течение года")

    return "\n".join(lines)


# === Вспомогательные функции ===

def _get_dataset(session_id: Optional[str]) -> Optional[pd.DataFrame]:
    sid = session_id or "default"
    return get_session_manager().get_dataset(sid)


def _format_period_label(period_label, period_type: str) -> str:
    """Форматирует метку периода для отображения."""
    try:
        if period_type == "daily":
            return period_label.strftime("%d.%m.%Y")
        elif period_type == "weekly":
            return f"Неделя {period_label.isocalendar()[1]:02d}/{period_label.year}"
        elif period_type == "monthly":
            return period_label.strftime("%B %Y").lower()
        elif period_type == "quarterly":
            return f"{period_label.year} Q{period_label.quarter}"
        return str(period_label)
    except Exception:
        return str(period_label)