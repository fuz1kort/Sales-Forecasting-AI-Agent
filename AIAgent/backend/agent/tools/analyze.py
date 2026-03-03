"""
Инструменты аналитики продаж.

Переэкспортирует функции из специализированных модулей для использования агентом.
"""

from agent.tools.analysis_main import (
    analyze_top_products_tool,
    analyze_trends_tool,
    analyze_general_tool,
)
from agent.tools.analysis_kpi import (
    analyze_kpi_tool,
    analyze_seasonality_tool,
)

__all__ = [
    "analyze_top_products_tool",
    "analyze_trends_tool",
    "analyze_kpi_tool",
    "analyze_seasonality_tool",
    "analyze_general_tool",
]
