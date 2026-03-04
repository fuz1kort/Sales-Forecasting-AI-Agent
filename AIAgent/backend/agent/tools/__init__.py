"""
Реестр всех инструментов для AI-агента.

Импортируйте отсюда для регистрации в агенте:
    from backend.agent.tools import register_all_tools
    register_all_tools(agent)
"""
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from smolagents import MultiStepAgent

logger = logging.getLogger(__name__)

# === Data tools ===
from .data.load_tools import load_dataset, get_dataset_info

# === Analytics tools ===
from .analytics.product_analytics import analyze_top_products, analyze_product_categories
from .analytics.trend_analytics import analyze_trends, analyze_seasonality
from .analytics.kpi_analytics import analyze_kpi, analyze_general
from .analytics.store_analytics import analyze_store_profitability, compare_stores

# === Forecast tools ===
from .forecast.forecast_tools import build_forecast, get_forecast_summary
from .forecast.backtest_tools import run_backtest

# === Экспорт для удобного импорта ===
__all__ = [
    # Data
    "load_dataset",
    "get_dataset_info",

    # Product Analytics
    "analyze_top_products",
    "analyze_product_categories",

    # Trend Analytics
    "analyze_trends",
    "analyze_seasonality",

    # KPI Analytics
    "analyze_kpi",
    "analyze_general",

    # Store Analytics
    "analyze_store_profitability",
    "compare_stores",

    # Forecast
    "build_forecast",
    "get_forecast_summary",
    "run_backtest",

]

# === Реестр для массовой регистрации ===
TOOLS_REGISTRY = {name: globals()[name] for name in __all__}


def register_all_tools(agent: "MultiStepAgent") -> None:
    """
    Зарегистрировать все инструменты в агенте.

    Usage:
        from backend.agent.tools import register_all_tools
        register_all_tools(my_agent)
    """
    for tool_name, tool_func in TOOLS_REGISTRY.items():
        agent.register_tool(tool_func)
        logger.debug(f"✓ Registered tool: {tool_name}")

    logger.info(f"✅ Registered {len(TOOLS_REGISTRY)} tools")