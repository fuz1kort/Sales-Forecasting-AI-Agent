"""
Типы данных для всего бэкенда.

Централизованный экспорт TypedDict и других типов.
Импортируйте отсюда, а не из вложенных модулей:
    from backend.schemas import ForecastResult
    вместо
    from backend.schemas.forecast_types import ForecastResult
"""

# === Forecast schemas ===
from .forecast_types import (
    ForecastPoint,
    ForecastMetrics,
    ForecastResult,
    ForecastSummary,
)

# === Экспорт для IDE и mypy ===
__all__ = [
    "ForecastPoint",
    "ForecastMetrics",
    "ForecastResult",
    "ForecastSummary",
]

# === Мета-информация (опционально) ===
# Помогает отладке и документации
__version__ = "1.0"
__description__ = "Типы данных для Sales Forecasting AI Agent"