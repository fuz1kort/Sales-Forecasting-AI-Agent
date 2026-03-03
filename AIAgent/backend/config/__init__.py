"""
Конфигурация приложения.
"""

from .settings import AppSettings
from .constants import (
    COLUMN_KEYWORDS,
    DEFAULT_FORECAST_PERIODS,
    MAX_BACKTEST_DAYS,
    ERROR_MESSAGES,
)

__all__ = [
    "AppSettings",
    "COLUMN_KEYWORDS",
    "DEFAULT_FORECAST_PERIODS",
    "MAX_BACKTEST_DAYS",
    "ERROR_MESSAGES",
]

