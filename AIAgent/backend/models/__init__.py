"""
Модели прогнозирования продаж.

Этот пакет содержит реализации различных моделей машинного обучения
для прогнозирования временных рядов продаж.
"""

from backend.models.sarima_model import sarima_forecast
from backend.models.catboost_model import catboost_forecast

__all__ = [
    "sarima_forecast",
    "catboost_forecast",
]

