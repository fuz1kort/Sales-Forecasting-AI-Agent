"""
Модели прогнозирования продаж.

Этот пакет содержит реализации различных моделей машинного обучения
для прогнозирования временных рядов продаж.

Доступные модели:
- sarima_forecast: SARIMA с поддержкой Auto-ARIMA (автоматический подбор параметров)
- prophet_forecast: Prophet с праздниками и регрессорами
- catboost_forecast: ML-модель CatBoost (быстрая, хороша для нелинейности)
- ensemble_forecast_optimized: Оптимизированный ансамбль из нескольких моделей
"""

from backend.models.sarima_model import sarima_forecast
from backend.models.catboost_model import catboost_forecast, ensemble_forecast_optimized
from backend.models.prophet_model import prophet_forecast

from backend.models.ensemble import (
    SimpleStackingEnsemble,
    OptimizedWeightEnsemble,
    HybridEnsemble,
    calculate_wmape,
    blend_models,
)

__all__ = [
    "sarima_forecast",
    "catboost_forecast",
    "prophet_forecast",
    "ensemble_forecast_optimized",
    "SimpleStackingEnsemble",
    "OptimizedWeightEnsemble",
    "HybridEnsemble",
    "calculate_wmape",
    "blend_models",
]

