"""
Стекинг и оптимизированный ансамбль моделей.

На основе лучшей практики из Colab:
- Метод 1: Линейная регрессия со строго положительными весами
- Метод 2: Оптимизация весов через scipy.optimize.minimize с WMAPE
- Оба метода гарантируют, что веса неотрицательные и суммируются в 1
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Dict, List, Tuple

from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Weighted Mean Absolute Percentage Error (WMAPE).

    WMAPE = sum(|actual - pred|) / sum(|actual|)

    Better than simple MAPE because it weights errors by actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        WMAPE as a decimal (0-1). Lower is better.
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


class SimpleStackingEnsemble:
    """
    Фасад для стекинга 6 моделей через Linear Regression.
    
    Обучает линейную регрессию с ограничением: веса >= 0 (positive=True).
    Веса автоматически нормализуются через softmax при предсказании.
    """

    def __init__(self, model_names: List[str] = None):
        """
        Args:
            model_names: Названия моделей для логирования
        """
        self.model_names = model_names or [f"Model_{i}" for i in range(6)]
        self.lr_model = None
        self.weights = None

    def fit(self, predictions: np.ndarray, y_true: np.ndarray):
        """
        Обучает мета-модель (Linear Regression) на предсказаниях базовых моделей.

        Args:
            predictions: Shape (n_models, n_samples) - предсказания каждой модели
            y_true: Shape (n_samples,) - истинные значения
        """
        # Transpose: (n_samples, n_models) для LinearRegression
        X = predictions.T

        self.lr_model = LinearRegression(positive=True)  # Веса >= 0
        self.lr_model.fit(X, y_true)

        # Нормализуем веса так чтобы суммировались в 1
        raw_weights = self.lr_model.coef_
        self.weights = raw_weights / np.sum(raw_weights)

        # Логируем веса
        logger.info("📊 Веса Stacking Ensemble (Linear Regression):")
        for name, weight in zip(self.model_names, self.weights):
            logger.info(f"  {name}: {weight:.4f}")

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Вычисляет взвешенное среднее предсказаний базовых моделей.

        Args:
            predictions: Shape (n_models, n_samples)

        Returns:
            Shape (n_samples,) - итоговые предсказания ансамбля
        """
        if self.weights is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        return np.dot(self.weights, predictions)


class OptimizedWeightEnsemble:
    """
    Оптимизирует веса моделей минимизацией WMAPE на валидационном наборе.
    
    Использует scipy.optimize.minimize с SLSQP методом.
    Гарантирует: веса >= 0, веса суммируются в 1, WMAPE минимальна.
    """

    def __init__(self, model_names: List[str] = None):
        """
        Args:
            model_names: Названия моделей для логирования
        """
        self.model_names = model_names or [f"Model_{i}" for i in range(6)]
        self.weights = None
        self.best_wmape = None

    def fit(self, predictions: np.ndarray, y_true: np.ndarray):
        """
        Оптимизирует веса через scipy.optimize.minimize.

        Args:
            predictions: Shape (n_models, n_samples) - предсказания каждой модели
            y_true: Shape (n_samples,) - истинные значения
        """

        def objective(weights):
            """Функция потерь: WMAPE."""
            ensemble_pred = np.dot(weights, predictions)
            return calculate_wmape(y_true, ensemble_pred)

        n_models = predictions.shape[0]

        # Начальные веса: явно равные
        initial_weights = np.ones(n_models) / n_models

        # Ограничения
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Сумма = 1
        bounds = [(0, 1)] * n_models  # 0 <= weight <= 1

        # Оптимизация
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        self.weights = result.x
        self.best_wmape = result.fun

        # Логируем результаты
        logger.info("📊 Оптимальные веса Ensemble (scipy.optimize):")
        for name, weight in zip(self.model_names, self.weights):
            logger.info(f"  {name}: {weight:.4f}")
        logger.info(f"  WMAPE на валидации: {self.best_wmape * 100:.2f}%")

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Вычисляет взвешенное среднее предсказаний базовых моделей.

        Args:
            predictions: Shape (n_models, n_samples)

        Returns:
            Shape (n_samples,) - итоговые предсказания ансамбля
        """
        if self.weights is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        return np.dot(self.weights, predictions)


class HybridEnsemble:
    """
    Гибридный ансамбль: комбинирует несколько методов взвешивания.
    
    Возвращает предсказания от обоих методов для сравнения.
    """

    def __init__(self, model_names: List[str] = None):
        self.model_names = model_names or [f"Model_{i}" for i in range(6)]
        self.stacking = SimpleStackingEnsemble(model_names)
        self.optimized = OptimizedWeightEnsemble(model_names)

    def fit(self, predictions: np.ndarray, y_true: np.ndarray):
        """Обучает обе мета-модели."""
        logger.info("\n🔧 Обучение Stacking Ensemble (Linear Regression)...")
        self.stacking.fit(predictions, y_true)

        logger.info("\n⚙️ Обучение Optimized Ensemble (scipy.optimize)...")
        self.optimized.fit(predictions, y_true)

    def predict(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Возвращает предсказания обоих методов.

        Returns:
            {
                "stacking": np.ndarray,
                "optimized": np.ndarray,
                "average": np.ndarray  # Среднее между двумя методами
            }
        """
        stacking_pred = self.stacking.predict(predictions)
        optimized_pred = self.optimized.predict(predictions)
        average_pred = (stacking_pred + optimized_pred) / 2

        return {
            "stacking": stacking_pred,
            "optimized": optimized_pred,
            "average": average_pred,
        }


def blend_models(
    predictions_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    method: str = "optimized",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Простая функция для быстрого бленда моделей.

    Args:
        predictions_dict: {model_name: predictions_array}
        y_true: Истинные значения
        method: "optimized", "stacking", или "equal_weight"

    Returns:
        (ensemble_predictions, weights)
    """

    model_names = list(predictions_dict.keys())
    predictions = np.array([predictions_dict[name] for name in model_names])

    if method == "equal_weight":
        weights = np.ones(len(model_names)) / len(model_names)
        ensemble = np.mean(predictions, axis=0)

    elif method == "stacking":
        ensemble_obj = SimpleStackingEnsemble(model_names)
        ensemble_obj.fit(predictions, y_true)
        weights = dict(zip(model_names, ensemble_obj.weights))
        ensemble = ensemble_obj.predict(predictions)

    elif method == "optimized":
        ensemble_obj = OptimizedWeightEnsemble(model_names)
        ensemble_obj.fit(predictions, y_true)
        weights = dict(zip(model_names, ensemble_obj.weights))
        ensemble = ensemble_obj.predict(predictions)

    else:
        raise ValueError(f"Unknown method: {method}")

    return ensemble, weights
