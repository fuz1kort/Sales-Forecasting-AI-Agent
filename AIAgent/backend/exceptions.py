"""
Кастомные исключения приложения.
"""


class SalesAgentException(Exception):
    """Базовое исключение приложения."""
    pass


class DatasetError(SalesAgentException):
    """Ошибка при работе с датасетом."""
    pass


class DatasetNotLoadedError(DatasetError):
    """Датасет не загружен."""
    pass


class InvalidDatasetError(DatasetError):
    """Датасет некорректен (пустой, без обязательных колонок)."""
    pass


class ModelError(SalesAgentException):
    """Ошибка при обучении или прогнозировании модели."""
    pass


class ForecastError(ModelError):
    """Ошибка при построении прогноза."""
    pass


class BacktestError(SalesAgentException):
    """Ошибка при проведении backtest."""
    pass


class InvalidParameterError(SalesAgentException):
    """Ошибка в параметрах функции."""
    pass

