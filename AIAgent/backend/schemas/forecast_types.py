"""
Типы данных для модуля прогнозирования.

Используем TypedDict для описания структуры словарей,
которые возвращают функции прогноза. Это даёт:
- Подсказки в IDE (автокомплит)
- Проверку типов через mypy
- Документацию прямо в коде
"""
from typing import List, Optional, TypedDict, Literal, Union


class ForecastPoint(TypedDict):
    """
    Одна точка прогноза.
    
    Пример: {"date": "2024-01-15", "forecast": 1250.50}
    """
    date: str                    # Дата в формате YYYY-MM-DD
    forecast: float              # Прогнозируемое значение
    lower_bound: Optional[float] # Нижняя граница доверительного интервала
    upper_bound: Optional[float] # Верхняя граница доверительного интервала


class ForecastMetrics(TypedDict, total=False):
    """
    Метрики качества модели.
    
    total=False означает, что все поля опциональны —
    не все модели возвращают полный набор метрик.
    """
    mape: Optional[float]        # Mean Absolute Percentage Error
    mae: Optional[float]         # Mean Absolute Error
    rmse: Optional[float]        # Root Mean Square Error
    model_type: str              # Название использованной модели
    training_time: Optional[float]  # Время обучения в секундах


class ForecastResult(TypedDict, total=False):
    """
    Результат выполнения прогноза.
    
    Это основной контракт между слоями:
    - tools → service → models → service → tools
    """
    # Статус выполнения
    status: Literal["success", "error", "no_data", "empty"]

    # Основные данные (присутствуют при успехе)
    forecast: Optional[List[ForecastPoint]]  # Список точек прогноза
    metrics: Optional[ForecastMetrics]       # Метрики модели

    # Информация об ошибке (присутствует при статусе "error")
    error: Optional[str]

    # Дополнительная мета-информация
    info: Optional[dict]  # model_used, periods, store_ids и т.д.


class ForecastSummary(TypedDict):
    """
    Краткая сводка по прогнозу для отображения пользователю.
    
    Используется в инструменте get_forecast_summary.
    """
    status: str                  # "success" | "no_forecast" | "empty"
    periods: int                 # Количество дней в прогнозе
    total_forecast: float        # Сумма всех прогнозируемых значений
    average_daily: float         # Среднее дневное значение
    min_daily: float             # Минимальное дневное значение
    max_daily: float             # Максимальное дневное значение
    first_date: str              # Дата начала прогноза
    last_date: str               # Дата окончания прогноза