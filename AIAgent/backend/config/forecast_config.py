"""
Конфигурация модуля прогнозирования.

Все «магические числа» и настройки вынесены сюда.
Если нужно изменить лимит периодов или добавить алиас модели —
правим только этот файл.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ForecastConfig:
    """
    Настройки прогнозирования.

    frozen=True защищает от случайного изменения конфигурации в runtime.
    """

    # === Ограничения горизонта прогноза ===
    MIN_PERIODS: int = 7           # Минимум дней для прогноза
    MAX_PERIODS: int = 365         # Максимум дней (год)
    DEFAULT_PERIODS: int = 30      # Значение по умолчанию

    # === Поддерживаемые модели и их алиасы ===
    # Основные названия
    VALID_MODELS: tuple[str, ...] = ("neuralprophet", "sarima", "auto")

    # Алиасы для удобства пользователя (например, из промпта агента)
    MODEL_ALIASES: dict[str, str] = field(default_factory=lambda: {
        "neural": "neuralprophet",
        "np": "neuralprophet",
        "prophet": "neuralprophet",
        "arima": "sarima",
        "sarimax": "sarima",
        "best": "auto",
        "select": "auto",
    })

    # === Типы прогнозов ===
    VALID_FORECAST_TYPES: tuple[str, ...] = ("general", "by_store")

    # === Параметры auto-select модели ===
    # Сколько дней брать для тестирования при авто-выборе
    AUTO_BACKTEST_MIN_DAYS: int = 14
    AUTO_BACKTEST_MAX_DAYS: int = 60

    # === Системные настройки ===
    DEFAULT_SESSION_ID: str = "default"  # ID сессии по умолчанию
    CACHE_FORECAST: bool = True          # Кэшировать ли результат в Redis


# Глобальный экземпляр конфигурации — импортируем его везде
forecast_config = ForecastConfig()