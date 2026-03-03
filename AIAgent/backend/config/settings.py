"""Переменные окружения и конфигурация приложения."""
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

class AppSettings:
    """Главные настройки приложения."""
    # FastAPI
    API_TITLE: str = "Sales Forecasting Agent API"
    API_DESCRIPTION: str = "Интеллектуальный агент для прогнозирования продаж"
    API_VERSION: str = "3.0"
    HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "yandex")
    YANDEX_API_KEY: Optional[str] = os.getenv("YANDEX_API_KEY")
    YANDEX_FOLDER_ID: Optional[str] = os.getenv("YANDEX_FOLDER_ID")
    YANDEX_MODEL: str = os.getenv("YANDEX_MODEL", "yandexgpt-lite")

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Модели прогнозирования
    DEFAULT_FORECAST_PERIODS: int = 30
    MAX_BACKTEST_DAYS: int = 365
    DEFAULT_MODEL: str = "neuralprophet"

    # CORS
    CORS_ORIGINS: list = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]

    # Логирование
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Таймауты
    UPLOAD_TIMEOUT: int = int(os.getenv("UPLOAD_TIMEOUT", "60"))
    CHAT_TIMEOUT: int = int(os.getenv("CHAT_TIMEOUT", "120"))
    FORECAST_TIMEOUT: int = int(os.getenv("FORECAST_TIMEOUT", "180"))

    # Пределы
    MAX_CSV_SIZE_MB: int = 100
    MAX_FORECAST_PERIODS: int = 365

    @classmethod
    def validate(cls) -> bool:
        """Проверить критические параметры."""
        if cls.LLM_PROVIDER == "yandex":
            if not cls.YANDEX_API_KEY or not cls.YANDEX_FOLDER_ID:
                raise ValueError(
                    "YANDEX_API_KEY и YANDEX_FOLDER_ID обязательны для provider='yandex'"
                )
        return True