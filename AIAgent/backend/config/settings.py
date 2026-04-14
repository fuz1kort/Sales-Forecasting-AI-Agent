"""Переменные окружения и конфигурация приложения."""
import os
from typing import Optional, List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import field_validator

load_dotenv()


class AppSettings(BaseSettings):
    """Главные настройки приложения."""

    # FastAPI
    api_title: str = "Sales Forecasting Agent API"
    api_description: str = "Интеллектуальный агент для прогнозирования продаж"
    api_version: str = "3.0"
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM
    llm_provider: str = "yandex"
    yandex_api_key: Optional[str] = None
    yandex_folder_id: Optional[str] = None
    yandex_model: str = "yandexgpt-lite"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    session_ttl_seconds: int = 86400  # 24 часа

    # Модели прогнозирования
    default_forecast_periods: int = 30
    max_backtest_days: int = 365
    default_model: str = "sarima"

    # CORS
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Логирование
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Таймауты
    upload_timeout: int = 60
    chat_timeout: int = 120
    forecast_timeout: int = 180
    yandex_request_timeout: int = 120

    # Пределы
    max_csv_size_mb: int = 100
    max_forecast_periods: int = 365

    class Config:
        """Конфигурация для Pydantic."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str, info):
        """Проверяет, что если provider=yandex, то заданы ключи."""
        if v == "yandex":
            data = info.data
            if not data.get("yandex_api_key") or not data.get("yandex_folder_id"):
                raise ValueError(
                    "YANDEX_API_KEY и YANDEX_FOLDER_ID обязательны для provider='yandex'"
                )
        return v

    # Compatability with old snake_case property names
    @property
    def API_TITLE(self) -> str:
        """Обратная совместимость."""
        return self.api_title

    @property
    def API_DESCRIPTION(self) -> str:
        return self.api_description

    @property
    def API_VERSION(self) -> str:
        return self.api_version

    @property
    def HOST(self) -> str:
        return self.host

    @property
    def PORT(self) -> int:
        return self.port

    @property
    def YANDEX_API_KEY(self) -> Optional[str]:
        return self.yandex_api_key

    @property
    def YANDEX_FOLDER_ID(self) -> Optional[str]:
        return self.yandex_folder_id

    @property
    def YANDEX_MODEL(self) -> str:
        return self.yandex_model

    @property
    def REDIS_HOST(self) -> str:
        return self.redis_host

    @property
    def REDIS_PORT(self) -> int:
        return self.redis_port

    @property
    def REDIS_DB(self) -> int:
        return self.redis_db

    @property
    def SESSION_TTL_SECONDS(self) -> int:
        return self.session_ttl_seconds

    @property
    def DEFAULT_FORECAST_PERIODS(self) -> int:
        return self.default_forecast_periods

    @property
    def MAX_BACKTEST_DAYS(self) -> int:
        return self.max_backtest_days

    @property
    def DEFAULT_MODEL(self) -> str:
        return self.default_model

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return self.cors_origins

    @property
    def CORS_ALLOW_CREDENTIALS(self) -> bool:
        return self.cors_allow_credentials

    @property
    def CORS_ALLOW_METHODS(self) -> List[str]:
        return self.cors_allow_methods

    @property
    def CORS_ALLOW_HEADERS(self) -> List[str]:
        return self.cors_allow_headers

    @property
    def LOG_LEVEL(self) -> str:
        return self.log_level

    @property
    def LOG_FORMAT(self) -> str:
        return self.log_format

    @property
    def UPLOAD_TIMEOUT(self) -> int:
        return self.upload_timeout

    @property
    def CHAT_TIMEOUT(self) -> int:
        return self.chat_timeout

    @property
    def FORECAST_TIMEOUT(self) -> int:
        return self.forecast_timeout

    @property
    def YANDEX_REQUEST_TIMEOUT(self) -> int:
        return self.yandex_request_timeout

    @property
    def MAX_CSV_SIZE_MB(self) -> int:
        return self.max_csv_size_mb

    @property
    def MAX_FORECAST_PERIODS(self) -> int:
        return self.max_forecast_periods

    @property
    def LLM_PROVIDER(self) -> str:
        return self.llm_provider