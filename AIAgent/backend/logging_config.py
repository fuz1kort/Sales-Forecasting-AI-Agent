"""
Конфигурация логирования приложения.
"""

import logging
import logging.handlers
import os
from config import AppSettings

settings = AppSettings()


def setup_logging(log_file: str = "logs/app.log") -> None:
    """
    Настроить логирование для приложения.

    Args:
        log_file: Путь к файлу логов
    """
    # Создаём папку логов если её нет
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Форматер логов
    formatter = logging.Formatter(settings.LOG_FORMAT)

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)

    # Удаляем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Обработчик для файла (с ротацией)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    root_logger.info(f"✅ Логирование инициализировано (уровень: {settings.LOG_LEVEL})")


def get_logger(name: str) -> logging.Logger:
    """Получить логгер с указанным именем."""
    return logging.getLogger(name)

