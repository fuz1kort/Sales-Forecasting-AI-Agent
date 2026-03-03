"""
Фабрика для создания моделей LLM.

Централизует логику инициализации различных провайдеров моделей.
"""

import logging
from typing import Optional

from agent.models.yandex import YandexGPTModel
from config import AppSettings

logger = logging.getLogger(__name__)
settings = AppSettings()


def create_model(provider: str, model_name: Optional[str] = None):
    """
    Создать модель LLM на основе провайдера.

    Args:
        provider: Провайдер модели ("yandex", "huggingface", "openai")
        model_name: Конкретная модель (опционально)

    Returns:
        Экземпляр модели

    Raises:
        ValueError: Если конфигурация неполна или провайдер не поддерживается
    """

    if provider == "yandex":
        return _create_yandex_model(model_name)

    elif provider == "huggingface":
        return _create_huggingface_model(model_name)

    elif provider == "openai":
        return _create_openai_model(model_name)

    else:
        logger.error(f"❌ Неизвестный провайдер: {provider}")
        raise ValueError(f"Провайдер {provider} не поддерживается")


def _create_yandex_model(model_name: Optional[str] = None):
    """Создать модель Yandex GPT."""
    if not settings.YANDEX_API_KEY or not settings.YANDEX_FOLDER_ID:
        logger.error("❌ YANDEX_API_KEY и YANDEX_FOLDER_ID обязательны")
        raise ValueError(
            "YANDEX_API_KEY и YANDEX_FOLDER_ID обязательны для provider='yandex'"
        )

    logger.info("🟡 Инициализируем Yandex GPT")
    return YandexGPTModel(
        api_key=settings.YANDEX_API_KEY,
        folder_id=settings.YANDEX_FOLDER_ID,
        model=model_name or settings.YANDEX_MODEL,
        temperature=0.7,
        max_tokens=2000,
    )


def _create_huggingface_model(model_name: Optional[str] = None):
    """Создать локальную модель HuggingFace."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("❌ HuggingFace модели требуют: pip install transformers huggingface-hub")
        raise ImportError("Требуются пакеты: transformers, huggingface-hub")

    model_id = model_name or "mistralai/Mistral-7B-Instruct-v0.1"
    logger.info(f"🟣 Загружаем модель HuggingFace: {model_id}")

    # Упрощённая реализация — в реальности нужна более сложная логика
    # для работы с локальными моделями
    class HFModel:
        def __init__(self, model_id):
            self.model_id = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)

    return HFModel(model_id)


def _create_openai_model(model_name: Optional[str] = None):
    """Создать модель OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("❌ OpenAI требует: pip install openai")
        raise ImportError("Требуется пакет: openai")

    api_key = settings.OPENAI_API_KEY if hasattr(settings, "OPENAI_API_KEY") else None
    if not api_key:
        logger.error("❌ OPENAI_API_KEY не установлен")
        raise ValueError("OPENAI_API_KEY обязателен для provider='openai'")

    logger.info("🔵 Инициализируем OpenAI")
    return OpenAI(api_key=api_key)

