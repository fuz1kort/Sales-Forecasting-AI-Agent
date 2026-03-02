"""
Прямой HTTP-клиент для YandexGPT API.
Минималистичная реализация для smolagents.
"""
import logging
import requests
from typing import Optional, List, Dict, Any

from smolagents import Model, ChatMessage

logger = logging.getLogger(__name__)

class _TU:
    """Микро-объект для токенов — чтобы smolagents не падал."""
    __slots__ = ("input_tokens", "output_tokens", "total_tokens")
    def __init__(self, inp, out):
        self.input_tokens = int(inp)
        self.output_tokens = int(out)
        self.total_tokens = self.input_tokens + self.output_tokens

class YandexResponse:
    """
    Стандартный ответ для smolagents.

    Обязательные атрибуты (требуются фреймворком):
    - content: str
    - output: str
    - input_tokens: int
    - output_tokens: int
    - token_usage: dict
    - role: str
    - tool_calls: Optional[List]
    """
    __slots__ = (
        "content", "output", "input_tokens", "output_tokens",
        "token_usage", "role", "tool_calls"
    )

    def __init__(
            self,
            text: str,
            input_tokens: int = 0,
            output_tokens: int = 0,
    ):
        self.content = text
        self.output = text
        self.input_tokens = int(input_tokens)
        self.output_tokens = int(output_tokens)
        self.token_usage = _TU(input_tokens, output_tokens)
        self.role = "assistant"
        self.tool_calls = None

    def __str__(self) -> str:
        return self.content


class YandexGPTModel(Model):
    """Модель для работы с YandexGPT через прямой HTTP-запрос."""

    def __init__(
            self,
            api_key: str,
            folder_id: str,
            model: str = "yandexgpt-lite",
            temperature: float = 0.7,
            max_tokens: int = 2000,
            api_endpoint: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.folder_id = folder_id
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_endpoint = api_endpoint.rstrip("/")
        self.supports_tools = True
        logger.info(f"✅ YandexGPTModel инициализирован: {model} в каталоге {folder_id}")

    def generate(
            self,
            messages,
            stop_sequences: Optional[List[str]] = None,
            **kwargs
    ) -> YandexResponse:
        """
        Отправить запрос в YandexGPT и получить ответ.
        Возвращает YandexResponse со всеми обязательными атрибутами.
        """


        yandex_messages = self._convert_messages(messages)

        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_name}",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            },
            "messages": yandex_messages,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-folder-id": self.folder_id,
        }

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content = self._parse_response(result)

            # Пытаемся извлечь реальные токены из ответа Yandex
            usage = result.get("result", {}).get("usage", {})
            input_tokens = usage.get("inputTextTokens", 0)
            output_tokens = usage.get("completionTokens", 0)

            return YandexResponse(
                text=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            logger.error(f"YandexGPT request failed: {e}")
            return YandexResponse(text=f"Ошибка: {e}")

    def __call__(
            self,
            messages: List[ChatMessage],
            stop_sequences: Optional[List[str]] = None,
            **kwargs
    ) -> YandexResponse:
        """Делегирует generate() — не нужно усложнять."""
        return self.generate(messages, stop_sequences, **kwargs)

    @staticmethod
    def _convert_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Конвертировать сообщения в формат Yandex API."""
        yandex_msgs = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)

            if isinstance(msg.content, list):
                text = " ".join(
                    part.get("text", "")
                    for part in msg.content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            elif isinstance(msg.content, str):
                text = msg.content
            else:
                text = str(msg.content)

            if role == "system":
                yandex_msgs.append({"role": "user", "text": f"Инструкция: {text}"})
            elif role in ("user", "assistant"):
                yandex_msgs.append({"role": role, "text": text})
            else:
                yandex_msgs.append({"role": "user", "text": text})

        return yandex_msgs

    @staticmethod
    def _parse_response(result: Dict[str, Any]) -> str:
        """Распарсить JSON-ответ от YandexGPT API."""
        if "result" not in result:
            raise ValueError(f"Неверный формат ответа: {result}")
        alternatives = result["result"].get("alternatives", [])
        if not alternatives:
            raise ValueError("Пустой ответ от YandexGPT")
        content = alternatives[0].get("message", {}).get("text", "")
        if not content:
            raise ValueError("Пустой текст в ответе")
        return content