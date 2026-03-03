"""Ядро интеллектуального агента."""
import logging
from typing import Optional, Dict, Any, List

from smolagents import CodeAgent, ToolCallingAgent, PromptTemplates

from agent.memory import SalesAgentMemory
from agent.models.yandex import YandexGPTModel
from agent.tools.analyze import (
    analyze_top_products_tool,
    analyze_trends_tool,
    analyze_kpi_tool,
    analyze_seasonality_tool,
    analyze_general_tool,
)
from agent.tools.backtest import run_backtest_tool
from agent.tools.data import load_dataset, get_dataset_info
from agent.tools.forecast import build_forecast, get_forecast_summary
from config import AppSettings


logger = logging.getLogger(__name__)

settings = AppSettings()

DEFAULT_SYSTEM_PROMPT = """
Ты — интеллектуальный ассистент для прогнозирования и анализа продаж.

🗣️ ЯЗЫК ОТВЕТА:
- ВСЕГДА отвечай на РУССКОМ языке, независимо от языка запроса или инструментов.
- ВСЕГДА передавай session_id во все инструменты
- Даже если инструмент вернул текст на английском — переведи его и дай ответ на русском.
- Используй эмодзи для наглядности: 📊 📈 💰 🔍 ✅ ❌

📋 ПРАВИЛА:
1. Сначала проверяй наличие данных (get_dataset_info)
2. Если нет данных → попроси загрузить CSV на русском
3. Используй инструменты для анализа, но интерпретируй результаты на русском
4. Отвечай кратко, по делу, структурированно
5. При ошибке объясни причину на русском и предложи решение

Доступные инструменты:
📦 load_dataset(csv_content) — загрузка CSV
🔮 build_forecast(periods, model_type, forecast_type) — прогноз
🧪 run_backtest_tool(test_days) — сравнение моделей
📊 analyze_top_products_tool(limit) — топ товаров
📈 analyze_trends_tool(period) — тренды
🎯 analyze_kpi_tool() — KPI
🌟 analyze_seasonality_tool() — сезонность
ℹ️ analyze_general_tool() — общие выводы
"""

class SmolSalesAgent:
    """Интеллектуальный агент для прогнозирования продаж."""

    def __init__(
            self,
            model_provider: str = "yandex",
            model_name: Optional[str] = None,
            session_id: Optional[str] = None,
            system_prompt: Optional[str] = None,
            use_code_agent: bool = True
    ):
        self.use_code_agent = use_code_agent
        self.memory = SalesAgentMemory(system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT)
        self.model = self._init_model(model_provider, model_name)
        self.session_id = session_id

        # Базовый список инструментов
        self.tools = [
            load_dataset,
            get_dataset_info,
            build_forecast,
            get_forecast_summary,
            run_backtest_tool,
            analyze_top_products_tool,
            analyze_trends_tool,
            analyze_kpi_tool,
            analyze_seasonality_tool,
            analyze_general_tool,
        ]

        AgentClass = CodeAgent if use_code_agent else ToolCallingAgent
        agent_kwargs = {}
        if use_code_agent:
            agent_kwargs = {"additional_authorized_imports": ["pandas", "numpy", "datetime", "math"]}

        self.agent = AgentClass(
            tools=self.tools,
            model=self.model,
            max_steps=8 if use_code_agent else 3,
            verbosity_level=1,
            **agent_kwargs
        )

        logger.info(f"✅ SmolSalesAgent создан (CodeAgent={use_code_agent})")

    def _init_model(self, provider: str, model_name: Optional[str]):
        """Инициализация модели."""
        if provider == "yandex":
            api_key = settings.YANDEX_API_KEY
            folder_id = settings.YANDEX_FOLDER_ID

            if not api_key or not folder_id:
                logger.error("❌ YANDEX_API_KEY и YANDEX_FOLDER_ID обязательны")
                raise ValueError(
                    "YANDEX_API_KEY и YANDEX_FOLDER_ID обязательны для provider='yandex'"
                )

            return YandexGPTModel(
                api_key=api_key,
                folder_id=folder_id,
                model=model_name or settings.YANDEX_MODEL,
                temperature=0.7,
                max_tokens=2000,
            )

        elif provider == "huggingface":
            from smolagents import TransformersModel
            import torch

            model_id = model_name or "mistralai/Mistral-7B-Instruct-v0.1"
            logger.info(f"🔄 Загружаем HF модель: {model_id}")

            return TransformersModel(
                model_id=model_id,
                max_new_tokens=512,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        raise ValueError(f"Неизвестный провайдер: {provider}")

    def run(self, query: str) -> Dict[str, Any]:
        """Обработать запрос пользователя."""
        try:
            self.memory.add("user", query)
            result = self.agent.run(query, additional_args={"session_id": self.session_id})

            if hasattr(result, "output"):
                answer = result.output
            elif isinstance(result, dict) and "answer" in result:
                answer = result["answer"]
            else:
                answer = str(result)

            self.memory.add("assistant", answer)
            self.memory.maybe_summarize()

            return {
                "answer": answer,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения агента: {e}", exc_info=True)
            return {
                "answer": "❌ Произошла ошибка. Попробуйте переформулировать запрос.",
                "status": "error",
                "error": str(e)
            }

    async def run_async(self, query: str) -> Dict[str, Any]:
        """Асинхронная обработка."""
        import asyncio
        return await asyncio.to_thread(self.run, query)

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Получить историю диалога."""
        return self.memory.get_history(limit=limit)

    def clear_history(self):
        """Очистить историю."""
        self.memory.clear()