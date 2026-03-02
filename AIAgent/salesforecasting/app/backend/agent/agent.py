# app/backend/agent/agent.py
"""
Ядро интеллектуального агента для прогнозирования продаж.

Использует smolagents для автономного выполнения задач.
"""
import logging
import os
from typing import Optional, Dict, Any, List

from smolagents import CodeAgent, ToolCallingAgent

from salesforecasting.app.backend.agent.memory import SalesAgentMemory
from salesforecasting.app.backend.agent.models.yandex import YandexGPTModel
from salesforecasting.app.backend.agent.tools.analyze import (
    analyze_top_products_tool,
    analyze_trends_tool,
    analyze_kpi_tool,
    analyze_seasonality_tool,
    analyze_general_tool,
)
from salesforecasting.app.backend.agent.tools.backtest import run_backtest_tool
from salesforecasting.app.backend.agent.tools.data import load_dataset, get_dataset_info
from salesforecasting.app.backend.agent.tools.forecast import build_forecast, get_forecast_summary

logger = logging.getLogger(__name__)

# Системный промпт — определяет поведение агента
DEFAULT_SYSTEM_PROMPT = """
Ты — интеллектуальный ассистент для прогнозирования и анализа продаж.

Правила форматирования кода:
- Код ВСЕГДА оборачивай в теги <code> и </code>
- НЕ используй markdown-блоки ```python ... ```
- Пример правильного формата:

Thought: Я проанализирую данные
<code>
info = get_dataset_info()
print(info)
result = analyze_general_tool()
final_answer(result)
</code>

- Закрывай каждый <code> тегом </code>

Доступные инструменты:
📦 load_dataset(csv_content, encoding) — загрузка CSV
🔮 build_forecast(periods, model_type, forecast_type, store_ids) — прогноз
🧪 run_backtest_tool(test_days) — сравнение моделей (backtest)
📊 analyze_top_products_tool(limit) — топ товаров
📈 analyze_trends_tool(period) — тренды продаж
🎯 analyze_kpi_tool() — дашборд метрик
🌟 analyze_seasonality_tool() — сезонность
ℹ️ analyze_general_tool() — общие выводы
ℹ️ get_dataset_info() — метаданные текущего датасета

Правила:
1. Сначала проверяй, загружены ли данные (get_dataset_info), прежде чем анализировать
2. Если данных нет — попроси пользователя загрузить CSV через load_dataset
3. Отвечай на русском, кратко и по делу, с эмодзи для наглядности
4. Используй инструменты — не выдумывай цифры и факты
5. При ошибке инструмента — объясни причину простыми словами и предложи решение
6. Для сложных вычислений пиши Python-код (у тебя есть доступ к pandas)

Пример диалога:
Пользователь: "Покажи топ-5 товаров"
Ты: [вызываешь analyze_top_products_tool(limit=5)]
Ты: "✅ Топ-5 товаров по продажам:
1. 🥇 Товар А — $10,234 (156 заказов)
2. 🥈 Товар Б — $8,901 (132 заказа)
..."

Пример:
Пользователь: "Какая модель лучше для моих данных?"
Ты: [вызываешь run_backtest_tool(test_days=30)]
Ты: "🏆 По результатам backtest на 30 днях:
• NeuralProphet: MAPE 5.2%, MAE $123
• SARIMA: MAPE 7.8%, MAE $189
Лучшая модель: **NeuralProphet**"

🧪 Когда возвращаешь результат backtest:
1. Сначала напиши краткий вывод на русском: какая модель лучше, ключевые метрики
2. В конце добавь JSON-блок с полными данными для визуализации:

```json
{
  "status": "success",
  "best_model": "sarima",
  "metrics": {
    "sarima": {"mae": 735.96, "mape": 0.03},
    "neuralprophet": {"mae": 892.11, "mape": 0.05}
  },
  "predictions": {
    "date": ["2024-01-01", ...],
    "actual": [1000, ...],
    "sarima": [1020, ...],
    "neuralprophet": [980, ...]
  }
}
"""


class SmolSalesAgent:
    """
    Интеллектуальный агент для прогнозирования продаж.

    Атрибуты:
        agent: Экземпляр CodeAgent или ToolCallingAgent из smolagents
        memory: Экземпляр SalesAgentMemory
        tools: Список доступных инструментов
        use_code_agent: Флаг использования CodeAgent (умнее, но медленнее)
    """

    def __init__(
            self,
            model_provider: str = "yandex",
            model_name: Optional[str] = None,
            system_prompt: Optional[str] = None,
            use_code_agent: bool = True  # ← По умолчанию CodeAgent!
    ):
        """
        Инициализация агента.

        Args:
            model_provider: "yandex", "huggingface", "openai"
            model_name: Конкретная модель (опционально)
            system_prompt: Кастомный системный промпт
            use_code_agent: Использовать CodeAgent (умнее) или ToolCallingAgent (быстрее)
        """
        self.use_code_agent = use_code_agent

        self.memory = SalesAgentMemory(
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT
        )

        # Инициализация модели
        self.model = self._init_model(model_provider, model_name)

        # Список инструментов
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

        # Выбор класса агента
        AgentClass = CodeAgent if use_code_agent else ToolCallingAgent

        # Дополнительные параметры для CodeAgent
        agent_kwargs = {}
        if use_code_agent:
            agent_kwargs = {
                "additional_authorized_imports": ["pandas", "numpy", "datetime", "math"]
            }
            logger.info("🧠 Используем CodeAgent (с выполнением Python-кода)")
        else:
            logger.info("⚡ Используем ToolCallingAgent (только вызов инструментов)")

        # Создание агента smolagents
        self.agent = AgentClass(
            tools=self.tools,
            model=self.model,
            max_steps=15 if use_code_agent else 10,  # CodeAgent может делать больше шагов
            verbosity_level=1,
            **agent_kwargs
        )

        logger.info(f"✅ SmolSalesAgent создан (CodeAgent={use_code_agent})")

    def _init_model(
            self,
            provider: str,
            model_name: Optional[str]
    ):
        """Инициализировать модель: Yandex (HTTP) или HuggingFace (локально)."""

        # ============ YANDEX GPT (HTTP) ============
        if provider == "yandex":
            api_key = os.getenv("YANDEX_API_KEY")
            folder_id = os.getenv("YANDEX_FOLDER_ID")

            if not api_key or not folder_id:
                logger.error("❌ YANDEX_API_KEY и YANDEX_FOLDER_ID обязательны")
                # Fallback на HF, если критично
                return self._init_model("huggingface", model_name)

            return YandexGPTModel(
                api_key=api_key,
                folder_id=folder_id,
                model=model_name or "yandexgpt-lite",
                temperature=0.7,
                max_tokens=2000,
            )

        # ============ HUGGINGFACE (локально, опционально) ============
        elif provider == "huggingface":
            from smolagents import TransformersModel
            import torch

            model_id = model_name or "Qwen/Qwen2.5-Coder-1.5B-Instruct"
            logger.info(f"🔄 Загрузка HF модели: {model_id}")

            return TransformersModel(
                model_id=model_id,
                max_new_tokens=512,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        return None

    def run(self, query: str) -> Dict[str, Any]:
        """
        Обработать запрос пользователя.

        Args:
            query: Текст запроса на естественном языке

        Returns:
            Словарь с результатом выполнения
        """
        try:
            # Сохраняем запрос в память
            self.memory.add("user", query)

            # Выполняем агента
            result = self.agent.run(query)

            logger.info(f"📤 Результат агента: {type(result)}")

            # Извлекаем ответ (разные форматы у CodeAgent / ToolCallingAgent)
            if hasattr(result, "output"):
                answer = result.output
            elif isinstance(result, dict) and "answer" in result:
                answer = result["answer"]
            else:
                answer = str(result)

            # Сохраняем ответ в память
            self.memory.add("assistant", answer)
            self.memory.maybe_summarize()

            return {
                "answer": answer,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"❌ Ошибка выполнения агента: {e}", exc_info=True)

            return {
                "answer": "❌ Произошла ошибка при обработке запроса. Попробуйте переформулировать или упростить задачу.",
                "status": "error",
                "error": str(e)
            }

    async def run_async(self, query: str) -> Dict[str, Any]:
        """Асинхронная обработка запроса."""
        import asyncio
        return await asyncio.to_thread(self.run, query)

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Получить историю диалога."""
        return self.memory.get_history(limit=limit)

    def clear_history(self):
        """Очистить историю диалога."""
        self.memory.clear()

    def get_tools_list(self) -> List[str]:
        """Вернуть список доступных инструментов."""
        return [t.name for t in self.tools]

    def update_dataset_context(self, info: Dict[str, Any]):
        """
        Обновить контекст данных в памяти (вызывается после загрузки датасета).

        Args:
            info: Метаданные датасета (rows, columns, date_range, etc.)
        """
        self.memory.set_dataset_context(info)
        logger.debug(f"📊 Dataset context updated in memory: {info.get('rows', 0)} rows")