"""Ядро интеллектуального агента."""
import logging
from typing import Optional, Dict, Any, List, Tuple

import uuid, re, json
from smolagents import CodeAgent, ToolCallingAgent

from agent.memory import SalesAgentMemory
from agent.models.yandex import YandexGPTModel
from agent.tools import load_dataset, get_dataset_info, build_forecast,get_forecast_summary,run_backtest,analyze_top_products,analyze_trends,analyze_kpi,analyze_seasonality,analyze_general

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


⚠️ КРИТИЧЕСКИ ВАЖНО — РАБОТА С ПРОГНОЗАМИ:

1️⃣ ПРОВЕРЬ КЭШ ПЕРЕД ВЫЧИСЛЕНИЯМИ:
   Сначала вызови get_forecast_summary(session_id=session_id)
   → Если там есть данные, НЕ вызывай build_forecast!
   → Это экономит 60-90 секунд

2️⃣ НЕ ИСПОЛЬЗУЙ model_type="auto":
   → По умолчанию: model_type="neuralprophet" (15-30 сек)
   → "auto" запускает backtest и работает в 3 раза дольше
   → Используй "auto" ТОЛЬКО через run_backtest() по явному запросу

3️⃣ ОДИН ПРОГНОЗ НА СЕССИЮ:
   → Не строй прогноз для 1 дня и отдельно для 7 дней
   → Построй один прогноз на 7 дней, возьми первый день из него
   → Каждый build_forecast = новое обучение модели!

4️⃣ ОДИН final_answer():
   → Вызывай final_answer() ТОЛЬКО ОДИН РАЗ в конце
   → Не вызывай его после каждого шага

✅ ПРАВИЛЬНЫЙ ПРИМЕР:
Thought: Пользователь просит прогноз на завтра и на неделю.
<code>
# Сначала проверим кэш
summary = get_forecast_summary(session_id=session_id)

# Если кэш пустой — строим прогноз один раз на 7 дней
if summary.get("status") == "no_forecast":
    forecast = build_forecast(periods=7, model_type="neuralprophet", session_id=session_id)
    summary = get_forecast_summary(session_id=session_id)

# Формируем ответ из одного прогноза
tomorrow = summary["average_daily"]  # Среднее за период
week = summary["total_forecast"]
final_answer(f"Прогноз на завтра: ~${tomorrow:,.2f}. На неделю: ${week:,.2f}")
</code>

❌ НЕПРАВИЛЬНО:
- build_forecast(periods=1) + build_forecast(periods=7) ← два обучения!
- model_type="auto" ← долго!
- final_answer() вызван дважды ← ошибка!
"""


def _extract_answer(result: Any) -> str:
    """Извлечь текст ответа агента."""

    if hasattr(result, "output"):
        return result.output

    if isinstance(result, dict):
        return result.get("answer", str(result))

    return str(result)

def _parse_model_actions(output: str) -> List[Dict[str, Any]]:
    """
    Парсинг выхода модели с несколькими JSON Action.
    Формат, который ожидаем:
    Action:
    { ... }

    Action:
    { ... }
    """
    actions = []
    # Ищем все блоки { ... } после Action:
    matches = re.findall(r'Action:\s*({.*?})\s*(?=Action:|$)', output, flags=re.DOTALL)
    for block in matches:
        try:
            action_dict = json.loads(block)
            actions.append(action_dict)
        except json.JSONDecodeError as e:
            logging.warning(f"Ошибка декодирования Action: {e} | блок: {block[:100]}...")
    return actions


class SmolSalesAgent:
    """Интеллектуальный агент для анализа и прогнозирования продаж."""

    # Базовый список инструментов
    TOOLS = [
        load_dataset,
        get_dataset_info,
        build_forecast,
        get_forecast_summary,
        run_backtest,
        analyze_top_products,
        analyze_trends,
        analyze_kpi,
        analyze_seasonality,
        analyze_general,
    ]

    def __init__(
            self,
            model_provider: str = "yandex",
            model_name: Optional[str] = None,
            session_id: Optional[str] = None,
            system_prompt: Optional[str] = None,
            use_code_agent: bool = False,
            max_steps_code_agent: int = 12,
            max_steps_tool_agent: int = 4,
            history_limit: int = 10,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = SalesAgentMemory(system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT)
        self.model = self._init_model(model_provider, model_name)
        self.use_code_agent = use_code_agent
        self.max_steps_code_agent = max_steps_code_agent
        self.max_steps_tool_agent = max_steps_tool_agent
        self.history_limit = history_limit
        self._forecast_cache: Dict[Tuple[int, str], dict] = {}

        AgentClass = CodeAgent if use_code_agent else ToolCallingAgent

        agent_kwargs = {}
        if use_code_agent:
            agent_kwargs["additional_authorized_imports"] = [
                "pandas",
                "numpy",
                "datetime",
                "math",
            ]

        self.agent = AgentClass(
            tools=self.TOOLS,
            model=self.model,
            instructions=self.memory.system_prompt,
            max_steps=self.max_steps_code_agent if use_code_agent else self.max_steps_tool_agent,
            verbosity_level=1,
            **agent_kwargs,
        )

        logger.info(
            f"✅ SmolSalesAgent создан | session={self.session_id} | CodeAgent={use_code_agent}"
        )

    @staticmethod
    def _init_model(provider: str, model_name: Optional[str]):
        """Инициализация LLM."""

        if provider == "yandex":
            if not settings.YANDEX_API_KEY or not settings.YANDEX_FOLDER_ID:
                raise ValueError("YANDEX_API_KEY и YANDEX_FOLDER_ID обязательны")

            return YandexGPTModel(
                api_key=settings.YANDEX_API_KEY,
                folder_id=settings.YANDEX_FOLDER_ID,
                model=model_name or settings.YANDEX_MODEL,
                temperature=0.2,
                max_tokens=2000,
            )

        if provider == "huggingface":
            from smolagents import TransformersModel
            import torch
            model_id = model_name or "mistralai/Mistral-7B-Instruct-v0.1"

            logger.info(f"🔄 Загружаем HF модель: {model_id}")
            return TransformersModel(
                model_id=model_id,
                max_new_tokens=512,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        raise ValueError(f"Неизвестный provider: {provider}")

    def run(self, query: str) -> Dict[str, Any]:
        """Обработать запрос пользователя."""

        try:
            self.memory.add("user", query)

            history = self.memory.get_history(limit=self.history_limit)

            result = self.agent.run(
                task=query,
                additional_args={"session_id": self.session_id, "history": history},
                max_steps=self.max_steps_code_agent if self.use_code_agent else self.max_steps_tool_agent,
            )

            answer = _extract_answer(result)
            self.memory.add("assistant", answer)

            if len(self.memory.get_history()) > self.history_limit * 2:
                self.memory.maybe_summarize()

            return {
                "status": "success",
                "answer": answer,
                "session_id": self.session_id,
            }

        except Exception as e:
            logger.error("❌ Ошибка выполнения агента", exc_info=True)

            return {
                "status": "error",
                "answer": "❌ Произошла ошибка обработки запроса.",
                "error": str(e),
                "session_id": self.session_id,
            }

    async def run_async(self, query: str) -> Dict[str, Any]:
        """Асинхронный запуск."""
        import asyncio
        return await asyncio.to_thread(self.run, query)

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Получить историю диалога."""
        return self.memory.get_history(limit)

    def clear_history(self):
        """Очистить историю."""
        self.memory.clear()