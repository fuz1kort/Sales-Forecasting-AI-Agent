"""Ядро интеллектуального агента."""
import logging
from typing import Optional, Dict, Any, List, Tuple

import uuid, re, json
from smolagents import CodeAgent, ToolCallingAgent

from backend.agent.memory import SalesAgentMemory
from backend.agent.models.yandex import YandexGPTModel
from backend.agent.state import get_session_manager
from backend.agent.tools import load_dataset, get_dataset_info, build_forecast,get_forecast_summary,run_backtest,analyze_top_products,analyze_trends,analyze_kpi,analyze_seasonality,analyze_general,analyze_stationarity_tool,visualize_correlations,visualize_distributions,visualize_time_series,visualize_top_products,visualize_abc_analysis,analyze_product_by_name

from backend.config import AppSettings

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
6. **НЕ УСТАНАВЛИВАЙ ДАТЫ ПО УМОЛЧАНИЮ** — если пользователь не указал даты, передавай date_from=None, date_to=None
7. **ДАННЫЕ АВТОМАТИЧЕСКИ ОЧИЩАЮТСЯ ПРИ ЗАГРУЗКЕ** — плохие товары, нулевые транзакции удаляются сразу
8. **ОЧЕНЬ ВАЖНО: Когда пользователь спрашивает о КОНКРЕТНОМ ТОВАРЕ (по названию)** — ВСЕГДА используй `analyze_product_by_name`, а НЕ `analyze_top_products`! Например:
   - "расскажи про продажи PEN, 10 COLOURS" → analyze_product_by_name("PEN, 10 COLOURS")
   - "какие продажи у товара X" → analyze_product_by_name("X")
   - "сколько продано REGENCY CAKESTAND" → analyze_product_by_name("REGENCY CAKESTAND")
   - "информация о товаре Y" → analyze_product_by_name("Y")

Доступные инструменты:
📦 load_dataset(csv_content, use_builtin_data=False) — загрузка CSV или встроенных данных магазина
� analyze_product_by_name(product_name) — информация о конкретном товаре по названию (ИСПОЛЬЗУЙ ЭТОТ инструмент, когда пользователь спрашивает о конкретном товаре типа "расскажи про PEN, 10 COLOURS")
📊 analyze_top_products(limit, sort_by, date_from=None, date_to=None) — топ товаров (данные уже очищены от мусора)
📈 analyze_trends(period) — тренды
🎯 analyze_kpi() — KPI
🌟 analyze_seasonality() — сезонность
ℹ️ analyze_general() — общие выводы
🧪 analyze_stationarity_tool() — тест ADF на стационарность
🔮 build_forecast(periods, model_type, forecast_type) — прогноз
🧪 run_backtest(test_days) — сравнение моделей
📈 visualize_correlations() — матрица корреляций
📊 visualize_distributions() — распределения и выбросы
📈 visualize_time_series() — временной ряд
📦 visualize_top_products(limit) — топ-продукты
🔤 visualize_abc_analysis() — ABC-анализ


⚠️ КРИТИЧЕСКИ ВАЖНО — РАБОТА С ПРОГНОЗАМИ:

1️⃣ ПРОВЕРЬ КЭШ ПЕРЕД ВЫЧИСЛЕНИЯМИ:
   Сначала вызови get_forecast_summary(session_id=session_id)
   → Если там есть данные, НЕ вызывай build_forecast!
   → Это экономит 60-90 секунд

2️⃣ ВСЕГДА используй model_type="auto" для автоматического выбора лучшей модели:
   → По умолчанию для общего прогноза: model_type="auto" (авто-выбор лучшей модели по backtest)
   → "auto" запускает backtest и работает дольше, но выбирает лучшую модель
   → Используй конкретную модель ("prophet", "sarima", "ensemble") ТОЛЬКО если пользователь явно просит

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
    forecast = build_forecast(periods=7, model_type="ensemble", session_id=session_id)
    summary = get_forecast_summary(session_id=session_id)

# Формируем ответ из одного прогноза
tomorrow = summary["average_daily"]  # Среднее за период
week = summary["total_forecast"]
final_answer(f"Прогноз на завтра: ~${tomorrow:,.2f}. На неделю: ${week:,.2f}")
</code>

✅ ПРАВИЛЬНЫЙ АНАЛИЗ ТОВАРОВ:
Thought: Пользователь просит топ худших товаров без указания дат.
<code>
# Данные уже очищены при загрузке (плохие товары и нулевые транзакции удалены)
# НЕ устанавливаем даты по умолчанию — передаём None
result = analyze_top_products(limit=20, sort_by="bottom", session_id=session_id)
final_answer(result)
</code>

❌ НЕПРАВИЛЬНО:
- analyze_top_products(date_from="2021-01-01", date_to="2023-12-31") ← не устанавливать даты, если не просили!
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
        analyze_product_by_name,
        analyze_trends,
        analyze_kpi,
        analyze_seasonality,
        analyze_general,
        analyze_stationarity_tool,
        visualize_correlations,
        visualize_distributions,
        visualize_time_series,
        visualize_top_products,
        visualize_abc_analysis,
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
            max_execution_time: int = 300,  # 5 минут для backtest
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = SalesAgentMemory(system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT)
        self.model = self._init_model(model_provider, model_name)
        self.use_code_agent = use_code_agent
        self.max_steps_code_agent = max_steps_code_agent
        self.max_steps_tool_agent = max_steps_tool_agent
        self.history_limit = history_limit
        self._forecast_cache: Dict[Tuple[int, str], dict] = {}

        self._restore_history()

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
                request_timeout=settings.YANDEX_REQUEST_TIMEOUT,
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
            get_session_manager().add_message(self.session_id, "user", query)

            history = self.memory.get_history(limit=self.history_limit)

            logger.info(
                f"🚀 Running agent task for session={self.session_id[:8]} use_code_agent={self.use_code_agent} "
                f"max_steps={self.max_steps_code_agent if self.use_code_agent else self.max_steps_tool_agent}"
            )
            result = self.agent.run(
                task=query,
                additional_args={"session_id": self.session_id, "history": history},
                max_steps=self.max_steps_code_agent if self.use_code_agent else self.max_steps_tool_agent,
            )
            logger.info(f"✅ Agent finished task for session={self.session_id[:8]} result_keys={list(result.keys()) if isinstance(result, dict) else type(result)}")

            answer = _extract_answer(result)
            self.memory.add("assistant", answer)
            get_session_manager().add_message(self.session_id, "assistant", answer)

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

    def _restore_history(self) -> None:
        """Восстанавливает историю диалога из SessionManager для текущей сессии."""
        try:
            history = get_session_manager().get_history(self.session_id)
            if history:
                self.memory.history = history[-self.history_limit:] if self.history_limit else history
                logger.debug(f"🔄 Восстановлена история сессии {self.session_id[:8]} ({len(self.memory.history)} сообщений)")
        except Exception as e:
            logger.warning(f"Не удалось восстановить историю сессии {self.session_id}: {e}")

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Получить историю диалога."""
        return self.memory.get_history(limit)

    def clear_history(self):
        """Очистить историю."""
        self.memory.clear()