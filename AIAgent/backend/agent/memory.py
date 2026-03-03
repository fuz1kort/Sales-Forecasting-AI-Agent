"""
Модуль памяти агента.

Расширяет базовую память smolagents, добавляя:
- Контекст сессии (загруженные данные, настройки)
- Историю вызовов инструментов
- Автоматическое суммаризирование длинных диалогов
"""

from smolagents import AgentMemory
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SalesAgentMemory(AgentMemory):
    """
    Память агента для прогнозирования продаж.

    Хранит не только историю сообщений, но и:
    - Метаданные загруженных датасетов
    - Историю выполненных инструментов
    - Контекстные подсказки для LLM
    """

    def __init__(
            self,
            max_turns: int = 50,
            summarize_after: int = 30,
            system_prompt: Optional[str] = None
    ):
        """
        Инициализация памяти.

        Args:
            max_turns: Максимальное количество сообщений в истории
            summarize_after: После какого числа_turns начинать суммаризацию
        """
        super().__init__(system_prompt=system_prompt or "You are a helpful sales forecasting assistant.")
        self.max_turns = max_turns
        self.summarize_after = summarize_after
        self.tool_history: List[Dict[str, Any]] = []
        self._summary: Optional[str] = None
        self.history: List[Dict[str, str]] = []
        # Контекст данных (заполняется извне при загрузке датасета)
        self.dataset_context: Dict[str, Any] = {}

        logger.debug(f"SalesAgentMemory initialized: max_turns={max_turns}")

    def set_dataset_context(self, info: Dict[str, Any]):
        """
        Обновить контекст данных (вызывается после загрузки датасета).

        Args:
            info: Метаданные датасета (rows, columns, date_range, etc.)
        """
        self.dataset_context = info
        logger.debug(f"📊 Dataset context updated: {info.get('rows', 0)} rows")

    def add_tool_execution(self, tool_name: str, args: Dict, result: Dict):
        """
        Записать выполнение инструмента в историю.

        Args:
            tool_name: Название вызванного инструмента
            args: Аргументы вызова
            result: Результат выполнения
        """
        entry = {
            "tool": tool_name,
            "args": {k: v for k, v in args.items() if k != "csv_content"},  # Не логируем большие данные
            "success": "error" not in result,
            "result_summary": self._summarize_result(result)
        }
        self.tool_history.append(entry)

        # Ограничиваем размер истории инструментов
        if len(self.tool_history) > 20:
            self.tool_history.pop(0)

    @staticmethod
    def _summarize_result(result: Dict) -> str:
        """Создать краткое описание результата инструмента."""
        if "error" in result:
            return f"Ошибка: {result['error']}"

        if "forecast" in result and isinstance(result["forecast"], list):
            count = len(result["forecast"])
            total = sum(r.get("forecast", 0) for r in result["forecast"])
            return f"Прогноз на {count} периодов, сумма: {total:,.0f}"

        if "answer" in result:
            answer = result["answer"]
            return answer[:100] + "..." if len(answer) > 100 else answer

        return "Выполнено"

    def get_context_prefix(self) -> str:
        """
        Сформировать префикс контекста для промпта LLM.

        Returns:
            Строка с контекстной информацией
        """
        parts = []

        # Добавляем информацию о датасете из локального контекста
        if self.dataset_context:
            info = self.dataset_context
            if info.get("rows"):
                parts.append(f"📊 Загружено данных: {info['rows']} строк")
            if info.get("date_range"):
                parts.append(f"📅 Период: {info['date_range']}")
            if info.get("stores_count"):
                parts.append(f"🏪 Магазинов: {info['stores_count']}")

        # Добавляем историю инструментов
        if self.tool_history:
            recent = [t["tool"] for t in self.tool_history[-3:]]
            parts.append(f"🔧 Последние действия: {', '.join(recent)}")

        # Добавляем саммари длинного диалога
        if self._summary:
            parts.append(f"📝 Краткое содержание диалога:\n{self._summary}")

        return "\n".join(parts) if parts else ""

    def to_llm_messages(self) -> List[Dict[str, str]]:
        """
        Конвертировать память в формат сообщений для LLM.

        Returns:
            Список сообщений в формате [{"role": str, "content": str}, ...]
        """
        messages = []

        # Системный контекст
        context = self.get_context_prefix()
        if context:
            messages.append({
                "role": "system",
                "content": f"Контекст сессии:\n{context}"
            })

        # История диалога
        history = getattr(self, 'history', [])

        if len(history) > self.max_turns:
            history = history[-self.max_turns:]

        for turn in history:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", "")
            })

        return messages

    def maybe_summarize(self):
        """
        Попытаться создать саммари, если диалог стал длинным.

        Вызывается периодически для оптимизации контекста.
        """
        if len(self.history) >= self.summarize_after and not self._summary:
            # Здесь можно вызвать LLM для суммаризации
            # Пока просто ставим флаг
            self._summary = "Диалог содержит обсуждение прогноза продаж и аналитики."
            logger.info("Memory summarized after %d turns", len(self.history))

    def add(self, role: str, content: str) -> None:
        """
        Добавить сообщение в историю диалога.

        Args:
            role: Роль отправителя ('user', 'assistant', 'system')
            content: Текст сообщения
        """
        self.history.append({"role": role, "content": content})

        # Ограничиваем размер истории
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Получить историю диалога.

        Args:
            limit: Максимальное количество последних сообщений

        Returns:
            Список сообщений в формате [{"role": str, "content": str}, ...]
        """
        if limit is None:
            return list(self.history)
        return list(self.history)[-limit:]

    def clear(self):
        """Очистить всю память, включая историю инструментов."""
        self.history = []
        self.tool_history = []
        self._summary = None
        self.dataset_context = {}

        # Пытаемся очистить родительскую память, если метод существует
        # type: ignore[attr-defined] — метод может быть динамическим
        if hasattr(AgentMemory, 'clear') and callable(getattr(super(), 'clear', None)):
            try:
                super().clear()  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                pass  # Игнорируем, если у родителя нет реализации

        logger.debug("SalesAgentMemory cleared")
