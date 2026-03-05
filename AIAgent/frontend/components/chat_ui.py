"""Интерфейс чата с агентом."""
from datetime import datetime

import pandas as pd
import streamlit as st
import json
from matplotlib import pyplot as plt


def render_chat_section(api_client):
    """Рендер секции чата с блокировкой во время запроса."""
    st.header("💬 Чат с аналитиком")

    # Флаг блокировки
    if "loading" not in st.session_state:
        st.session_state.loading = False

    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Начните диалог или задайте вопрос из списка ниже!")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    render_agent_response(msg["content"], msg.get("is_code", False))

    # Блокируем поле ввода если идет запрос
    if st.session_state.loading:
        st.chat_input("⏳ Подождите, агент обрабатывает запрос...", disabled=True)
        return

    prompt = st.chat_input("Спросите о продажах, прогнозе или данных...")

    if prompt:
        # Устанавливаем блокировку
        st.session_state.loading = True

        # Добавляем сообщение пользователя
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Отправляем запрос
        try:
            with st.spinner("🤖 Агент анализирует данные..."):
                response = api_client.send_query(prompt)

                if response.get("status") == "success":
                    answer = response.get("answer", "")

                    with st.chat_message("assistant"):
                        render_agent_response(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "is_code": "<code>" in answer
                    })
                else:
                    error_msg = response.get("error", "Неизвестная ошибка")
                    st.error(f"❌ Ошибка: {error_msg}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"⚠️ Не удалось обработать запрос: {error_msg}",
                        "is_code": False
                    })
        finally:
            # Снимаем блокировку
            st.session_state.loading = False


def render_agent_response(text: str, is_code: bool = False):
    """Форматирует ответ агента для отображения и строит график/таблицу, если это прогноз."""
    if not text:
        return

    # Попробуем распарсить прогноз
    forecast_result = try_parse_forecast(text)
    if forecast_result:
        st.subheader("📊 Таблица прогноза")
        render_forecast_table(forecast_result)
        st.subheader("📈 График прогноза")
        plot_forecast(forecast_result)
        return

    # Иначе это обычный текст/markdown
    safe_text = format_markdown_safe(text)

    # Если есть блок кода <code>...</code> — обрабатываем отдельно
    if is_code:
        safe_text = format_code_blocks(safe_text)

    st.markdown(safe_text)

def render_forecast_table(forecast_result):
    if forecast_result["status"] != "success" or not forecast_result.get("forecast"):
        st.info("Нет данных для прогноза")
        return

    table = pd.DataFrame(forecast_result["forecast"])
    table["date"] = pd.to_datetime(table["date"])
    st.table(table)

def try_parse_forecast(text: str) -> dict | None:
    """Пробуем распознать JSON-строку с прогнозом."""
    if isinstance(text, dict) and "forecast" in text:
        return text

    try:
        # Иногда агент возвращает JSON в виде строки
        parsed = json.loads(text)
        if "forecast" in parsed:
            return parsed
    except Exception:
        return None

    return None


def plot_forecast(forecast_result, n_days: int = 30):
    """Построение графика прогноза."""
    forecast_data = forecast_result.get("forecast", [])[:n_days]
    if not forecast_data:
        st.info("Нет данных для построения графика")
        return

    dates = [datetime.strptime(day["date"], "%Y-%m-%d") for day in forecast_data]
    values = [day["forecast"] for day in forecast_data]
    lower = [day.get("lower_bound", v) for v, day in zip(values, forecast_data)]
    upper = [day.get("upper_bound", v) for v, day in zip(values, forecast_data)]

    plt.figure(figsize=(10,5))
    plt.plot(dates, values, label="Прогноз", color="blue", marker="o")
    plt.fill_between(dates, lower, upper, color="lightblue", alpha=0.4, label="Доверительный интервал")
    plt.title("Прогноз продаж")
    plt.xlabel("Дата")
    plt.ylabel("Продажи")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())

def format_markdown_safe(text: str) -> str:
    """Безопасно форматирует Markdown текст."""
    if not text:
        return ""

    # Экранируем специальные символы
    text = text.replace("\\n", "\n")

    # Защита от HTML injection
    dangerous_chars = ['<script', '</script>', 'javascript:', 'onerror=', 'onclick=']
    for char in dangerous_chars:
        if char.lower() in text.lower():
            text = text.replace(char, "[REMOVED]")

    return text


def format_code_blocks(text: str) -> str:
    """Преобразует теги <code> в блокнотный вывод."""
    # Разделяем текст на части между тегами <code>
    import re

    parts = []
    regex = r'(<code>(.*?)</code>)'
    last_end = 0

    for match in re.finditer(regex, text, re.DOTALL):
        # Добавляем часть текста перед кодом
        if match.start() > last_end:
            before = text[last_end:match.start()]
            if before.strip():
                parts.append(before)

        # Обрабатываем код
        code_block = match.group(1)
        code_content = match.group(2).strip()

        # Показываем как блок кода
        parts.append(f"```python\n{code_content}\n```")
        last_end = match.end()

    # Последняя часть после последнего кода
    if last_end < len(text):
        after = text[last_end:]
        if after.strip():
            parts.append(after)

    return "\n".join(parts)