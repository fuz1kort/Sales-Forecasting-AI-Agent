"""Интерфейс чата с агентом."""

import streamlit as st


def render_chat_section(api_client):
    """Рендер секции чата."""
    st.header("💬 Чат с аналитиком")

    # Отображение истории
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Начните диалог или задайте вопрос из списка ниже!")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    render_agent_response(msg["content"], msg.get("is_code", False))

    # Поле ввода
    prompt = st.chat_input("Спросите о продажах, прогнозе или данных...", key="chat_input")

    if prompt:
        # Добавляем сообщение пользователя
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Отображаем вопрос
        with st.chat_message("user"):
            st.markdown(prompt)

        # Отправляем запрос и ждём ответ
        with st.spinner("🤖 Агент анализирует данные..."):
            response = api_client.send_query(prompt)

            if response.get("status") == "success":
                answer = response.get("answer", "")

                # Отображаем ответ
                with st.chat_message("assistant"):
                    render_agent_response(answer)

                # Сохраняем в историю
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


def render_agent_response(text: str, is_code: bool = False):
    """Форматирует ответ агента для отображения."""
    if not text:
        return

    # Замена маркдаун тегов на безопасные
    safe_text = format_markdown_safe(text)

    # Если есть блок кода <code>...</code> — обрабатываем отдельно
    if is_code:
        safe_text = format_code_blocks(safe_text)

    st.markdown(safe_text)


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