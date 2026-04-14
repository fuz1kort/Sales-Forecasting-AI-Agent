"""Основное приложение Streamlit."""

import streamlit as st

from components.chat_ui import render_chat_section
from components.common import render_metrics_panel
from components.layout import render_header, render_footer, render_sidebar
from components.upload_ui import render_upload_section
# Импорт компонентов
from utils.api_client import APIClient


def _dedupe_history(history: list[dict]) -> list[dict]:
    """
    Удаляет дубликаты из истории чата.

    Дубликаты возникают когда подряд приходят сообщения от одного того же роля
    с идентичным содержимым.

    Args:
        history: Исходная история сообщений

    Returns:
        Очищенная история без дублей
    """
    cleaned = []
    for item in history:
        if not cleaned or cleaned[-1].get("role") != item.get("role") or cleaned[-1].get("content") != item.get("content"):
            cleaned.append(item)
    return cleaned


def main():
    """Запуск приложения Streamlit для взаимодействия с агентом."""
    st.set_page_config(
        page_title="Sales Forecasting Agent",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Инициализация session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = None
    if "last_forecast" not in st.session_state:
        st.session_state.last_forecast = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # Создаём API клиент (сессия хранится внутри клиента)
    api_client = APIClient()

    if st.session_state.session_id:
        api_client.session_id = st.session_state.session_id

        # Попробуем восстановить данные из существующей сессии
        if not st.session_state.dataset_info:
            session_info = api_client.get_session_info()
            if session_info.get("status") == "success":
                st.session_state.dataset_info = session_info

        # Восстанавливаем историю чата, если она есть
        if not st.session_state.chat_history:
            history_result = api_client.get_session_history()
            if history_result.get("status") == "success" and isinstance(history_result.get("history"), list):
                st.session_state.chat_history = _dedupe_history(history_result.get("history"))

    # Рендерим основной интерфейс
    render_header()
    render_sidebar(st.session_state.session_id)

    st.divider()

    # Секция загрузки
    render_upload_section(api_client)

    # Если есть данные — показываем остальное
    if st.session_state.dataset_info:
        render_metrics_panel(st.session_state.dataset_info)
        st.divider()

        # Всегда показываем чат после загрузки
        render_chat_section(api_client)

    render_footer()


if __name__ == "__main__":
    main()