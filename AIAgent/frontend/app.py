"""Основное приложение Streamlit."""

import streamlit as st

from components.chat_ui import render_chat_section
from components.common import render_metrics_panel
from components.layout import render_header, render_footer, render_sidebar
from components.upload_ui import render_upload_section
# Импорт компонентов
from utils.api_client import APIClient

# TODO переработать UI
def main():
    """Запуск приложения."""
    st.set_page_config(
        page_title="📊 Sales Forecasting Agent",
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

        st.divider()

        # Всегда показываем чат после загрузки
        render_chat_section(api_client)

    render_footer()


if __name__ == "__main__":
    main()