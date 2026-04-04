"""Общий макет приложения."""
from typing import Optional

import streamlit as st
from datetime import datetime


def render_header():
    """Заголовок приложения."""
    st.title("📊 Sales Forecasting Agent")
    st.caption("Интеллектуальный агент для анализа продаж")


def render_footer():
    """Подвал страницы."""
    st.divider()
    footer_text = (
        "Sales Forecasting Agent v1.0 | "
        "Модели: Prophet, SARIMA, CatBoost | "
        f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    st.caption(footer_text)


def render_sidebar(session_id: Optional[str] = None):
    """Боковая панель с информацией."""
    with st.sidebar:
        st.header("ℹ️ О проекте")

        st.markdown("""
        ### Что умеет агент:
        - 📈 Прогнозировать продажи (ансамбль моделей)
        - 🔮 Анализировать тренды и сезонность
        - 📊 Визуализировать данные (корреляции, распределения, ABC-анализ)
        - 🧪 Проверять стационарность рядов (ADF-тест)
        - 🏆 Сравнивать модели и выбирать лучшую
        - 📦 Анализировать товары, клиентов и KPI
        - 🔄 Выполнять бэктестирование моделей
        """
                    )

        st.divider()

        if session_id:
            st.info(f"**Session ID**: `{session_id[:8]}...`")
            if st.session_state.get("session_restored"):
                st.success("✅ Сессия восстановлена после перезагрузки")