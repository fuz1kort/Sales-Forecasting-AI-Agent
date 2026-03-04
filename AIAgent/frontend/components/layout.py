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
        "Sales Forecasting Agent v3.0 | "
        "Powered by smolagents + NeuralProphet | "
        f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    st.caption(footer_text)


def render_sidebar(session_id: Optional[str] = None):
    """Боковая панель с информацией."""
    with st.sidebar:
        st.header("ℹ️ О проекте")

        st.markdown("""
        ### Что умеет агент:
        • 📈 Прогнозировать продажи  
        • 🔮 Анализировать тренды  
        • 🏆 Выбирать лучшую модель  
        • 📦 Анализ товаров и клиентов  
        
        ### Режим работы:
        """
                    )

        mode = st.radio(
            "Выберите режим LLM:",
            ["CodeAgent (умнее)", "ToolCallingAgent (быстрее)"],
            help="CodeAgent может выполнять Python код, ToolCallingAgent только вызывает инструменты"
        )

        st.divider()

        if session_id:
            st.info(f"**Session ID**: `{session_id[:8]}...`")