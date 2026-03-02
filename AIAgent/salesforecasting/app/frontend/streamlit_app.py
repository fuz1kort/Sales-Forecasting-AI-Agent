"""
Frontend для Sales Forecasting Agent на базе Streamlit.

Упрощённая версия без управления сессиями — для локального тестирования.
"""

import io
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="📊 Sales Forecasting Agent",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ STATE
# ============================================================================

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_forecast" not in st.session_state:
    st.session_state.last_forecast = None
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "last_backtest" not in st.session_state:
    st.session_state.last_backtest = None
if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def upload_file_to_agent(file) -> dict | None:
    """
    Загрузить CSV-файл в агента через эндпоинт /upload.
    """
    try:
        files = {"file": (file.name, file.getvalue(), "text/csv")}

        response = requests.post(
            f"{API_BASE}/upload",
            files=files,
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Ошибка загрузки: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Неожиданная ошибка: {e}")
        return None


def send_chat_query(query: str) -> dict | None:
    """
    Отправить запрос в чат-агент.
    """
    try:
        data = {"query": query}

        response = requests.post(
            f"{API_BASE}/chat",
            data=data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        st.warning("⏱️ Таймаут ответа от агента. Попробуйте упростить запрос.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Ошибка связи с API: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Неожиданная ошибка: {e}")
        return None


def build_forecast_request(file, params: dict) -> dict | None:
    """
    Построить прогноз через API.
    """
    try:
        files = {"file": (file.name, file.getvalue(), "text/csv")}

        response = requests.post(
            f"{API_BASE}/forecast",
            files=files,
            data=params,
            timeout=180
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        st.warning("⏱️ Таймаут при построении прогноза. Попробуйте уменьшить горизонт.")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка прогноза: {e}")
        return None


def run_backtest(file, test_days: int) -> dict | None:
    """
    Запустить backtest моделей.
    """
    try:
        files = {"file": (file.name, file.getvalue(), "text/csv")}
        data = {"test_days": test_days}

        response = requests.post(
            f"{API_BASE}/backtest",
            files=files,
            data=data,
            timeout=180
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        st.error(f"❌ Ошибка backtest: {e}")
        return None


def format_markdown(text: str) -> str:
    """Безопасно форматировать markdown-ответ от агента."""
    if not text:
        return ""
    return text


# ============================================================================
# UI: ЗАГОЛОВОК
# ============================================================================

st.title("📊 Sales Forecasting Agent")
st.caption("Интеллектуальный агент для прогнозирования и анализа продаж на базе LLM")

# Быстрая статистика (если данные загружены)
if st.session_state.dataset_info:
    info = st.session_state.dataset_info
    cols = st.columns(3)

    cols[0].metric("📈 Записей", f"{info.get('rows', 0):,}")

    # 🔧 Форматируем date_range из dict в строку
    date_range = info.get('date_range')
    if isinstance(date_range, dict):
        date_range_str = f"{date_range.get('start', '—')} — {date_range.get('end', '—')}"
    elif isinstance(date_range, str):
        date_range_str = date_range
    else:
        date_range_str = '—'

    cols[1].metric("📅 Период", date_range_str)

    if info.get('stores_count'):
        cols[2].metric("🏪 Магазинов", info['stores_count'])

st.divider()

# ============================================================================
# UI: ЗАГРУЗКА ДАННЫХ
# ============================================================================

st.header("📁 1. Загрузка данных")

uploaded_file = st.file_uploader(
    "Загрузите CSV-файл с продажами",
    type=["csv"],
    help="Файл должен содержать колонки с датой и объёмом продаж"
)

if uploaded_file is not None:
    # Показываем превью данных
    with st.expander("👀 Предпросмотр данных", expanded=False):
        try:
            df_preview = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), nrows=5)
            st.dataframe(df_preview, width='stretch')
        except Exception as e:
            st.error(f"Не удалось прочитать CSV: {e}")

    # Кнопка загрузки в агента
    if st.button("🚀 Загрузить в агента", type="primary"):
        with st.spinner("Загружаю данные в агента..."):
            result = upload_file_to_agent(uploaded_file)

            if result and result.get("status") == "success":
                st.success(f"✅ Загружено {result.get('rows', 0)} строк")
                st.session_state.dataset_info = result
                st.session_state.uploaded_file = uploaded_file

                # Авто-анализ после загрузки
                with st.spinner("Анализирую данные..."):
                    chat_result = send_chat_query("Сделай краткий обзор моих данных о продажах")
                    if chat_result and chat_result.get("status") == "success":
                        st.session_state.last_analysis = chat_result["answer"]
                        st.rerun()
            else:
                error_msg = result.get("error", "Неизвестная ошибка") if result else "Нет ответа от сервера"
                st.error(f"❌ Ошибка: {error_msg}")

st.divider()

# ============================================================================
# UI: ПРОГНОЗ ПРОДАЖ
# ============================================================================

st.header("🔮 2. Прогноз продаж")

if st.session_state.dataset_info is None:
    st.info("👆 Сначала загрузите данные в разделе выше")
else:
    with st.expander("⚙️ Параметры прогноза", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            forecast_type = st.radio(
                "Тип прогноза",
                ["general", "by_store"],
                format_func=lambda x: "🌐 Общий" if x == "general" else "🏪 По магазинам",
                horizontal=True,
            )

            if forecast_type == "by_store":
                store_ids = st.text_input(
                    "ID магазинов (через запятую)",
                    placeholder="1, 2, 3",
                    help="Оставьте пустым для всех магазинов"
                )
            else:
                store_ids = None

        with col2:
            model_type = st.radio(
                "Модель",
                ["neuralprophet", "arima"],
                format_func=lambda x: "🧠 NeuralProphet" if x == "neuralprophet" else "📐 ARIMA",
                horizontal=True,
            )

            horizon_mode = st.radio(
                "Горизонт",
                ["days", "dates"],
                format_func=lambda x: "📅 В днях" if x == "days" else "🗓️ По датам",
                horizontal=True,
            )

    # Параметры горизонта
    periods = start_date = end_date = None
    if horizon_mode == "days":
        periods = st.number_input("Дней вперёд", min_value=7, max_value=365, value=30)
    else:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Начало", value=datetime.now() + timedelta(days=1))
        with col_d2:
            end_date = st.date_input("Конец", value=datetime.now() + timedelta(days=30))

    # Кнопка построения
    if st.button("🎯 Построить прогноз", type="primary", width='stretch'):
        if st.session_state.uploaded_file is None:
            st.error("❌ Файл не загружен. Загрузите данные сначала.")
        else:
            params = {
                "forecast_type": forecast_type,
                "model_type": model_type,
            }
            if periods:
                params["periods"] = periods
            if start_date:
                params["start_date"] = start_date.strftime("%Y-%m-%d")
            if end_date:
                params["end_date"] = end_date.strftime("%Y-%m-%d")
            if store_ids and store_ids.strip():
                params["store_ids"] = store_ids.strip()

            with st.spinner("🔮 Строю прогноз..."):
                result = build_forecast_request(st.session_state.uploaded_file, params)

                if result and "forecast" in result:
                    st.success("✅ Прогноз построен!")
                    st.session_state.last_forecast = result["forecast"]
                    st.rerun()
                elif result and "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.error("❌ Не удалось получить прогноз")

# Отображение прогноза
if st.session_state.last_forecast:
    st.subheader("📊 Результат прогноза")

    try:
        forecast_df = pd.DataFrame(st.session_state.last_forecast)

        # Визуализация
        if "store_id" in forecast_df.columns:
            # Прогноз по магазинам
            stores = forecast_df["store_id"].unique()
            store_tabs = st.tabs([f"🏪 {s}" for s in stores])

            for tab, store in zip(store_tabs, stores):
                with tab:
                    sub_df = forecast_df[forecast_df["store_id"] == store]

                    fig = px.line(
                        sub_df, x="date", y="forecast",
                        title=f"Прогноз для магазина {store}",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Дата", yaxis_title="Продажи")
                    st.plotly_chart(fig, width='stretch')

                    # Статистика
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("Сумма прогноза", f"${sub_df['forecast'].sum():,.0f}")
                    col_s2.metric("Среднее в день", f"${sub_df['forecast'].mean():,.0f}")
                    col_s3.metric("Период", f"{len(sub_df)} дней")
        else:
            # Общий прогноз
            fig = px.line(
                forecast_df, x="date", y="forecast",
                title="📈 Прогноз продаж",
                markers=True
            )

            # Доверительные интервалы если есть
            if all(col in forecast_df.columns for col in ["lower_bound", "upper_bound"]):
                fig.add_traces([
                    px.line(forecast_df, x="date", y="lower_bound").data[0],
                    px.line(forecast_df, x="date", y="upper_bound").data[0],
                ])
                fig.data[1].name = "Нижняя граница"
                fig.data[2].name = "Верхняя граница"
                fig.update_traces(opacity=0.3, selector=dict(name="Нижняя граница"))
                fig.update_traces(opacity=0.3, selector=dict(name="Верхняя граница"))

            fig.update_layout(xaxis_title="Дата", yaxis_title="Продажи")
            st.plotly_chart(fig, width='stretch')

            # Статистика
            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric("Сумма прогноза", f"${forecast_df['forecast'].sum():,.0f}")
            col_t2.metric("Среднее в день", f"${forecast_df['forecast'].mean():,.0f}")
            col_t3.metric("Период", f"{len(forecast_df)} дней")

        # Экспорт в CSV
        csv_buf = io.StringIO()
        forecast_df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Скачать прогноз (CSV)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Ошибка отображения: {e}")

st.divider()

# ============================================================================
# UI: BACKTEST МОДЕЛЕЙ
# ============================================================================

st.header("🧪 3. Сравнение моделей (Backtest)")

if st.session_state.dataset_info is None:
    st.info("👆 Сначала загрузите данные")
else:
    test_days = st.slider(
        "Дней на тест",
        min_value=7, max_value=90, value=30,
        help="Последние N дней будут использованы для проверки точности моделей"
    )

    if st.button("🔬 Запустить backtest"):
        with st.spinner("🤖 Агент сравнивает модели..."):
            # 🔧 Отправляем запрос через чат — агент сам вызовет run_backtest_tool
            query = f"Запусти сравнение моделей (backtest) на последние {test_days} дней. Покажи какая модель лучше."
            response = send_chat_query(query)

            if response and response.get("status") == "success":
                answer = response["answer"]
                st.success("✅ Backtest завершён!")

                # Показываем ответ агента
                with st.expander("📋 Ответ агента", expanded=True):
                    st.markdown(format_markdown(answer))

                # 🔧 Если в ответе есть predictions — сохраняем для графика
                # (агент может вернуть структурированные данные в answer)
                st.session_state.last_backtest = response
                st.rerun()
            else:
                error = response.get("error", "Неизвестная ошибка") if response else "Нет связи с сервером"
                st.error(f"❌ {error}")

# Отображение результатов backtest (если агент вернул структурированные данные)
if st.session_state.last_backtest and st.session_state.last_backtest.get("status") == "success":
    bt = st.session_state.last_backtest

    # 🔧 Пробуем извлечь predictions, если агент вернул их в answer или отдельно
    # Вариант 1: агент вернул dict с predictions внутри answer
    # Вариант 2: ты можешь попросить агента вернуть JSON в конце ответа

    # Пока показываем текстовый ответ агента
    if "answer" in bt and bt["answer"]:
        with st.expander("📊 Детали backtest", expanded=False):
            st.markdown(format_markdown(bt["answer"]))

    # Если в будущем агент будет возвращать predictions как dict, раскомментируй:
    # if "predictions" in bt and bt["predictions"].get("date"):
    #     preds = bt["predictions"]
    #     n = min(len(preds["date"]), len(preds["actual"]))
    #     plot_df = pd.DataFrame({
    #         "date": preds["date"][:n],
    #         "Факт": preds["actual"][:n],
    #         "NeuralProphet": preds.get("neuralprophet", [None]*n)[:n],
    #         "SARIMA": preds.get("sarima", [None]*n)[:n],
    #     })
    #     plot_melt = plot_df.melt(id_vars=["date"], var_name="Модель", value_name="Значение")
    #     fig_bt = px.line(plot_melt, x="date", y="Значение", color="Модель", markers=True)
    #     st.plotly_chart(fig_bt, width='stretch')

st.divider()

# ============================================================================
# UI: ЧАТ С АГЕНТОМ
# ============================================================================

st.header("💬 4. Чат с аналитиком")

# Отображение истории чата
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(format_markdown(msg["content"]))

# Поле ввода
if prompt := st.chat_input("Спросите о продажах, прогнозе или данных..."):
    # Сообщение пользователя
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ответ агента
    with st.chat_message("assistant"):
        with st.spinner("🤖 Думаю..."):
            response = send_chat_query(prompt)

            if response and response.get("status") == "success":
                answer = response["answer"]
                st.markdown(format_markdown(answer))
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_answer = answer
            else:
                error = response.get("error", "Неизвестная ошибка") if response else "Нет связи с сервером"
                st.error(f"❌ {error}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Не удалось обработать запрос: {error}"
                })

# Кнопки быстрого доступа
with st.expander("⚡ Быстрые вопросы", expanded=False):
    quick_questions = [
        "Какие товары продаются лучше всего?",
        "Покажи тренд продаж по месяцам",
        "Спрогнозируй продажи на следующий месяц",
        "Какая общая выручка?",
        "Есть ли сезонность в данных?",
        "Покажи ключевые метрики (KPI)",
    ]
    for q in quick_questions:
        if st.button(q, width='stretch'):
            # Прямая отправка запроса без хаков с chat_input
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.chat_message("assistant"):
                with st.spinner("🤖 Думаю..."):
                    response = send_chat_query(q)
                    if response and response.get("status") == "success":
                        answer = response["answer"]
                        st.markdown(format_markdown(answer))
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    else:
                        error = response.get("error", "Неизвестная ошибка") if response else "Нет связи с сервером"
                        st.error(f"❌ {error}")

# ============================================================================
# UI: FOOTER
# ============================================================================

st.divider()
st.caption(
    "Sales Forecasting Agent v3.0 | "
    "Powered by smolagents + NeuralProphet"
)