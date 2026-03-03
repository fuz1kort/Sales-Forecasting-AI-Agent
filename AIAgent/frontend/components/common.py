"""Общие компоненты и утилиты."""

import streamlit as st
import plotly.express as px
import pandas as pd
import io


def render_metrics_panel(info: dict):
    """Отображение метрик датасета."""
    cols = st.columns(3)

    cols[0].metric("📈 Записей", f"{info.get('rows', 0):,}")

    date_range = info.get('date_range')
    if isinstance(date_range, dict):
        date_str = f"{date_range.get('start', '—')} — {date_range.get('end', '—')}"
    else:
        date_str = str(date_range) if date_range else '—'

    cols[1].metric("📅 Период", date_str)

    if info.get('stores_count'):
        cols[2].metric("🏪 Магазинов", info['stores_count'])


def render_forecast_chart(forecast_list: list, title: str = "Прогноз продаж"):
    """Отображение графика прогноза."""
    try:
        forecast_df = pd.DataFrame(forecast_list)

        if forecast_df.empty:
            st.warning("⚠️ Нет данных для визуализации прогноза.")
            return

        # Проверка на прогноз по магазинам
        has_store = "store_id" in forecast_df.columns

        if has_store:
            stores = forecast_df["store_id"].unique()
            num_tabs = min(len(stores), 5)  # Максимум 5 табов

            store_tabs = st.tabs([f"🏪 {s}" for s in stores[:num_tabs]])

            for tab, store in zip(store_tabs, stores):
                with tab:
                    sub_df = forecast_df[forecast_df["store_id"] == store]
                    fig = px.line(
                        sub_df, x="date", y="forecast",
                        title=f"Прогноз для магазина {store}",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Дата", yaxis_title="Продажи")
                    st.plotly_chart(fig, use_container_width=True)

            # Итоговая статистика
            total_sum = forecast_df["forecast"].sum()
            avg_daily = forecast_df["forecast"].mean()
            period_days = len(forecast_df)

            st.write("### 📊 Итоговая статистика")
            c1, c2, c3 = st.columns(3)
            c1.metric("📈 Сумма прогноза", f"${total_sum:,.0f}")
            c2.metric("📅 Среднее в день", f"${avg_daily:,.0f}")
            c3.metric("📆 Дней", period_days)
        else:
            fig = px.line(
                forecast_df, x="date", y="forecast",
                title=title,
                markers=True
            )

            # Доверительные интервалы
            has_lower = "lower_bound" in forecast_df.columns
            has_upper = "upper_bound" in forecast_df.columns

            if has_lower and has_upper:
                fig.add_traces([
                    px.line(forecast_df, x="date", y="lower_bound").data[0],
                    px.line(forecast_df, x="date", y="upper_bound").data[0],
                ])
                fig.data[1].name = "Нижняя граница"
                fig.data[2].name = "Верхняя граница"
                fig.update_traces(opacity=0.3, selector=dict(name="Нижняя граница"))
                fig.update_traces(opacity=0.3, selector=dict(name="Верхняя граница"))

            fig.update_layout(xaxis_title="Дата", yaxis_title="Продажи")
            st.plotly_chart(fig, use_container_width=True)

            # Статистика
            total_sum = forecast_df["forecast"].sum()
            avg_daily = forecast_df["forecast"].mean()
            period_days = len(forecast_df)

            st.write("### 📊 Итоговая статистика")
            c1, c2, c3 = st.columns(3)
            c1.metric("📈 Сумма прогноза", f"${total_sum:,.0f}")
            c2.metric("📅 Среднее в день", f"${avg_daily:,.0f}")
            c3.metric("📆 Дней", period_days)

        # Скачивание CSV
        csv_buf = io.StringIO()
        forecast_df.to_csv(csv_buf, index=False)
        download_btn = f"📥 Скачать прогноз (CSV)"
        st.download_button(
            label=download_btn,
            data=csv_buf.getvalue(),
            file_name=f"forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"❌ Ошибка отображения графика: {e}")