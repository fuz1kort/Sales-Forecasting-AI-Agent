"""Интерфейс загрузки данных."""

import io

import pandas as pd
import streamlit as st


def render_upload_section(api_client):
    """Рендер секции загрузки данных."""
    st.header("📁 Загрузка данных")

    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл с продажами",
        type=["csv"],
        help="Файл должен содержать колонки с датой и объёмом продаж"
    )

    preview_data = None

    if uploaded_file is not None:
        # Предпросмотр данных
        st.subheader("👀 Предпросмотр данных")
        try:
            df_preview = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), nrows=10)
            st.dataframe(df_preview, width='stretch')
            preview_data = uploaded_file.getvalue()
            preview_filename = uploaded_file.name
        except Exception as e:
            st.error(f"⚠️ Не удалось прочитать CSV: {e}")
            preview_data = None

        # Кнопка загрузки
        st.divider()

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🚀 Загрузить в агент", type="primary"):
                with st.spinner("⏳ Загружаю данные в агент..."):
                    result, session_id = api_client.upload_file(preview_data, preview_filename)

                    if result.get("status") == "success":
                        st.success(f"✅ Загружено {result.get('rows', 0):,} строк!")

                        # Сохраняем информацию о датасете
                        st.session_state.dataset_info = result

                        # Автоанализ после загрузки
                        with st.expander("📊 Данные загружены успешно!"):
                            st.write(f"**Диапазон дат:** {result.get('date_range', {}).get('start', '—')} — {result.get('date_range', {}).get('end', '—')}")
                            st.write(f"**Колонки:** {', '.join(result.get('columns', []))}")

                            if session_id:
                                st.info(f"**Session ID сохранён:** `{session_id}`")
                    else:
                        error_msg = result.get("error", "Неизвестная ошибка")
                        st.error(f"❌ Ошибка загрузки: {error_msg}")

    return preview_data if preview_data is not None else None