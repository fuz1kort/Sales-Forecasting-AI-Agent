# app/backend/agent/tools/data.py
"""
Инструменты для работы с данными.

Предоставляет функции загрузки, валидации и предобработки
датасетов для прогнозирования продаж.
"""

import io
import logging
from typing import Optional

import pandas as pd
from smolagents import tool

from utils import find_columns, validate_dataset
from config.constants import ERROR_MESSAGES

logger = logging.getLogger(__name__)


@tool
def load_dataset(
        csv_content: str,
        encoding: str = "utf-8"
) -> dict:
    """
    Загрузить и валидировать датасет продаж из CSV.

    Функция автоматически определяет кодировку, находит
    ключевые колонки (дата, продажи, магазин) и сохраняет
    данные в состоянии сессии.

    Args:
        csv_content: Содержимое CSV-файла как строка
        encoding: Кодировка файла (по умолчанию: utf-8)

    Returns:
        Словарь с результатом загрузки:
        - status: "success" или "error"
        - rows: количество строк
        - columns: список колонок
        - date_column: найденная колонка даты
        - sales_column: найденная колонка продаж
        - error: сообщение об ошибке (если есть)

    Example:
        result = load_dataset(
            csv_content="date,sales\\n2024-01-01,1000\\n...",
            session_id="user_123"
        )
    """
    from agent.state import set_current_dataset
    from utils import find_columns

    encodings_to_try = [encoding, 'utf-8-sig', 'cp1252', 'latin-1']

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(io.StringIO(csv_content), encoding=enc)
            logger.info(f"Dataset loaded with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Encoding {enc} failed: {e}")
            continue
    else:
        # Если все кодировки не подошли, пробуем с игнорированием ошибок
        df = pd.read_csv(io.StringIO(csv_content), encoding='utf-8', errors='ignore')

    # Валидация: проверяем, что датафрейм не пустой
    if df.empty:
        return {"error": ERROR_MESSAGES["EMPTY_DATASET"], "status": "error"}

    # Автопоиск ключевых колонок
    date_col, sales_col, store_col = find_columns(df)

    if not date_col or not sales_col:
        return {
            "error": ERROR_MESSAGES["INVALID_COLUMNS"],
            "found_columns": list(df.columns),
            "status": "error"
        }

    # Предобработка: приводим дату к datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        logger.warning(f"Could not parse date column: {e}")

    # Сохраняем в состояние сессии
    info = set_current_dataset(df)

    return {
        "status": "success",
        "rows": info["rows"],
        "columns": info["columns"],
        "date_column": date_col,
        "sales_column": sales_col,
        "store_column": store_col,
        "date_range": info.get("date_range"),
        "message": f"✅ Загружено {info['rows']} строк данных"
    }


@tool
def get_dataset_info() -> dict:
    """
    Получить метаданные текущего датасета.

    Returns:
        Словарь с информацией о датасете
    """
    from agent.state import get_global_state

    state = get_global_state()
    info = state.get("dataset_info", {})

    if not info:
        return {"status": "no_data", "message": "Датасет не загружен"}

    return {
        "status": "success",
        **info
    }
