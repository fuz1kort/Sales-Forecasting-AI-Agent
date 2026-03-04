"""Инструменты для загрузки и управления датасетами."""
import io
import logging
from typing import Optional

import pandas as pd
from smolagents import tool

from agent.state import get_session_manager
from utils import find_columns

logger = logging.getLogger(__name__)


@tool
def load_dataset(
        csv_content: str,
        encoding: str = "utf-8",
        session_id: Optional[str] = None
) -> dict:
    """
    Загрузить датасет продаж из CSV-строки.

    Вызывайте, когда пользователь загрузил файл или вставил CSV-данные.

    Args:
        csv_content: Содержимое CSV-файла как строка
        encoding: Кодировка файла (по умолчанию utf-8)
        session_id: ID сессии пользователя (опционально)

    Returns:
        dict с результатом:
        - status: "success" | "error"
        - rows: количество строк
        - columns: список колонок
        - date_column: найденная колонка даты
        - sales_column: найденная колонка продаж
        - error: сообщение об ошибке (если есть)
    """
    # Пробуем разные кодировки для надёжности
    encodings_to_try = [encoding, 'utf-8-sig', 'cp1252', 'latin-1']
    df = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(io.StringIO(csv_content), encoding=enc)
            logger.info(f"✓ Dataset loaded with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.debug(f"Encoding {enc} failed: {e}")
            continue

    # Фоллбэк: игнорировать ошибки кодировки
    if df is None:
        df = pd.read_csv(io.StringIO(csv_content), encoding='utf-8', errors='ignore')

    if df is None or df.empty:
        return {"status": "error", "error": "Пустой или невалидный датасет"}

    # Авто-поиск обязательных колонок
    date_col, sales_col, store_col = find_columns(df)

    if not date_col or not sales_col:
        return {
            "status": "error",
            "error": "Не найдены обязательные колонки. Нужны: дата + продажи",
            "found_columns": list(df.columns),
            "suggestion": "Переименуйте колонки в 'date'/'data' и 'sales'/'revenue'/'выручка'"
        }

    # Парсинг даты
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        logger.warning(f"⚠️ Could not parse date column '{date_col}': {e}")
        return {
            "status": "error",
            "error": f"Не удалось распарсить дату в колонке '{date_col}'"
        }

    # Сохраняем в сессию
    session_manager = get_session_manager()
    sid = session_id or "default"
    info = session_manager.save_dataset(sid, df)

    return {
        "status": "success",
        "rows": info["rows"],
        "columns": list(df.columns),
        "date_column": date_col,
        "sales_column": sales_col,
        "store_column": store_col,
        "date_range": info.get("date_range"),
        "message": f"✅ Загружено {info['rows']:,} строк за период {info.get('date_range', 'N/A')}"
    }


@tool
def get_dataset_info(session_id: Optional[str] = None) -> dict:
    """
    Получить метаданные текущего датасета.

    Вызывайте, чтобы проверить, какие данные загружены,
    перед запуском анализа или прогноза.

    Args:
        session_id: ID сессии (по умолчанию "default")

    Returns:
        dict с информацией о датасете или статусом ошибки
    """
    session_manager = get_session_manager()
    sid = session_id or "default"
    df = session_manager.get_dataset(sid)

    if df is None:
        return {
            "status": "no_data",
            "message": "Датасет не загружен. Сначала вызовите load_dataset."
        }

    date_col, sales_col, store_col = find_columns(df)

    return {
        "status": "success",
        "rows": len(df),
        "columns": list(df.columns),
        "date_column": date_col,
        "sales_column": sales_col,
        "store_column": store_col,
        "date_range": f"{df[date_col].min().date()} — {df[date_col].max().date()}" if date_col else None,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }