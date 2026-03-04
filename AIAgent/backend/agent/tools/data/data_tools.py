"""Инструменты для работы с данными."""
import io
import logging
from typing import Optional
import pandas as pd
from smolagents import tool
from agent.state import get_session_manager

logger = logging.getLogger(__name__)

@tool
def load_dataset(
        csv_content: str,
        encoding: str = "utf-8",
        session_id: Optional[str] = None
) -> dict:
    """
    Загрузить и валидировать датасет продаж из CSV.
    
    Args:
        csv_content: Содержимое CSV-файла как строка
        encoding: Кодировка файла
        session_id: ID сессии (если передан)
    
    Returns:
        Словарь с результатом загрузки
    """
    encodings_to_try = [encoding, 'utf-8-sig', 'cp1252', 'latin-1']
    df = None

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

    if df is None:
        df = pd.read_csv(io.StringIO(csv_content), encoding='utf-8', errors='ignore')

    if df.empty:
        return {"error": "Пустой датасет", "status": "error"}

    from utils import find_columns
    date_col, sales_col, store_col = find_columns(df)

    if not date_col or not sales_col:
        return {
            "error": "Не найдены обязательные колонки. Нужны: date + sales",
            "found_columns": list(df.columns),
            "status": "error"
        }

    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        logger.warning(f"Could not parse date column: {e}")

    session_manager = get_session_manager()
    info = session_manager.save_dataset(session_id or "default", df)

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
def get_dataset_info(session_id: Optional[str] = None) -> dict:
    """Получить метаданные текущего датасета.

    Args:
        session_id: ID сессии для получения информации (по умолчанию "default").
    """
    session_manager = get_session_manager()
    dataset = session_manager.get_dataset(session_id or "default")

    if dataset is None:
        return {"status": "no_data", "message": "Датасет не загружен"}

    return {
        "status": "success",
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "data_type": type(dataset).__name__,
    }