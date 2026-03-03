"""
Глобальное состояние агента.
Для прототипа — один датасет на всё приложение.
"""
import logging
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Глобальное состояние (одно на всё приложение)
_global_state: Dict[str, Any] = {
    "dataset": None,
    "dataset_info": {},
    "last_forecast": None,
    "preferences": {},
}


def get_global_state() -> Dict[str, Any]:
    """Получить глобальное состояние."""
    return _global_state


def set_current_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Сохранить датасет в глобальное состояние."""
    from utils import find_columns

    _global_state["dataset"] = df

    date_col, sales_col, store_col = find_columns(df)

    info = {
        "rows": len(df),
        "columns": list(df.columns),
        "date_column": date_col,
        "sales_column": sales_col,
        "store_column": store_col,
        "date_range": None,
        "stores_count": None,
    }

    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        info["date_range"] = {
            "start": str(df[date_col].min()),
            "end": str(df[date_col].max())
        }
    if store_col:
        info["stores_count"] = df[store_col].nunique()

    _global_state["dataset_info"] = info
    logger.info(f"💾 Dataset saved: {info['rows']} rows")
    return info


def get_current_dataset() -> Optional[pd.DataFrame]:
    """Получить текущий датасет."""
    return _global_state.get("dataset")


def clear_state():
    """Очистить глобальное состояние."""
    global _global_state
    _global_state = {
        "dataset": None,
        "dataset_info": {},
        "last_forecast": None,
        "preferences": {},
    }
    logger.info("🧹 Global state cleared")
