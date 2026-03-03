"""Утилиты для работы с данными и временными рядами."""
import numpy as np
import pandas as pd
from config.constants import COLUMN_KEYWORDS

def find_columns(df: pd.DataFrame):
    """
    Определяет столбцы даты, продаж и магазина в датафрейме.

    Args:
        df: DataFrame для анализа

    Returns:
        Кортеж (date_col, sales_col, store_col)
    """
    date_col = sales_col = store_col = None

    for col in df.columns:
        col_lower = col.lower()

        if any(w in col_lower for w in COLUMN_KEYWORDS["date"]):
            date_col = col
        elif any(w in col_lower for w in COLUMN_KEYWORDS["sales"]):
            sales_col = col
        elif any(w in col_lower for w in COLUMN_KEYWORDS["store"]):
            store_col = col

    # Fallback: если sales_col не найден, берём первый числовой столбец
    if not sales_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sales_col = numeric_cols[0]

    return date_col, sales_col, store_col

def make_daily_series(df: pd.DataFrame, date_col: str, sales_col: str) -> pd.Series:
    """
    Создаёт ежедневный временной ряд из данных с датами и продажами.

    Args:
        df: Исходный DataFrame
        date_col: Имя столбца с датами
        sales_col: Имя столбца с продажами

    Returns:
        Series с индексом дат и значениями продаж
    """
    daily = df.groupby(date_col)[sales_col].sum()
    return daily.astype(float)

def calc_mae_mape(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Вычисляет MAE и MAPE между истинными и предсказанными значениями.

    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения

    Returns:
        Словарь с метриками {'mae': float, 'mape': float}
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denominator = y_true.copy()
    denominator[denominator == 0] = np.nan
    mape = float(
        np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100
    ) if np.isnan(denominator).any() is False else 0

    return {"mae": mae, "mape": mape}

def safe_number(value) -> float:
    """Безопасное преобразование в float."""
    try:
        f = float(value)
        if np.isfinite(f):
            return f
    except (ValueError, TypeError):
        pass
    return 0.0