"""Утилиты для работы с данными и временными рядами."""
import numpy as np
import pandas as pd
from backend.config.constants import COLUMN_KEYWORDS

def find_columns(df: pd.DataFrame):
    """
    Определяет столбцы даты, продаж, магазина и товара в датафрейме.

    Args:
        df: DataFrame для анализа

    Returns:
        Кортеж (date_col, sales_col, store_col, product_col)
    """
    date_col = sales_col = store_col = product_col = None

    for col in df.columns:
        col_lower = col.lower()

        if any(w in col_lower for w in COLUMN_KEYWORDS["date"]):
            date_col = col
        elif any(w in col_lower for w in COLUMN_KEYWORDS["sales"]):
            sales_col = col
        elif any(w in col_lower for w in COLUMN_KEYWORDS["store"]):
            store_col = col
        elif any(w in col_lower for w in COLUMN_KEYWORDS["product"]):
            product_col = col

    # Fallback: если sales_col не найден, берём первый числовой столбец
    if not sales_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sales_col = numeric_cols[0]

    return date_col, sales_col, store_col, product_col

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

def aggregate_transactions(df: pd.DataFrame, date_col: str, price_col: str = None, 
                          quantity_col: str = None, sales_col: str = None) -> pd.DataFrame:
    """
    Агрегирует данные транзакций в дневные продажи.
    
    Если есть отдельные колонки Price и Quantity, вычисляет выручку.
    Иначе использует готовую колонку sales_col.
    
    Args:
        df: DataFrame с транзакциями
        date_col: Имя столбца с датами
        price_col: Имя столбца цены (опционально)
        quantity_col: Имя столбца количества (опционально)
        sales_col: Имя готовой колонки продаж (опционально)
    
    Returns:
        DataFrame с дневными агрегированными продажами
    """
    df = df.copy()
    
    # Конвертируем дату
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    date_only = df[date_col].dt.date
    
    # Вычисляем выручку если нужно
    if price_col and quantity_col and price_col in df.columns and quantity_col in df.columns:
        df['Revenue'] = df[price_col] * df[quantity_col]
        revenue_col = 'Revenue'
    elif sales_col and sales_col in df.columns:
        revenue_col = sales_col
    else:
        # Fallback: ищем числовую колонку
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            revenue_col = numeric_cols[-1]
        else:
            raise ValueError("Не найдена колонка продаж для агрегирования")
    
    # Агрегируем по датам
    daily_df = df.groupby(date_only).agg({
        revenue_col: 'sum',
        quantity_col if quantity_col else revenue_col: 'count'
    }).rename(columns={quantity_col if quantity_col else revenue_col: 'Transactions'})
    
    daily_df.index.name = 'Date'
    daily_df = daily_df.reset_index()
    daily_df.columns = ['Date', 'Revenue', 'Transactions']
    
    return daily_df

def detect_transaction_data(df: pd.DataFrame) -> bool:
    """
    Определяет, содержит ли датафрейм отдельные транзакции (много строк на дату)
    или уже это агрегированные данные.
    
    Args:
        df: DataFrame для проверки
    
    Returns:
        True если данные это отдельные транзакции
    """
    date_col, _, _, _ = find_columns(df)
    if not date_col or date_col not in df.columns:
        return False
    
    # Считаем среднее количество транзакций на дату
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    
    transaction_counts = df_temp.groupby(df_temp[date_col].dt.date).size()
    avg_transactions = transaction_counts.mean()
    
    # Если в среднем > 10 транзакций на дату, то это транзакционные данные
    return avg_transactions > 10

def get_data_structure_info(df: pd.DataFrame) -> dict:
    """
    Анализирует структуру датасета и возвращает информацию для правильной обработки.
    
    Args:
        df: DataFrame для анализа
    
    Returns:
        dict с информацией о структуре:
        - has_transactions: есть ли отдельные транзакции
        - location_column: какая колонка используется для location (store/country/region)
        - product_column: какая колонка используется для product
        - price_quantity: есть ли отдельные price и quantity колонки
        - structure_type: "standard" | "custom"
    """
    date_col, sales_col, store_col, product_col = find_columns(df)
    
    # Проверяем наличие отдельных Price и Quantity колонок
    has_price_qty = ("Price" in df.columns or "price" in df.columns.str.lower()) and \
                    ("Quantity" in df.columns or "quantity" in df.columns.str.lower())
    
    # Определяем type location (store, country, region, etc)
    location_type = "unknown"
    if store_col:
        col_lower = store_col.lower()
        if "country" in col_lower:
            location_type = "country"
        elif "store" in col_lower:
            location_type = "store"
        elif "customer" in col_lower:
            location_type = "customer"
        elif "region" in col_lower:
            location_type = "region"
        else:
            location_type = "location"
    
    return {
        "has_transactions": detect_transaction_data(df),
        "location_column": store_col,
        "location_type": location_type,  # ⭐ важный для filters
        "product_column": product_col,
        "price_quantity": has_price_qty,
        "structure_type": "standard" if has_price_qty else "custom",
        "date_column": date_col,
        "sales_column": sales_col,
    }


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения

    Returns:
        sMAPE в процентах
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # Защита от деления на ноль
    return np.mean(diff) * 100


def adf_test(series, title=''):
    """
    Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series data
        title: Title for the test

    Returns:
        Dict with test results
    """
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series, autolag='AIC')
    output = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4],
        'Conclusion': 'Stationary' if result[1] < 0.05 else 'Non-Stationary'
    }
    return output

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