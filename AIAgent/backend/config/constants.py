"""
Константы приложения.
"""

# Ключевые слова для автоматического поиска колонок
COLUMN_KEYWORDS = {
    "date": ["date", "time", "day", "month", "year", "дата"],
    "sales": ["sales", "revenue", "amount", "value", "price", "total", "сумма", "объем", "выручка"],
    "store": ["store", "store_id", "storeid", "магазин", "id"],
    "product": ["product", "item", "товар", "продукт", "sku", "name", "название"],
}

# Параметры прогнозирования по умолчанию
DEFAULT_FORECAST_PERIODS = 30
MAX_BACKTEST_DAYS = 365
MIN_DATA_POINTS = 10

# Пределы для входных данных
MAX_CSV_SIZE_MB = 100
MAX_FORECAST_PERIODS = 365

# Параметры моделей
NEURALPROPHET_PARAMS = {
    "n_lags": 10,
    "n_forecasts": 1,
    "yearly_seasonality": True,
    "weekly_seasonality": True,
}

SARIMA_PARAMS = {
    "order": (1, 0, 0),
    "seasonal_order": (1, 1, 1, 7),
}

# Сообщения об ошибках
ERROR_MESSAGES = {
    "DATASET_NOT_LOADED": "Датасет не загружен. Сначала загрузите данные через load_dataset.",
    "EMPTY_DATASET": "Пустой датасет",
    "INVALID_COLUMNS": "Не найдены обязательные колонки. Нужны колонки с датой и объёмом продаж.",
    "INSUFFICIENT_DATA": "Недостаточно данных для прогноза (минимум ~10 дат).",
    "STORE_NOT_FOUND": "Для указанных ID магазинов нет данных.",
    "INVALID_DATE_COLUMN": "Не удалось парсить колонку дат.",
}

# Форматы вывода
DATE_FORMAT = "%Y-%m-%d"
FLOAT_FORMAT = "{:,.2f}"

