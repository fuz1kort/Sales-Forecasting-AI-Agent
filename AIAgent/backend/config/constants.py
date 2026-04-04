"""
Константы приложения.
"""

# ===== АВТОМАТИЧЕСКИЙ ПОИСК КОЛОНОК =====
# Система автоматически ищет критически важные колонки по ключевым словам
# Приоритет поиска: первая найденная колонка в списке используется
COLUMN_KEYWORDS = {
    "date": ["date", "time", "day", "month", "year", "дата"],
    # Выручка/продажи - может быть готовая сумма или quantity*price
    "sales": ["sales", "revenue", "amount", "value", "price", "total", "quantity", "сумма", "объем", "выручка"],
    # Локация/магазин - может быть store_id, customer, region, country, location
    "store": ["store", "store_id", "storeid", "магазин", "location", "регион", "id", "customer", "customerno", "country", "страна"],
    # Товар - может быть product_id или description/name
    "product": ["product", "item", "товар", "продукт", "sku", "name", "название", "productno", "productname", "description", "описание", "desc"],
}

# ===== ОЖИДАЕМЫЕ КОЛОНКИ ДЛЯ ТРАНЗАКЦИЙ =====
# ПРИМЕЧАНИЕ: Структура может отличаться в зависимости от источника данных
# 
# Стандартная структура retail-датасета:
TRANSACTION_COLUMNS = {
    "date_column": "Date",                    # Дата транзакции
    "transaction_no": "TransactionNo",        # Номер/ID транзакции
    "customer_no": "CustomerNo",              # ID клиента/магазина
    "product_no": "ProductNo",                # ID товара
    "product_name": "ProductName",            # Название товара
    "quantity":  "Quantity",                  # Количество (отрицательное = отмена)
    "price": "Price",                         # Цена за единицу
    "customer_location": "Country",           # Страна/локация покупателя ⭐ (может быть вместо store_id)
}

# РАСШИРЕННАЯ ИНФОРМАЦИЯ:
# Когда магазина нет в данных, используется другой location фактор:
#   - Country (страна) - география продаж
#   - Region (регион) - регион
#   - City (город) - населённый пункт
# В анализах country/region учитываются как location factor в фильтрах и группировках

# ===== ПРАВИЛА ОБРАБОТКИ ТРАНЗАКЦИЙ =====
TRANSACTION_RULES = {
    # Как определить отмену
    "cancellation_indicators": [
        "quantity_negative",      # Quantity < 0
        "transaction_prefix_c",   # TransactionNo начинается с 'C'
    ],
    
    # Как вычислить выручку
    "revenue_calculation": "Price * Quantity",  # Может быть отрицательное для отмен
    
    # Группировка для прогноза
    "aggregation_methods": {
        "daily": "Date (без времени) -> sum(Price * Quantity) per day",
        "by_product": "ProductNo -> sum(Price * Quantity) per date",
        "by_customer": "CustomerNo -> sum(Price * Quantity) per date",
    },
}

# ===== ПАРАМЕТРЫ ПРОГНОЗИРОВАНИЯ =====
DEFAULT_FORECAST_PERIODS = 30
MAX_BACKTEST_DAYS = 365
MIN_DATA_POINTS = 10

# ===== ПРЕДЕЛЫ ДЛЯ ВХОДНЫХ ДАННЫХ =====
MAX_CSV_SIZE_MB = 100
MAX_FORECAST_PERIODS = 365

# ===== ПАРАМЕТРЫ МОДЕЛЕЙ =====
SARIMA_PARAMS = {
    "order": (1, 0, 0),
    "seasonal_order": (1, 1, 1, 7),
}

# ===== СООБЩЕНИЯ ОБ ОШИБКАХ =====
ERROR_MESSAGES = {
    "DATASET_NOT_LOADED": "Датасет не загружен. Сначала загрузите данные через load_dataset.",
    "EMPTY_DATASET": "Пустой датасет",
    "INVALID_COLUMNS": "Не найдены обязательные колонки. Нужны колонки с датой и объёмом продаж.",
    "INSUFFICIENT_DATA": "Недостаточно данных для прогноза (минимум ~10 дат).",
    "STORE_NOT_FOUND": "Для указанных ID магазинов нет данных.",
    "INVALID_DATE_COLUMN": "Не удалось парсить колонку дат.",
    "TRANSACTION_DATA_TYPE": "Данные содержат отдельные транзакции, будут агрегированы по датам.",
}

# ===== ФОРМАТЫ ВЫВОДА =====
DATE_FORMAT = "%Y-%m-%d"
FLOAT_FORMAT = "{:,.2f}"

