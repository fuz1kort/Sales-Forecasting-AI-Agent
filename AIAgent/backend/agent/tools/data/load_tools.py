"""Инструменты для загрузки и управления датасетами."""
import io
import logging
import json
from typing import Optional

import pandas as pd
import numpy as np
from smolagents import tool

from backend.agent.state import get_session_manager
from backend.services.data_preprocessing_service import preprocessing_service
from backend.utils import find_columns

logger = logging.getLogger(__name__)


# === Вспомогательная функция для JSON сериализации ===

def _convert_to_serializable(obj):
    """Конвертирует numpy типы в обычные Python типы для JSON сериализации."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj


# === Вспомогательные функции для работы с датасетом ===

def _get_dataset(session_id: Optional[str]) -> Optional[pd.DataFrame]:
    """Безопасное получение датасета из сессии."""
    sid = session_id or "default"
    return get_session_manager().get_dataset(sid)

def _detect_product_column(df: pd.DataFrame) -> Optional[str]:
    """Эвристика для поиска колонки с названием товара."""
    keywords = ["product", "item", "товар", "продукт", "sku", "name", "название", "articul", "description", "описание"]
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            # Исключаем колонки, которые явно не товары
            if any(ex in col_lower for ex in ["category", "категория", "type", "тип"]):
                continue
            return col
    return None


def _detect_quantity_column(df: pd.DataFrame) -> Optional[str]:
    """Эвристика для поиска колонки с количеством товара."""
    keywords = ["quantity", "qty", "количество", "шт", "amount", "count"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return None


def _detect_price_column(df: pd.DataFrame) -> Optional[str]:
    """Эвристика для поиска колонки с ценой товара."""
    keywords = ["price", "unitprice", "unit_price", "цена", "cost", "rate"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return None


def _detect_category_column(df: pd.DataFrame) -> Optional[str]:
    """Эвристика для поиска колонки с категорией товара."""
    keywords = ["category", "категория", "type", "тип", "group", "группа", "class"]
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            return col
    return None


def _filter_bad_products(df: pd.DataFrame, product_col: str) -> pd.DataFrame:
    """
    Фильтрует плохие товары (корректировки, списания, потери инвентаря).
    
    Удаляет товары, названия которых содержат признаки браков:
    - Adjust, adjustment (корректировки)
    - wrong, wrongly (неправильно)
    - missing, lost, gone (потеряно, потеря)
    - given away, oops (списания, ошибки)
    - ?sold, sold individual, sold sets (неопределённость)
    - No Stock, FBA, found, display (системные записи)
    - Amazon (маркетплейс специфичное)
    
    Args:
        df: Исходный DataFrame
        product_col: Название колонки товаров
        
    Returns:
        DataFrame с отфильтрованными товарами
    """
    bad_keywords = [
        "adjust", "adjustment", "корректив", "корректур",
        "wrong", "wrongly", "неправ",
        "missing", "потер", "потеря", "lost", "gone", "went missing",
        "given away", "даден", "списа",
        "oops", "ошибка",
        "?", "unknown", "unkn",
        "no stock", "нет товара", "out of stock",
        "fba", "fulfillment",
        "found", "found again", "found some",
        "display", "демо",
        "amazon", "амазон",
        "not rcvd", "not received", "не получен",
        "merchant", "chandler", "credit error", "sto",
        "can't find", "cant find",
        "came coded as",
        "broken", "uneven bottom", "broken zips", "broken glass",
        "barcode problem",
        "bad quality",
        "amendment",
        "allocate stock", "dotcom orders",
        "alan hodge", "cant mamage",
        "add stock", "online orders",
        "zebra", "invcing error",
        "wet", "rusty", "thrown away", "water damaged", "wet/mouldy",
        "unsaleable", "destroyed",
        "error", "problem", "damaged", "mouldy", "rusty",
        "invcing", "invoicing", "invoice",
        "stock allocation", "stock for orders",
        "website fixed", "faulty", "eurobargain invc/credit",
        "ebay sales", "ebay", "dotcom sold sets", "dotcom sales",
        "for online retail orders", "dotcom email", "had been put aside",
        "discoloured", "dirty", "did a credit and did not tick ret",
        "debenhams", "damges", "donated to the food chain charity",
        "incorrectly entered", "for show", "phil said so", "john lewis",
        "on cargo order", "non colour fast", "mystery", "mixed up",
        "mix up with c", "mailout addition", "mailout", "label mix up",
        "incorrectly credited", "incorrect credit", "damages/showroom etc",
        "damages/credits from asos", "damages wax", "damages etc",
        "damages/samples", "damages", "dagamed", "85123a mixed",
        "dotcom sets", "show samples", "sold as 1 on dotcom",
        "sold as a/b", "sold as c", "crushed", "carton qnty was 216 not 144 as stat",
        "travel card wallet dotcomgiftshop",
    ]
    
    # Создаём маску для плохих товаров (используем regex=False для простого поиска подстрок)
    bad_mask = pd.Series(False, index=df.index)
    for kw in bad_keywords:
        bad_mask |= df[product_col].astype(str).str.lower().str.contains(
            kw, 
            case=False, 
            na=False,
            regex=False
        )
    
    bad_count = bad_mask.sum()
    if bad_count > 0:
        logger.debug(f"🗑️ Найдено {bad_count} плохих товаров, удаляем из анализа")
        logger.debug(f"Плохие товары: {df[bad_mask][product_col].unique().tolist()}")
    
    # Возвращаем только хорошие товары
    return df[~bad_mask].copy()


def _apply_data_filters(df: pd.DataFrame, product_col: Optional[str], sales_col: str, context: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Применяет все фильтры к датасету (плохие товары, нулевые значения).
    
    Args:
        df: Исходный DataFrame
        product_col: Название колонки товаров (или None)
        sales_col: Название колонки продаж
        context: Контекст для логирования (например, "при загрузке" или "при загрузке встроенных данных")
        
    Returns:
        Кортеж (отфильтрованный DataFrame, словарь со статистикой удаления)
    """
    initial_rows = len(df)
    stats = {"bad_products": 0, "zero_sales": 0, "zero_quantity": 0, "zero_price": 0, "total": 0}
    
    # Фильтруем плохие товары (корректировки, списания, потери)
    if product_col:
        df = _filter_bad_products(df, product_col)
        stats["bad_products"] = initial_rows - len(df)
        if stats["bad_products"] > 0:
            logger.info(f"🗑️ Удалено {stats['bad_products']} плохих товаров {context}")
    
    # Фильтруем транзакции с нулевой выручкой
    zero_sales = (df[sales_col] <= 0).sum()
    if zero_sales > 0:
        df = df[df[sales_col] > 0]
        stats["zero_sales"] = zero_sales
        logger.info(f"💰 Удалено {zero_sales} транзакций с нулевой выручкой {context}")
    
    # Фильтруем транзакции с нулевым количеством (если колонка найдена)
    quantity_col = _detect_quantity_column(df)
    if quantity_col and quantity_col in df.columns:
        zero_quantity = (df[quantity_col] <= 0).sum()
        if zero_quantity > 0:
            df = df[df[quantity_col] > 0]
            stats["zero_quantity"] = zero_quantity
            logger.info(f"📦 Удалено {zero_quantity} транзакций с нулевым количеством {context}")
    
    # Фильтруем транзакции с нулевой ценой (если колонка найдена)
    price_col = _detect_price_column(df)
    if price_col and price_col in df.columns:
        zero_price = (df[price_col] <= 0).sum()
        if zero_price > 0:
            df = df[df[price_col] > 0]
            stats["zero_price"] = zero_price
            logger.info(f"💵 Удалено {zero_price} транзакций с нулевой ценой {context}")
    
    stats["total"] = initial_rows - len(df)
    if stats["total"] > 0:
        logger.info(f"✅ Очищено {stats['total']} записей из {initial_rows} {context}")
    
    return df, stats


@tool
def load_dataset(
        csv_content: Optional[str] = None,
        use_builtin_data: bool = False,
        encoding: str = "utf-8",
        session_id: Optional[str] = None
) -> dict:
    """
    Загрузить датасет продаж из CSV-строки или встроенного датасета.

    Вызывайте, когда пользователь загрузил файл, вставил CSV-данные или хочет использовать встроенный датасет.

    Args:
        csv_content: Содержимое CSV-файла как строка (опционально)
        use_builtin_data: Загрузить встроенный датасет вместо CSV (по умолчанию False)
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
    df = None
    
    if use_builtin_data:
        # Загружаем встроенный датасет
        try:
            df = preprocessing_service.process_full_pipeline()
            logger.debug("✓ Built-in dataset loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading built-in data: {e}")
            return {"status": "error", "error": f"Не удалось загрузить встроенный датасет: {str(e)}"}
    else:
        # Загружаем из CSV строки
        if not csv_content:
            return {"status": "error", "error": "CSV содержимое не предоставлено"}
        
        # Пробуем разные кодировки для надёжности
        encodings_to_try = [encoding, 'utf-8-sig', 'cp1252', 'latin-1']

        for enc in encodings_to_try:
            try:
                df = pd.read_csv(io.StringIO(csv_content), encoding=enc)
                logger.debug(f"✓ Dataset loaded with encoding: {enc}")
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

    # Trimming строковых колонок (удаляем пробелы в начале/конце)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Авто-поиск обязательных колонок
    date_col, sales_col, store_col, product_col = find_columns(df)

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

    # Определяем тип данных (транзакции или агрегированные)
    from backend.utils import detect_transaction_data, aggregate_transactions, get_data_structure_info
    
    # Анализируем структуру данных
    data_structure = get_data_structure_info(df)
    is_transaction_data = data_structure["has_transactions"]
    
    # Формируем информационное сообщение о структуре
    data_type_info = ""
    if is_transaction_data:
        data_type_info = f"📊 Обнаружены транзакционные данные (отдельные операции).\n"
        data_type_info += f"   📍 Location фактор: {data_structure['location_type']} ({data_structure['location_column']})\n"
        data_type_info += f"   📦 Товар: {data_structure['product_column']}\n"
        if data_structure['price_quantity']:
            data_type_info += f"   💰 Выручка вычисляется: Price × Quantity\n"
    else:
        data_type_info = f"📈 Данные уже агрегированы (по датам/периодам).\n"
        data_type_info += f"   📍 Location: {data_structure['location_column']}\n"

    # Фильтруем плохие товары и транзакции с нулевыми значениями
    df, filter_stats = _apply_data_filters(df, product_col, sales_col, "при загрузке")

    # Сохраняем в сессию
    session_manager = get_session_manager()
    sid = session_id or "default"
    info = session_manager.save_dataset(sid, df)

    # Логирование в конце
    logger.info(f"✓ Dataset loaded and cleaned: {info['rows']:,} rows")

    result = {
        "status": "success",
        "rows": info["rows"],
        "columns": list(df.columns),
        "date_column": date_col,
        "sales_column": sales_col,
        "store_column": store_col,
        "store_type": data_structure['location_type'],  # ⭐ важно для фильтров
        "product_column": product_col,
        "data_type": "transactions" if is_transaction_data else "aggregated",
        "data_structure": data_structure,  # полная информация о структуре
        "data_type_info": data_type_info,
        "date_range": info.get("date_range"),
        "message": f"✅ Загружено {info['rows']:,} строк за период {info.get('date_range', 'N/A')}"
    }
    
    # Конвертируем в JSON-совместимый формат
    return _convert_to_serializable(result)


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

    date_col, sales_col, store_col, product_col = find_columns(df)

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