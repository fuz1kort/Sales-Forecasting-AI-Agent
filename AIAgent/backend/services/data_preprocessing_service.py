"""Сервис для предобработки данных продаж."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessingService:
    """Сервис для загрузки и предобработки данных транзакций."""

    def __init__(self, data_path: Optional[str] = None):
        """
        Инициализация сервиса.

        Args:
            data_path: Путь к CSV файлу с данными. Если None, используется путь по умолчанию.
        """
        if data_path is None:
            # Путь к встроенному датасету в проекте
            workspace_root = Path(__file__).parent.parent.parent
            self.data_path = workspace_root / "backend" / "data" / "retail_full.csv"
        else:
            self.data_path = Path(data_path)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Загрузка сырых данных из CSV файла.

        Returns:
            DataFrame с сырыми данными
        """
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"✅ Загружено {len(df)} строк, {len(df.columns)} колонок из {self.data_path}")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных: конвертация типов, очистка, флаги.

        Args:
            df: Сырые данные

        Returns:
            Предобработанный DataFrame
        """
        df = df.copy()

        # Адаптация к структуре retail_full.csv
        # Переименовываем колонки для единообразия
        column_mapping = {
            'InvoiceDate': 'Date',
            'Invoice': 'TransactionNo',
            'StockCode': 'ProductNo',
            'Customer ID': 'CustomerNo',
            'Description': 'ProductName'
        }
        df = df.rename(columns=column_mapping)

        # Конвертация типов
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['TransactionNo'] = df['TransactionNo'].astype(str)
        df['ProductNo'] = df['ProductNo'].astype(str)
        df['CustomerNo'] = df['CustomerNo'].fillna(0).astype(int).astype(str)

        # Флаг отмены транзакции (отрицательное количество)
        df['IsCancelled'] = (df['Quantity'] < 0) | (df['TransactionNo'].str.startswith('C', na=False))

        # Удаление строк с критическими пропусками
        df = df.dropna(subset=['Date', 'Price', 'Quantity'])

        # Фильтрация отменённых транзакций (опционально)
        # df = df[~df['IsCancelled']].copy()

        logger.info(f"✅ Предобработка завершена: {len(df)} строк после очистки")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание дополнительных признаков.

        Args:
            df: Предобработанные данные

        Returns:
            DataFrame с признаками
        """
        df = df.copy()

        # === Временные признаки ===
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['Quarter'] = df['Date'].dt.quarter

        # Сезонность (для ритейла)
        df['IsHolidaySeason'] = df['Month'].isin([11, 12]).astype(int)  # ноябрь-декабрь

        # === Финансовые признаки ===
        df['TransactionValue'] = (df['Price'] * df['Quantity']).round(2)
        df['TransactionValueAbs'] = df['TransactionValue'].abs()  # для анализа независимо от отмен

        # === Признаки клиента ===
        # Частота покупок (можно агрегировать позже)
        df['IsRepeatCustomer'] = df.groupby('CustomerNo')['CustomerNo'].transform('count') > 1

        # === Признаки страны ===
        # Топ-страны (остальные -> 'Other' для уменьшения размерности)
        top_countries = df['Country'].value_counts().nlargest(10).index
        df['CountryGrouped'] = df['Country'].apply(lambda x: x if x in top_countries else 'Other')

        # === Признаки транзакции ===
        df['ItemsPerTransaction'] = df.groupby('TransactionNo')['TransactionNo'].transform('count')

        logger.info(f"✅ Признаки созданы: {len(df.columns)} колонок")
        return df

    def aggregate_daily_product(self, df: pd.DataFrame, exclude_cancelled: bool = True) -> pd.DataFrame:
        """
        Агрегация данных по дням и продуктам.

        Args:
            df: DataFrame с признаками
            exclude_cancelled: Исключать отменённые транзакции

        Returns:
            Агрегированный DataFrame
        """
        if exclude_cancelled:
            df = df[~df['IsCancelled']].copy()

        # Дата без времени для группировки
        df['DateOnly'] = df['Date'].dt.floor('D')

        agg_df = df.groupby(['DateOnly', 'ProductNo']).agg(
            # === Основная информация ===
            ProductName=('ProductName', 'first'),
            Price=('Price', 'mean'),  # средняя цена, если были изменения

            # === Продажи ===
            TotalQuantity=('Quantity', 'sum'),
            TotalRevenue=('TransactionValue', 'sum'),
            NumTransactions=('TransactionNo', 'nunique'),

            # === Клиенты ===
            UniqueCustomers=('CustomerNo', 'nunique'),
            CountryMode=('Country', lambda x: x.mode()[0] if not x.empty else 'Unknown'),

        ).reset_index()

        logger.info(f"✅ Агрегация завершена: {len(agg_df)} строк (день × товар)")
        return agg_df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Кодирование категориальных признаков.

        Args:
            df: Агрегированный DataFrame

        Returns:
            DataFrame с закодированными признаками
        """
        df = df.copy()

        # Для деревьев (RandomForest, XGBoost) можно использовать Label Encoding
        le_cols = ['ProductNo', 'CustomerNo', 'CountryGrouped']
        for col in le_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

        # Для линейных моделей лучше One-Hot Encoding для CountryGrouped
        # df = pd.get_dummies(df, columns=['CountryGrouped'], prefix='country')

        logger.info("✅ Категориальные признаки закодированы")
        return df

    def clean_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка финального датасета от ненужных колонок.

        Args:
            df: DataFrame после кодирования

        Returns:
            Очищенный DataFrame
        """
        # Колонки для удаления (зависит от задачи)
        cols_to_drop = [
            'TransactionNo',      # ID транзакции
            'Date',               # Дата
            'PriceCategory',      # one-hot, можно удалить оригинал
            'QuantityCategory',
        ]

        # Удаляем только если они есть в датасете
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        logger.info(f"✅ Финальная очистка: {len(df)} строк, {len(df.columns)} колонок")
        return df

    def process_full_pipeline(self) -> pd.DataFrame:
        """
        Полный пайплайн обработки данных.

        Returns:
            Полностью обработанный DataFrame для моделирования
        """
        logger.info("🚀 Начало полного пайплайна обработки данных")

        # 1. Загрузка
        df = self.load_raw_data()

        # 2. Предобработка
        df = self.preprocess_data(df)

        # 3. Создание признаков
        df = self.create_features(df)

        # 4. Агрегация
        df = self.aggregate_daily_product(df)

        # 5. Кодирование
        df = self.encode_categorical_features(df)

        # 6. Очистка
        df = self.clean_final_dataset(df)

        logger.info("✅ Пайплайн обработки данных завершён")
        return df


# Глобальный экземпляр сервиса
preprocessing_service = DataPreprocessingService()