"""
Обработка и предварительная подготовка данных.

Централизует логику работы с датафреймами, которая была
распределена между моделями прогнозирования.
"""

import pandas as pd
import numpy as np
from typing import Optional

from utils import find_columns, validate_dataset
from exceptions import InvalidDatasetError, InvalidParameterError


class DataPreprocessor:
    """Класс для предварительной обработки данных перед прогнозированием."""

    @staticmethod
    def parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Парсит колонку дат, добавляет синтетические даты если нужно.

        Args:
            df: DataFrame для обработки
            date_col: Имя колонки с датами

        Returns:
            DataFrame с обработанными датами
        """
        df = df.copy()

        if not date_col or date_col not in df.columns:
            # Добавляем синтетические даты
            df['synthetic_date'] = pd.date_range(
                start='2023-01-01',
                periods=len(df),
                freq='D'
            )
            return df

        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise InvalidDatasetError(f"Не удалось парсить колонку дат '{date_col}': {e}")

        return df

    @staticmethod
    def prepare_general_data(
        df: pd.DataFrame,
        date_col: str,
        sales_col: str
    ) -> pd.DataFrame:
        """
        Подготавливает данные для общего прогноза (без разделения по магазинам).

        Args:
            df: Исходный DataFrame
            date_col: Имя колонки с датами
            sales_col: Имя колонки с продажами

        Returns:
            DataFrame с колонками 'ds' (дата) и 'y' (продажи)

        Raises:
            InvalidDatasetError: Если недостаточно данных
        """
        df_clean = df[[date_col, sales_col]].copy().dropna().sort_values(date_col)

        # Агрегируем по датам (на случай дублей)
        daily_sales = df_clean.groupby(date_col)[sales_col].sum().reset_index()

        result = pd.DataFrame({
            'ds': daily_sales[date_col],
            'y': daily_sales[sales_col]
        })

        if len(result) < 2:
            raise InvalidDatasetError("Недостаточно точек для прогноза (минимум 2)")

        return result

    @staticmethod
    def prepare_store_data(
        df: pd.DataFrame,
        date_col: str,
        sales_col: str,
        store_col: str,
        store_ids: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Подготавливает данные для прогноза по магазинам.

        Args:
            df: Исходный DataFrame
            date_col: Имя колонки с датами
            sales_col: Имя колонки с продажами
            store_col: Имя колонки с ID магазина
            store_ids: Список конкретных ID магазинов (None = все)

        Returns:
            DataFrame с колонками 'ds', 'y', 'ID'

        Raises:
            InvalidDatasetError: Если нет данных для указанных магазинов
        """
        cols = [date_col, sales_col, store_col]
        df_clean = df[cols].copy().dropna()

        if store_ids:
            df_clean = df_clean[
                df_clean[store_col].astype(str).isin([str(s) for s in store_ids])
            ]

        if df_clean.empty:
            raise InvalidDatasetError(
                f"Нет данных для магазинов: {store_ids}"
            )

        df_clean[date_col] = pd.to_datetime(df_clean[date_col])

        df_grouped = (
            df_clean
            .groupby([store_col, date_col])[sales_col]
            .sum()
            .reset_index()
            .sort_values([store_col, date_col])
        )

        return pd.DataFrame({
            'ds': df_grouped[date_col],
            'y': df_grouped[sales_col],
            'ID': df_grouped[store_col].astype(str)
        })

    @staticmethod
    def make_daily_series(df: pd.DataFrame, date_col: str, sales_col: str) -> pd.Series:
        """
        Создаёт ежедневный временной ряд с заполнением пропусков.

        Args:
            df: Исходный DataFrame
            date_col: Имя колонки с датами
            sales_col: Имя колонки с продажами

        Returns:
            Series с индексом дат и значениями продаж
        """
        daily = df.groupby(date_col)[sales_col].sum()
        daily = daily.astype(float)

        # Заполняем пропуски нулями или интерполяцией
        daily = daily.reindex(
            pd.date_range(daily.index.min(), daily.index.max(), freq='D'),
            fill_value=0
        )

        return daily

    @staticmethod
    def validate_forecast_parameters(periods: int, max_periods: int = 365) -> bool:
        """
        Валидирует параметры прогноза.

        Args:
            periods: Количество периодов прогноза
            max_periods: Максимально допустимый горизонт

        Returns:
            True если валидно

        Raises:
            InvalidParameterError: Если параметры некорректны
        """
        if not isinstance(periods, int) or periods <= 0:
            raise InvalidParameterError(f"periods должен быть положительным целым числом, получено: {periods}")

        if periods > max_periods:
            raise InvalidParameterError(f"periods не может быть больше {max_periods}")

        return True

