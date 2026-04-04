"""Управление сессиями через Redis."""
import json
import logging
import uuid
from typing import Dict, List, Any, Optional
import pandas as pd
import redis

from backend.config import AppSettings

logger = logging.getLogger(__name__)

class SessionManager:
    """Менеджер сессий с хранением в Redis."""

    def __init__(self, host: str = None, port: int = None):
        settings = AppSettings()
        self.host = host or settings.REDIS_HOST
        self.port = int(port or settings.REDIS_PORT)
        self.ttl = settings.SESSION_TTL_SECONDS

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=0,
                decode_responses=True,
                socket_timeout=1,
                socket_connect_timeout=1,
                health_check_interval=0
            )
            self.redis_client.ping()
            logger.info(f"✅ Подключено к Redis на {self.host}:{self.port} (TTL: {self.ttl}s)")
        except redis.ConnectionError:
            logger.critical(f"❌ Не удалось подключиться к Redis")
            self.redis_client = None
            self.in_memory_store = {}
            logger.warning("⚠️ Используем in-memory storage вместо Redis")

    DATASET_KEY = "dataset:"
    HISTORY_KEY = "history:"
    INFO_KEY = "info:"
    FORECAST_KEY = "forecast:"
    BACKTEST_KEY = "backtest:"

    @classmethod
    def generate_session_id(cls) -> str:
        """Генерирует уникальный ID сессии."""
        return str(uuid.uuid4())

    def save_dataset(self, session_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Сохраняет датасет для конкретной сессии."""
        from backend.utils import find_columns

        key = f"{self.DATASET_KEY}{session_id}"
        info_key = f"{self.INFO_KEY}{session_id}"

        if self.redis_client:
            df_csv = df.to_csv(index=False)
            self.redis_client.set(key, df_csv, ex=self.ttl)

            date_col, sales_col, store_col, product_col = find_columns(df)
            info = {
                "rows": len(df),
                "columns": list(df.columns),
                "date_column": date_col,
                "sales_column": sales_col,
                "store_column": store_col,
            }

            if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                info["date_range"] = {
                    "start": str(df[date_col].min()),
                    "end": str(df[date_col].max())
                }

            if store_col:
                info["stores_count"] = df[store_col].nunique()

            self.redis_client.set(info_key, json.dumps(info), ex=self.ttl)
        else:
            self.in_memory_store[key] = df.to_csv(index=False)
            self.in_memory_store[info_key] = json.dumps({
                "rows": len(df),
                "columns": list(df.columns)
            })

        logger.debug(f"💾 Датасет сохранён для сессии {session_id[:8]}...")
        return info

    def get_dataset(self, session_id: str) -> Optional[pd.DataFrame]:
        """Загружает датасет для конкретной сессии."""
        key = f"{self.DATASET_KEY}{session_id}"

        if self.redis_client:
            csv_str = self.redis_client.get(key)
            if csv_str:
                import io
                df = pd.read_csv(io.StringIO(csv_str))
                # Конвертируем дату обратно в datetime
                self._convert_date_column(df)
                return df
        else:
            csv_str = self.in_memory_store.get(key)
            if csv_str:
                import io
                df = pd.read_csv(io.StringIO(csv_str))
                # Конвертируем дату обратно в datetime
                self._convert_date_column(df)
                return df

        logger.debug(f"📂 Нет датасета для сессии {session_id[:8]}")
        return None

    def _convert_date_column(self, df: pd.DataFrame):
        """Конвертирует колонку даты в datetime."""
        if df.empty:
            return
        
        # Ищем колонку даты по ключевым словам
        date_keywords = ["date", "data", "дата", "invoice", "инвойс"]
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in date_keywords):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.debug(f"✓ Converted date column '{col}' to datetime")
                    break  # Конвертируем только первую найденную колонку
                except Exception as e:
                    logger.debug(f"⚠️ Could not convert column '{col}' to datetime: {e}")
                    continue

    def add_message(self, session_id: str, role: str, content: str):
        """Добавляет сообщение в историю сессии."""
        history_key = f"{self.HISTORY_KEY}{session_id}"
        message = {"role": role, "content": content}

        if self.redis_client:
            self.redis_client.rpush(history_key, json.dumps(message))
            self.redis_client.expire(history_key, self.ttl)
            self.redis_client.ltrim(history_key, 0, 99)
        else:
            if history_key not in self.in_memory_store:
                self.in_memory_store[history_key] = []
            self.in_memory_store[history_key].append(json.dumps(message))
            self.in_memory_store[history_key] = self.in_memory_store[history_key][-100:]

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, str]]:
        """Получает историю сессии."""
        history_key = f"{self.HISTORY_KEY}{session_id}"

        if self.redis_client:
            messages_json = self.redis_client.lrange(history_key, 0, limit - 1)
            return [json.loads(m) for m in messages_json]
        else:
            messages_json = self.in_memory_store.get(history_key, [])
            return [json.loads(m) for m in messages_json[-limit:]]

    def set_forecast(self, session_id: str, forecast_data: Dict[str, Any]):
        """Сохраняет результат прогноза."""
        key = f"{self.FORECAST_KEY}{session_id}"

        if self.redis_client:
            self.redis_client.set(key, json.dumps(forecast_data), ex=self.ttl)
        else:
            self.in_memory_store[key] = json.dumps(forecast_data)

    def get_forecast_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Получает результат последнего прогноза."""
        key = f"{self.FORECAST_KEY}{session_id}"

        if self.redis_client:
            result = self.redis_client.get(key)
            return json.loads(result) if result else None
        else:
            result = self.in_memory_store.get(key)
            return json.loads(result) if result else None

    def clear_session(self, session_id: str):
        """Очищает все данные сессии."""
        prefix_keys = [self.DATASET_KEY, self.HISTORY_KEY, self.INFO_KEY,
                       self.FORECAST_KEY, self.BACKTEST_KEY]
        keys_to_delete = [f"{prefix}{session_id}" for prefix in prefix_keys]

        if self.redis_client:
            self.redis_client.delete(*keys_to_delete)
        else:
            for key in keys_to_delete:
                self.in_memory_store.pop(key, None)

        logger.info(f"🗑 Очистка сессии {session_id[:8]}")


_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """Получить глобальный экземпляр менеджера сессий."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager