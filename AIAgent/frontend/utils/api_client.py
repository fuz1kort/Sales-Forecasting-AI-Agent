"""Обёртка для всех API запросов к бэкенду."""
import os
import time
from typing import Optional, Dict, Any, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()


class APIClient:
    """Централизованный клиент для работы с API."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("API_BASE")
        self.session_id = None
        self.max_retries = 3
        self.retry_delay = 1  # секунды

    def _set_session_id(self, session_id: str):
        """Устанавливаем session_id из ответа сервера."""
        self.session_id = session_id

    def _retry_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Отправить запрос с повторами при ошибках соединения."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                if method == "POST":
                    return requests.post(url, **kwargs)
                elif method == "GET":
                    return requests.get(url, **kwargs)
            except (requests.ConnectionError, requests.Timeout, ConnectionResetError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Экспоненциальная задержка
                    continue
                raise
        
        raise last_exception if last_exception else Exception("Max retries exceeded")

    def upload_file(self, file_content: bytes, filename: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Загрузить файл и получить session_id."""
        try:
            files = {"file": (filename, file_content, "text/csv")}
            headers = {}

            # Если session_id уже есть — передаём его
            if self.session_id:
                headers["X-Session-ID"] = self.session_id

            response = self._retry_request(
                "POST",
                f"{self.base_url}/upload",
                files=files,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            # Сохраняем session_id если получен
            if result.get("session_id"):
                self._set_session_id(result["session_id"])
                try:
                    import streamlit as st
                    st.session_state.session_id = self.session_id
                except ImportError:
                    pass

            return result, self.session_id

        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "error": "Таймаут загрузки. Попробуйте меньший файл или проверьте соединение с сервером."
            }, None
        except requests.ConnectionError:
            return {
                "status": "error",
                "error": "Не удаётся подключиться к серверу. Проверьте, что бэкенд запущен."
            }, None
        except Exception as e:
            return {"status": "error", "error": f"Ошибка загрузки: {str(e)}"}, None

    def send_query(self, query: str) -> Dict[str, Any]:
        """Отправить запрос агенту."""
        try:
            data = {"query": query}
            headers = {"X-Session-ID": self.session_id} if self.session_id else {}

            response = self._retry_request(
                "POST",
                f"{self.base_url}/chat",
                data=data,
                headers=headers,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            if result.get("session_id"):
                self._set_session_id(result["session_id"])
                try:
                    import streamlit as st
                    st.session_state.session_id = self.session_id
                except ImportError:
                    pass
            return result

        except requests.exceptions.Timeout:
            return {"status": "error", "error": "Таймаут ответа от агента (>300 сек)"}
        except requests.ConnectionError:
            return {"status": "error", "error": "Не удалось подключиться к серверу для чата"}
        except Exception as e:
            return {"status": "error", "error": f"Ошибка чата: {str(e)}"}

    def build_forecast(
            self,
            file_content: bytes,
            periods: int,
            model_type: str = "ensemble",
            forecast_type: str = "general",
            store_ids: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Построить прогноз через API."""
        try:
            params = {
                "periods": periods,
                "model_type": model_type,
                "forecast_type": forecast_type,
            }
            if store_ids:
                params["store_ids"] = store_ids
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            files = {"file": ("forecast.csv", file_content, "text/csv")}
            headers = {"X-Session-ID": self.session_id} if self.session_id else {}

            response = self._retry_request(
                "POST",
                f"{self.base_url}/forecast",
                files=files,
                data=params,
                headers=headers,
                timeout=180
            )
            response.raise_for_status()
            result = response.json()
            if result.get("session_id"):
                self._set_session_id(result["session_id"])
                try:
                    import streamlit as st
                    st.session_state.session_id = self.session_id
                except ImportError:
                    pass
            return result

        except requests.exceptions.Timeout:
            return {"status": "error", "error": "Таймаут при построении прогноза (>180 сек)"}
        except requests.ConnectionError:
            return {"status": "error", "error": "Не удалось подключиться к серверу для прогноза"}
        except Exception as e:
            return {"status": "error", "error": f"Ошибка прогноза: {str(e)}"}

    def get_session_info(self) -> Dict[str, Any]:
        """Получить информацию о загруженной сессии и датасете."""
        try:
            headers = {"X-Session-ID": self.session_id} if self.session_id else {}
            response = self._retry_request(
                "GET",
                f"{self.base_url}/session_info",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": f"Ошибка получения информации о сессии: {str(e)}"}

    def get_session_history(self) -> Dict[str, Any]:
        """Получить историю чата для текущей сессии."""
        try:
            headers = {"X-Session-ID": self.session_id} if self.session_id else {}
            response = self._retry_request(
                "GET",
                f"{self.base_url}/session_history",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": f"Ошибка получения истории сессии: {str(e)}"}


    def run_backtest(
            self,
            test_days: int = 30,
            query_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Запустить backtest двумя способами:
        1. Через прямой эндпоинт (если есть)
        2. Через чат-запрос агенту
        """
        # Пробуем сначала отправить через чат-запрос (работает в текущей версии)
        if query_override:
            return self.send_query(query_override)
        else:
            query = f"Запусти сравнение моделей (backtest) на последние {test_days} дней. Покажи какая модель лучше."
            return self.send_query(query)