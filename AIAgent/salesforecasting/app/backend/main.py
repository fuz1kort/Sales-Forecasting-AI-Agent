"""
Главный модуль FastAPI.

Предоставляет REST API для взаимодействия с агентом.
Вся бизнес-логика вынесена в модуль agent.
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Импортируем агента
from salesforecasting.app.backend.agent.agent import SmolSalesAgent

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title="Sales Forecasting Agent API",
    description="Интеллектуальный агент для прогнозирования продаж на базе smolagents",
    version="3.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище активных агентов: {session_id: SmolSalesAgent}
_active_agents: dict[str, SmolSalesAgent] = {}


def _get_or_create_agent() -> SmolSalesAgent:
    """
    Получить существующего агента или создать нового.

    Returns:
        Экземпляр SmolSalesAgent
    """
    return SmolSalesAgent(model_provider=os.getenv("LLM_PROVIDER", "yandex"))


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Sales Forecasting Agent API",
        "version": "3.0",
        "status": "running"
    }


@app.post("/chat")
async def chat(
        query: str = Form(...),
):
    """
    Отправить запрос агенту и получить ответ.
    
    Поддерживает session_id через Form-параметр или заголовок.
    """

    try:
        agent = _get_or_create_agent()
        result = await agent.run_async(query)
        return result

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_dataset(
        file: UploadFile = File(...),
        session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """
    Загрузить CSV-файл с данными о продажах.
    
    Файл автоматически обрабатывается инструментом load_dataset.
    """

    try:
        # Читаем содержимое файла
        content = await file.read()
        csv_text = content.decode("utf-8", errors="ignore")

        # Вызываем инструмент напрямую (без LLM)
        from salesforecasting.app.backend.agent.tools.data import load_dataset
        result = load_dataset(
            csv_content=csv_text
        )

        return {
            **result
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки: {str(e)}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    print(f"🚀 Запуск Sales Forecasting Agent API на {host}:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")

    uvicorn.run(
        "salesforecasting.app.backend.main:app",
        host=host,
        port=port,
        reload=True
    )