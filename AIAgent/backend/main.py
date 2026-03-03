"""
Главный модуль FastAPI.

Предоставляет REST API для взаимодействия с агентом.
Вся бизнес-логика вынесена в модуль agent.
"""

import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import AppSettings
from agent.agent import SmolSalesAgent

# Загружаем конфигурацию
settings = AppSettings()
settings.validate()

# Настройка логирования
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)



def _get_or_create_agent() -> SmolSalesAgent:
    """
    Получить новый экземпляр агента.

    Returns:
        Экземпляр SmolSalesAgent
    """
    return SmolSalesAgent(model_provider=settings.LLM_PROVIDER)


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
        from agent.tools.data import load_dataset
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

    logger.info(f"🚀 Запуск Sales Forecasting Agent API на {settings.HOST}:{settings.PORT}")
    logger.info(f"📚 Docs: http://localhost:{settings.PORT}/docs")

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
