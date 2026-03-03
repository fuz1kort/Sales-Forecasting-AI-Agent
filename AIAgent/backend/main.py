"""Главный модуль FastAPI."""
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.smol_agent import SmolSalesAgent
from agent.state import get_session_manager
from config import AppSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = AppSettings()

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running"
    }

@app.post("/upload")
async def upload_dataset(
        file: UploadFile = File(...),
        session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Загрузить CSV-файл."""
    if not session_id:
        session_id = get_session_manager().generate_session_id()
        logger.info(f"🆕 Новая сессия создана: {session_id}")

    try:
        content = await file.read()
        csv_text = content.decode("utf-8", errors="ignore")

        from agent.tools.data import load_dataset
        result = load_dataset(
            csv_content=csv_text,
            session_id=session_id
        )

        result["session_id"] = session_id
        return result
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки: {str(e)}")

@app.post("/chat")
async def chat_endpoint(
        query: str = Form(...),
        session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Обработать запрос пользователя."""
    if not session_id:
        session_id = get_session_manager().generate_session_id()
        logger.info(f"🆕 Новая сессия для чата: {session_id}")

    try:
        agent = SmolSalesAgent(model_provider=settings.LLM_PROVIDER, session_id=session_id)
        query = f"Пожалуйста, ответь на русском: {query}"
        result = await agent.run_async(query)
        result["session_id"] = session_id
        return result
    except Exception as e:
        logger.error(f"Ошибка чата: {e}")
        raise HTTPException(status_code=500, detail=str(e))