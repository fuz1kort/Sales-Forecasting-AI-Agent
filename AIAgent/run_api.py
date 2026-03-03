#!/usr/bin/env python3
"""
Запуск Sales Forecasting Agent API
"""

import os
import sys
import uvicorn

# Добавляем backend папку в PYTHONPATH
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )

