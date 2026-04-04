#!/bin/bash
# Sales Forecasting Agent - Quick Start Script
# Запуск приложения в фоне

echo "🚀 Sales Forecasting AI Agent - Quick Start"
echo "=========================================="

# Проверка наличия виртуального окружения
if [ ! -d ".venv" ]; then
    echo "❌ Виртуальное окружение не найдено!"
    echo "Создайте виртуальное окружение: python -m venv .venv"
    exit 1
fi

# Активация виртуального окружения
echo "🔧 Активация виртуального окружения..."
source .venv/bin/activate  # для Linux/macOS
# или для Windows: .venv\Scripts\activate

# Проверка наличия .env файла
if [ ! -f ".env" ]; then
    echo "⚠️  Файл .env не найден!"
    echo "Скопируйте .env.example в .env и настройте переменные окружения"
    echo "cp .env.example .env"
    exit 1
fi

# Запуск backend в фоне
echo "🔧 Запуск Backend (FastAPI)..."
python backend/main.py &
BACKEND_PID=$!

# Ожидание запуска backend
sleep 3

# Проверка работы backend
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend запущен: http://localhost:8000"
else
    echo "❌ Backend не запустился!"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Запуск frontend
echo "🎨 Запуск Frontend (Streamlit)..."
python -m streamlit run frontend/app.py &
FRONTEND_PID=$!

# Ожидание запуска frontend
sleep 2

echo ""
echo "🎉 Приложение запущено!"
echo "📱 Frontend: http://localhost:8501"
echo "🔧 Backend:  http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Для остановки нажмите Ctrl+C"

# Ожидание завершения
trap "echo '🛑 Остановка...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait</content>
<parameter name="filePath">C:\Users\gafar\source\fuz1kort\Sales-Forecasting-AI-Agent\AIAgent\start.sh