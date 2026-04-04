@echo off
echo Starting Sales Forecasting AI Agent...
echo ==========================================
echo.
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Create virtual environment: python -m venv .venv
    pause
    exit /b 1
)
echo.
echo Activating virtual environment...
call .venv\Scripts\activate
echo.
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Copy .env.example to .env and configure environment variables
    echo copy .env.example .env
    pause
    exit /b 1
)
echo.
echo Starting Backend (FastAPI)...
start /B python backend/main.py
timeout /t 3 /nobreak >nul
echo.
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8000/' -UseBasicParsing -TimeoutSec 5; if ($response.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo SUCCESS: Backend started at http://localhost:8000
) else (
    echo ERROR: Backend failed to start!
    pause
    exit /b 1
)
echo.
echo Starting Frontend (Streamlit)...
start /B python -m streamlit run frontend/app.py
timeout /t 2 /nobreak >nul
echo.
echo.
echo APPLICATION STARTED!
echo Frontend: http://localhost:8501
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo To stop, close terminal windows or press Ctrl+C
echo.
pause 
