# Sales Forecasting Backend

This is the backend service for the Sales Forecasting AI project.

## Setup

1. Create and activate a virtual environment  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install prophet python-dotenv requests
   ```
3. Copy `.env.example` to `.env` and fill in your YandexGPT configuration:
   ```bash
   cp .env.example .env
   ```

   Required variables:
   - `YANDEX_API_KEY` – API‑ключ Yandex Cloud
   - `YANDEX_FOLDER_ID` – ID каталога в Yandex Cloud
   - (опционально) `YANDEX_MODEL_URI` – полный URI модели, например `gpt://<FOLDER_ID>/yandexgpt-lite`

## Running the Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

This will start the FastAPI server at `http://localhost:8000`.

## API Endpoints

- `POST /forecast` - Upload a CSV file for sales forecasting using Prophet
- `POST /embed-index` - Index a CSV file for embeddings (placeholder)
- `POST /chat` - Chat with the AI analyst about your sales data (uses YandexGPT)
- `GET /` - Health check endpoint
- `GET /test-yandexgpt` - Test YandexGPT API connectivity

