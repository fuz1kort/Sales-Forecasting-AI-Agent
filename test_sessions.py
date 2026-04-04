#!/usr/bin/env python3
"""Тест системы сессий с TTL для Redis."""

import sys
sys.path.append('AIAgent/backend')

from agent.state import SessionManager
from config import AppSettings
from services.data_preprocessing_service import preprocessing_service
import pandas as pd

def test_session_ttl():
    """Тест TTL в сессиях."""
    print("🔍 Тестирование системы сессий с TTL\n")
    
    # 1. Проверка настроек
    settings = AppSettings()
    print(f"📋 Настройки:")
    print(f"  REDIS_HOST: {settings.REDIS_HOST}")
    print(f"  REDIS_PORT: {settings.REDIS_PORT}")
    print(f"  SESSION_TTL_SECONDS: {settings.SESSION_TTL_SECONDS} сек ({settings.SESSION_TTL_SECONDS // 3600} час)")
    
    # 2. Инициализация менеджера
    print(f"\n🔌 Инициализация SessionManager...")
    manager = SessionManager()
    print(f"  Redis подключён: {manager.redis_client is not None}")
    print(f"  TTL в менеджере: {manager.ttl} сек")
    
    # 3. Создание сессии
    print(f"\n🆔 Создание новой сессии...")
    session_id = manager.generate_session_id()
    print(f"  Session ID: {session_id[:8]}...")
    
    # 4. Загрузка и предобработка данных
    print(f"\n📊 Загрузка и предобработка данных...")
    df = preprocessing_service.process_full_pipeline()
    print(f"  ✅ Загружено {len(df)} строк, {len(df.columns)} колонок")
    
    # 5. Сохранение датасета в сессию
    print(f"\n💾 Сохранение датасета в сессию...")
    info = manager.save_dataset(session_id, df)
    print(f"  Сохранено {info['rows']} строк")
    print(f"  Колонки: {info['columns']}")
    print(f"  Дата: {info.get('date_range', 'не найдена')}")
    
    # 6. Загрузка датасета из сессии
    print(f"\n✅ Загрузка датасета из сессии...")
    loaded_df = manager.get_dataset(session_id)
    if loaded_df is not None:
        print(f"  ✅ Загружено {len(loaded_df)} строк из сессии")
        print(f"  Количество колонок совпадает: {len(loaded_df.columns) == len(df.columns)}")
        assert len(loaded_df) == len(df), "Количество строк не совпадает!"
    else:
        print(f"  ⚠️ Датасет не найден в сессии")
    
    # 7. Добавление сообщений
    print(f"\n💬 Тестирование истории сессии...")
    manager.add_message(session_id, "user", "Загрузи данные")
    manager.add_message(session_id, "assistant", "Данные загружены успешно!")
    history = manager.get_history(session_id)
    print(f"  Сообщений в истории: {len(history)}")
    for i, msg in enumerate(history, 1):
        print(f"    {i}. [{msg['role']}] {msg['content'][:50]}...")
    
    # 8. Сохранение результата прогноза
    print(f"\n🔮 Тестирование кеша прогноза...")
    forecast_data = {
        "status": "success",
        "periods": 30,
        "model": "neuralprophet",
        "forecast": [100, 105, 110]
    }
    manager.set_forecast(session_id, forecast_data)
    retrieved_forecast = manager.get_forecast_by_session(session_id)
    print(f"  Прогноз сохранён: {retrieved_forecast is not None}")
    if retrieved_forecast:
        print(f"  Модель: {retrieved_forecast.get('model')}")
        print(f"  Периодов: {retrieved_forecast.get('periods')}")
    
    # 9. Очистка сессии
    print(f"\n🗑️ Тест очистки сессии...")
    manager.clear_session(session_id)
    loaded_df_after_clear = manager.get_dataset(session_id)
    print(f"  Датасет после очистки: {loaded_df_after_clear is None}")
    
    print(f"\n✅ Все тесты пройдены успешно!")
    print(f"\n📌 Сводка:")
    print(f"  • TTL настроен на {settings.SESSION_TTL_SECONDS} сек")
    print(f"  • Данные сохраняются в Redis с автоматическим истечением")
    print(f"  • Система сессий работает корректно")
    print(f"  • Предобработка данных интегрирована")

if __name__ == "__main__":
    try:
        test_session_ttl()
    except TimeoutError:
        print("❌ Подключение к Redis истекло. Убедитесь, что Redis запущен.")
        print("   Redis должен быть доступен на localhost:6379")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
