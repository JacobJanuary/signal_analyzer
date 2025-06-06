#!/usr/bin/env python3
"""Debug script to test OI processing."""
from modules.oi_processor import OIProcessor
import json


def test_oi_processing(symbol: str = "BTC"):
    """Test OI processing for a symbol."""
    print(f"\n🔍 Тестирование обработки OI для {symbol}")
    print("=" * 50)

    processor = OIProcessor()
    result = processor.process_symbol(symbol)

    print(f"\n📊 Результаты:")
    print(f"- Ошибка: {result.error}")
    print(f"- Биржа: {result.source_exchange.value if result.source_exchange else 'None'}")
    print(
        f"- Средний OI (30 дней): {result.average_oi_usdt:,.2f} USDT" if result.average_oi_usdt else "- Средний OI: None")
    print(f"- Текущий OI: {result.current_oi_usdt:,.2f} USDT" if result.current_oi_usdt else "- Текущий OI: None")
    print(
        f"- Изменение: {result.change_percentage:.2f}%" if result.change_percentage is not None else "- Изменение: None")

    # Детальная проверка
    print(f"\n🔧 Детальная проверка объекта:")
    print(f"result.current_oi_usdt = {result.current_oi_usdt}")
    print(f"type(result.current_oi_usdt) = {type(result.current_oi_usdt)}")

    # Проверка всех атрибутов
    print(f"\n📋 Все атрибуты OIResult:")
    for attr in ['average_oi_usdt', 'current_oi_usdt', 'change_percentage', 'source_exchange', 'error']:
        value = getattr(result, attr, 'ATTRIBUTE_NOT_FOUND')
        print(f"  - {attr}: {value}")

    return result


def test_database_save():
    """Test saving to database."""
    from database.models import EnrichedSignalData, signal_repository
    from datetime import datetime, timezone

    print("\n🔍 Тестирование сохранения в БД")
    print("=" * 50)

    # Создаем тестовые данные
    test_data = EnrichedSignalData(
        signal_id=999999,  # Тестовый ID
        symbol="TEST",
        created_at=datetime.now(timezone.utc),
        oi_average_usdt=1000000.0,
        oi_now_usdt=1100000.0,
        oi_change_pct_usdt=10.0,
        oi_source_usdt="binance"
    )

    print("\n📊 Данные для сохранения:")
    for field, value in test_data.__dict__.items():
        if value is not None:
            print(f"  - {field}: {value}")

    # Попытка сохранения
    try:
        # Сначала проверим структуру таблицы
        from database.connection import get_db_cursor
        with get_db_cursor() as cursor:
            cursor.execute("DESCRIBE signals_10min_enriched")
            columns = cursor.fetchall()
            print("\n📋 Структура таблицы signals_10min_enriched:")
            for col in columns:
                if 'oi' in col['Field']:
                    print(f"  - {col['Field']}: {col['Type']}")
    except Exception as e:
        print(f"\n❌ Ошибка при проверке структуры таблицы: {e}")


if __name__ == "__main__":
    # Тест 1: Обработка OI
    symbol = input("Введите символ для тестирования (по умолчанию BTC): ").strip().upper() or "BTC"
    result = test_oi_processing(symbol)

    # Тест 2: Проверка БД
    if input("\n\nПроверить сохранение в БД? (y/n): ").lower() == 'y':
        test_database_save()