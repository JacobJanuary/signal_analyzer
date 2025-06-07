#!/usr/bin/env python3
"""Diagnose issues with database saving and CoinMarketCap."""
from database.connection import get_db_cursor
from database.models import EnrichedSignalData, signal_repository
from api_clients.coinmarketcap_client import CoinMarketCapClient
from datetime import datetime, timezone
import traceback


def check_database_structure():
    """Check if all required columns exist in the table."""
    print("🔍 Проверка структуры таблицы signals_10min_enriched")
    print("=" * 60)

    try:
        with get_db_cursor() as cursor:
            cursor.execute("""
                           SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                           FROM INFORMATION_SCHEMA.COLUMNS
                           WHERE TABLE_SCHEMA = DATABASE()
                             AND TABLE_NAME = 'signals_10min_enriched'
                           ORDER BY ORDINAL_POSITION
                           """)

            columns = cursor.fetchall()

            print(f"Найдено колонок: {len(columns)}\n")

            # Проверяем важные колонки
            important_columns = [
                'signal_id', 'symbol', 'created_at',
                'spot_volume_usdt_average', 'spot_volume_usdt_current',
                'spot_volume_usdt_yesterday', 'spot_volume_usdt_change_current_to_yesterday',
                'spot_volume_usdt_change_current_to_average', 'spot_volume_source_usdt',
                'spot_price_usdt_average', 'spot_price_usdt_current',
                'spot_price_usdt_yesterday', 'spot_price_source_usdt'
            ]

            existing_columns = {col['COLUMN_NAME'] for col in columns}

            print("Проверка важных колонок:")
            for col_name in important_columns:
                if col_name in existing_columns:
                    print(f"✅ {col_name}")
                else:
                    print(f"❌ {col_name} - ОТСУТСТВУЕТ!")

    except Exception as e:
        print(f"❌ Ошибка при проверке структуры: {e}")
        traceback.print_exc()


def test_minimal_save():
    """Test saving minimal data."""
    print("\n\n🧪 Тест минимального сохранения")
    print("=" * 60)

    # Создаем минимальную запись
    minimal_data = EnrichedSignalData(
        signal_id=888888,
        symbol="TEST_MINIMAL",
        created_at=datetime.now(timezone.utc)
    )

    print("Попытка сохранить минимальные данные...")
    print(f"  signal_id: {minimal_data.signal_id}")
    print(f"  symbol: {minimal_data.symbol}")
    print(f"  created_at: {minimal_data.created_at}")

    try:
        # Прямое сохранение через SQL для диагностики
        with get_db_cursor() as cursor:
            cursor.execute("""
                           INSERT INTO signals_10min_enriched (signal_id, symbol, created_at)
                           VALUES (%s, %s, %s)
                           """, (minimal_data.signal_id, minimal_data.symbol, minimal_data.created_at))

            print("✅ Минимальное сохранение успешно!")

            # Удаляем тестовую запись
            cursor.execute("DELETE FROM signals_10min_enriched WHERE signal_id = 888888")

    except Exception as e:
        print(f"❌ Ошибка при минимальном сохранении: {e}")
        traceback.print_exc()


def test_foreign_key_constraint():
    """Check if foreign key constraint is the issue."""
    print("\n\n🔍 Проверка Foreign Key ограничений")
    print("=" * 60)

    try:
        with get_db_cursor() as cursor:
            # Проверяем есть ли FK constraint
            cursor.execute("""
                           SELECT CONSTRAINT_NAME,
                                  COLUMN_NAME,
                                  REFERENCED_TABLE_NAME,
                                  REFERENCED_COLUMN_NAME
                           FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                           WHERE TABLE_SCHEMA = DATABASE()
                             AND TABLE_NAME = 'signals_10min_enriched'
                             AND REFERENCED_TABLE_NAME IS NOT NULL
                           """)

            constraints = cursor.fetchall()

            if constraints:
                print("Найдены Foreign Key ограничения:")
                for constraint in constraints:
                    print(
                        f"  - {constraint['COLUMN_NAME']} -> {constraint['REFERENCED_TABLE_NAME']}.{constraint['REFERENCED_COLUMN_NAME']}")

                print("\n⚠️  Это может быть причиной ошибки!")
                print("Для сохранения записи signal_id должен существовать в таблице signals_10min")
            else:
                print("Foreign Key ограничения не найдены")

    except Exception as e:
        print(f"❌ Ошибка при проверке FK: {e}")


def test_coinmarketcap_symbols():
    """Test which symbols are available in CoinMarketCap."""
    print("\n\n🔍 Проверка символов в CoinMarketCap")
    print("=" * 60)

    cmc = CoinMarketCapClient()

    if not cmc.is_configured():
        print("❌ CoinMarketCap API key не настроен")
        return

    # Список символов для проверки
    test_symbols = [
        'KOMA', 'KOMAINU', 'KOMA-INU',  # Возможные варианты для KOMA
        'SKYAI', 'SKY', 'SKYNET',  # Возможные варианты для SKYAI
        'MUBARAK',  # Этот работает на Binance
        'PEPE', 'SHIB', 'DOGE'  # Известные мем-коины для проверки
    ]

    print("Проверка доступности символов:\n")

    for symbol in test_symbols:
        symbol_info = cmc.get_symbol_map(symbol)
        if symbol_info:
            print(f"✅ {symbol:<12} - ID: {symbol_info['id']}, Name: {symbol_info['name']}")
        else:
            print(f"❌ {symbol:<12} - Не найден")


def test_full_save_with_fk():
    """Test saving with proper foreign key."""
    print("\n\n🧪 Тест полного сохранения с правильным FK")
    print("=" * 60)

    try:
        with get_db_cursor() as cursor:
            # 1. Создаем запись в signals_10min
            cursor.execute("""
                           INSERT INTO signals_10min (symbol)
                           VALUES ('TEST_CMC')
                           """)
            test_signal_id = cursor.lastrowid
            print(f"✅ Создан тестовый сигнал ID: {test_signal_id}")

            # 2. Создаем полную запись для enriched
            enriched_data = EnrichedSignalData(
                signal_id=test_signal_id,
                symbol='TEST_CMC',
                created_at=datetime.now(timezone.utc),
                spot_volume_usdt_average=1000000.0,
                spot_volume_usdt_current=1100000.0,
                spot_volume_usdt_yesterday=950000.0,
                spot_volume_usdt_change_current_to_yesterday=15.79,
                spot_volume_usdt_change_current_to_average=10.0,
                spot_volume_source_usdt='coinmarketcap',
                spot_price_usdt_average=100.0,
                spot_price_usdt_current=105.0,
                spot_price_usdt_yesterday=98.0,
                spot_price_source_usdt='coinmarketcap'
            )

            # 3. Сохраняем через repository
            print("\nСохранение через repository...")
            success = signal_repository.save_enriched_data(enriched_data)

            if success:
                print("✅ Данные успешно сохранены!")

                # Проверяем
                cursor.execute("""
                               SELECT *
                               FROM signals_10min_enriched
                               WHERE signal_id = %s
                               """, (test_signal_id,))

                saved = cursor.fetchone()
                if saved:
                    print(f"\nСохраненные данные:")
                    print(f"  Symbol: {saved['symbol']}")
                    print(f"  Source: {saved['spot_volume_source_usdt']}")
                    print(f"  Volume: ${float(saved['spot_volume_usdt_current']):,.2f}")
            else:
                print("❌ Не удалось сохранить через repository")

            # 4. Очистка
            cursor.execute("DELETE FROM signals_10min_enriched WHERE signal_id = %s", (test_signal_id,))
            cursor.execute("DELETE FROM signals_10min WHERE id = %s", (test_signal_id,))
            print("\n✅ Тестовые данные очищены")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("🔧 Диагностика проблем с сохранением данных")
    print("=" * 60)

    # 1. Проверяем структуру БД
    check_database_structure()

    # 2. Тестируем минимальное сохранение
    test_minimal_save()

    # 3. Проверяем FK constraints
    test_foreign_key_constraint()

    # 4. Проверяем символы в CoinMarketCap
    test_coinmarketcap_symbols()

    # 5. Тестируем полное сохранение
    test_full_save_with_fk()