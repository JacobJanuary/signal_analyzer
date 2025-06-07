#!/usr/bin/env python3
"""Test full signal processing cycle."""
from database.models import SignalRecord
from main import SignalEnrichmentProcessor
from database.connection import get_db_cursor
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_test_signal(symbol: str) -> int:
    """Create a test signal in signals_10min table."""
    print(f"\n📝 Создание тестового сигнала для {symbol}...")

    with get_db_cursor() as cursor:
        # Проверяем структуру таблицы signals_10min
        cursor.execute("SHOW COLUMNS FROM signals_10min")
        columns = cursor.fetchall()
        print("\nСтруктура таблицы signals_10min:")
        for col in columns[:5]:  # Показываем первые 5 колонок
            print(f"  - {col['Field']}: {col['Type']}")

        # Создаем минимальную запись
        cursor.execute("""
                       INSERT INTO signals_10min (symbol)
                       VALUES (%s)
                       """, (symbol,))

        signal_id = cursor.lastrowid
        print(f"✅ Создан сигнал ID: {signal_id}")
        return signal_id


def test_full_processing(symbol: str):
    """Test full processing for a symbol."""
    print(f"\n🔄 Полный тест обработки для {symbol}")
    print("=" * 60)

    # 1. Создаем тестовый сигнал
    signal_id = create_test_signal(symbol)

    try:
        # 2. Запускаем обработку
        print("\n🚀 Запуск обработки...")
        processor = SignalEnrichmentProcessor()
        processed = processor.process_new_signals()

        print(f"\n✅ Обработано сигналов: {processed}")

        # 3. Проверяем результат
        print("\n📊 Проверка результатов в БД...")
        with get_db_cursor() as cursor:
            cursor.execute("""
                           SELECT signal_id,
                                  symbol,
                                  oi_source_usdt,
                                  oi_usdt_current,
                                  spot_volume_source_usdt,
                                  spot_volume_usdt_current,
                                  spot_price_source_usdt,
                                  spot_price_usdt_current,
                                  created_at
                           FROM signals_10min_enriched
                           WHERE signal_id = %s
                           """, (signal_id,))

            result = cursor.fetchone()

            if result:
                print("\n✅ Данные успешно сохранены:")
                print(f"   Signal ID: {result['signal_id']}")
                print(f"   Symbol: {result['symbol']}")
                print(f"   Created: {result['created_at']}")

                print("\n   📈 Open Interest:")
                print(f"      Source: {result['oi_source_usdt'] or 'N/A'}")
                print(f"      Current: ${float(result['oi_usdt_current']):,.2f}" if result[
                    'oi_usdt_current'] else "      Current: N/A")

                print("\n   📊 Spot Volume:")
                print(f"      Source: {result['spot_volume_source_usdt'] or 'N/A'}")
                print(f"      Current: ${float(result['spot_volume_usdt_current']):,.2f}" if result[
                    'spot_volume_usdt_current'] else "      Current: N/A")

                print("\n   💰 Spot Price:")
                print(f"      Source: {result['spot_price_source_usdt'] or 'N/A'}")
                print(f"      Current: ${float(result['spot_price_usdt_current']):,.2f}" if result[
                    'spot_price_usdt_current'] else "      Current: N/A")
            else:
                print("❌ Данные не найдены в таблице enriched")

    finally:
        # 4. Очистка
        print("\n🧹 Очистка тестовых данных...")
        with get_db_cursor() as cursor:
            cursor.execute("DELETE FROM signals_10min WHERE id = %s", (signal_id,))
            cursor.execute("DELETE FROM signals_10min_enriched WHERE signal_id = %s", (signal_id,))
            print("✅ Тестовые данные удалены")


def test_multiple_symbols():
    """Test multiple symbols with different sources."""
    symbols = [
        ('BTC', 'Should use Binance/Bybit'),
        ('KOMA', 'Should use CoinMarketCap'),
        ('SKYAI', 'Should use CoinMarketCap'),
    ]

    for symbol, description in symbols:
        print(f"\n\n{'=' * 60}")
        print(f"🧪 Тест: {symbol} - {description}")
        print(f"{'=' * 60}")
        test_full_processing(symbol)


if __name__ == "__main__":
    print("🔧 Тестирование полного цикла обработки сигналов")
    print("=" * 60)

    choice = input("""
Выберите тест:
1. Тест одного символа
2. Тест нескольких символов (BTC, KOMA, SKYAI)

Ваш выбор (1-2): """)

    if choice == "1":
        symbol = input("Введите символ для тестирования: ").strip().upper()
        if symbol:
            test_full_processing(symbol)
    elif choice == "2":
        test_multiple_symbols()