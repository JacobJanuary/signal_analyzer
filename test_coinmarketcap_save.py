#!/usr/bin/env python3
"""Test CoinMarketCap data saving to database."""
from datetime import datetime, timezone
from modules.spot_usdt_processor import SpotUSDTProcessor
from database.models import EnrichedSignalData, signal_repository
from database.connection import get_db_cursor
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_test_signal_with_fk(symbol: str, token_id: int = 1) -> int:
    """Create a test signal in signals_10min table with proper FK."""
    with get_db_cursor() as cursor:
        # Сначала проверим, какие поля обязательные
        cursor.execute("""
                       SELECT COLUMN_NAME, IS_NULLABLE, COLUMN_DEFAULT
                       FROM INFORMATION_SCHEMA.COLUMNS
                       WHERE TABLE_SCHEMA = DATABASE()
                         AND TABLE_NAME = 'signals_10min'
                         AND IS_NULLABLE = 'NO'
                         AND COLUMN_DEFAULT IS NULL
                         AND COLUMN_NAME != 'id'
                       """)
        required_fields = cursor.fetchall()

        # Создаем запись с минимальными обязательными полями
        cursor.execute("""
                       INSERT INTO signals_10min (symbol, token_id)
                       VALUES (%s, %s)
                       """, (symbol, token_id))

        return cursor.lastrowid


def test_coinmarketcap_processing_and_save():
    """Test processing symbol that only exists on CoinMarketCap and saving to DB."""
    print("🧪 Тестирование обработки и сохранения данных CoinMarketCap")
    print("=" * 60)

    # Символы, которые точно есть в CoinMarketCap
    test_symbols = ['KOMA', 'SKYAI', 'MUBARAK']

    processor = SpotUSDTProcessor()
    created_signal_ids = []

    try:
        for symbol in test_symbols:
            print(f"\n📊 Тестирование {symbol}:")
            print("-" * 40)

            # 1. Создаем запись в signals_10min для FK
            signal_id = create_test_signal_with_fk(symbol)
            created_signal_ids.append(signal_id)
            print(f"✅ Создан сигнал ID: {signal_id}")

            # 2. Обработка через процессор
            result = processor.process_symbol(symbol)

            if result.error:
                print(f"❌ Ошибка обработки: {result.error}")
                continue

            print(f"✅ Данные получены от: {result.source_exchange.value}")
            print(
                f"   Средний объем: ${result.avg_volume_usdt:,.2f}" if result.avg_volume_usdt else "   Средний объем: None")
            print(
                f"   Текущий объем: ${result.current_volume_usdt:,.2f}" if result.current_volume_usdt else "   Текущий объем: None")
            print(
                f"   Вчерашний объем: ${result.yesterday_volume_usdt:,.2f}" if result.yesterday_volume_usdt else "   Вчерашний объем: None")
            print(
                f"   Средняя цена: ${result.avg_price_usdt:,.2f}" if result.avg_price_usdt else "   Средняя цена: None")
            print(
                f"   Текущая цена: ${result.current_price_usdt:,.2f}" if result.current_price_usdt else "   Текущая цена: None")

            # 3. Создание записи для сохранения
            enriched_data = EnrichedSignalData(
                signal_id=signal_id,  # Используем реальный signal_id
                symbol=symbol,
                created_at=datetime.now(timezone.utc),
                # Volume data
                spot_volume_usdt_average=result.avg_volume_usdt,
                spot_volume_usdt_current=result.current_volume_usdt,
                spot_volume_usdt_yesterday=result.yesterday_volume_usdt,
                spot_volume_usdt_change_current_to_yesterday=result.volume_change_current_to_yesterday,
                spot_volume_usdt_change_current_to_average=result.volume_change_current_to_average,
                spot_volume_source_usdt=result.source_exchange.value if result.source_exchange else None,
                # Price data
                spot_price_usdt_average=result.avg_price_usdt,
                spot_price_usdt_current=result.current_price_usdt,
                spot_price_usdt_yesterday=result.yesterday_price_usdt,
                spot_price_usdt_change_1h=result.price_change_1h,
                spot_price_usdt_change_24h=result.price_change_24h,
                spot_price_usdt_change_7d=result.price_change_7d,
                spot_price_usdt_change_30d=result.price_change_30d,
                spot_price_source_usdt=result.source_exchange.value if result.source_exchange else None
            )

            # 4. Попытка сохранения
            print("\n💾 Сохранение в БД...")
            try:
                success = signal_repository.save_enriched_data(enriched_data)
                if success:
                    print("✅ Данные успешно сохранены в БД")

                    # 5. Проверка сохраненных данных
                    with get_db_cursor() as cursor:
                        cursor.execute("""
                                       SELECT signal_id,
                                              symbol,
                                              spot_volume_source_usdt,
                                              spot_volume_usdt_average,
                                              spot_volume_usdt_current,
                                              spot_price_source_usdt,
                                              spot_price_usdt_current
                                       FROM signals_10min_enriched
                                       WHERE signal_id = %s
                                       """, (signal_id,))

                        saved_data = cursor.fetchone()
                        if saved_data:
                            print("\n📋 Проверка сохраненных данных:")
                            print(f"   signal_id: {saved_data['signal_id']}")
                            print(f"   symbol: {saved_data['symbol']}")
                            print(f"   source: {saved_data['spot_volume_source_usdt']}")
                            print(f"   avg_volume: ${float(saved_data['spot_volume_usdt_average']):,.2f}" if saved_data[
                                'spot_volume_usdt_average'] else "   avg_volume: None")
                            print(f"   current_volume: ${float(saved_data['spot_volume_usdt_current']):,.2f}" if
                                  saved_data['spot_volume_usdt_current'] else "   current_volume: None")
                            print(
                                f"   current_price: ${float(saved_data['spot_price_usdt_current']):,.2f}" if saved_data[
                                    'spot_price_usdt_current'] else "   current_price: None")
                        else:
                            print("❌ Данные не найдены в БД после сохранения")
                else:
                    print("❌ Не удалось сохранить данные в БД")
            except Exception as e:
                print(f"❌ Ошибка при сохранении: {e}")
                import traceback
                traceback.print_exc()

    finally:
        # Очистка тестовых данных
        cleanup_test_data(created_signal_ids)


def cleanup_test_data(signal_ids: list):
    """Очистка тестовых данных."""
    if not signal_ids:
        return

    print("\n\n🧹 Очистка тестовых данных...")
    try:
        with get_db_cursor() as cursor:
            # Удаляем из enriched
            placeholders = ','.join(['%s'] * len(signal_ids))
            cursor.execute(f"""
                DELETE FROM signals_10min_enriched 
                WHERE signal_id IN ({placeholders})
            """, signal_ids)
            enriched_deleted = cursor.rowcount

            # Удаляем из signals_10min
            cursor.execute(f"""
                DELETE FROM signals_10min 
                WHERE id IN ({placeholders})
            """, signal_ids)
            signals_deleted = cursor.rowcount

            print(f"✅ Удалено записей:")
            print(f"   signals_10min_enriched: {enriched_deleted}")
            print(f"   signals_10min: {signals_deleted}")
    except Exception as e:
        print(f"❌ Ошибка при очистке: {e}")


def test_specific_symbol(symbol: str):
    """Test a specific symbol with detailed debugging."""
    print(f"\n\n🔍 Детальный тест для {symbol}")
    print("=" * 60)

    processor = SpotUSDTProcessor()

    # Проверяем каждый источник отдельно
    print("\n1️⃣ Проверка Binance:")
    binance_result = processor._process_binance_spot(f"{symbol}USDT")
    if binance_result.error:
        print(f"   ❌ {binance_result.error}")
    else:
        print(f"   ✅ Данные получены")

    print("\n2️⃣ Проверка Bybit:")
    bybit_result = processor._process_bybit_spot(f"{symbol}USDT")
    if bybit_result.error:
        print(f"   ❌ {bybit_result.error}")
    else:
        print(f"   ✅ Данные получены")

    print("\n3️⃣ Проверка CoinMarketCap:")
    cmc_result = processor._process_coinmarketcap(symbol)
    if cmc_result.error:
        print(f"   ❌ {cmc_result.error}")
    else:
        print(f"   ✅ Данные получены")
        print(
            f"   Объем: ${cmc_result.current_volume_usdt:,.2f}" if cmc_result.current_volume_usdt else "   Объем: None")
        print(f"   Цена: ${cmc_result.current_price_usdt:,.2f}" if cmc_result.current_price_usdt else "   Цена: None")


if __name__ == "__main__":
    # Основной тест
    test_coinmarketcap_processing_and_save()

    # Детальный тест для проблемных символов
    print("\n\n" + "=" * 60)
    print("🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА")
    print("=" * 60)

    for symbol in ['KOMA', 'SKYAI']:
        test_specific_symbol(symbol)