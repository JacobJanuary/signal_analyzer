#!/usr/bin/env python3
"""Final integration test for the complete signal enrichment system."""
from datetime import datetime, timezone
from database.connection import get_db_cursor
from main import SignalEnrichmentProcessor
from utils.logger import setup_logger
import time

logger = setup_logger(__name__)


def create_test_signals():
    """Create test signals for different scenarios."""
    test_cases = [
        # (symbol, token_id, description)
        ('BTC', 1, 'Major crypto - should use Binance/Bybit'),
        ('ETH', 2, 'Major crypto - should use Binance/Bybit'),
        ('KOMA', 3, 'Only on CoinMarketCap'),
        ('SKYAI', 4, 'Only on CoinMarketCap'),
        ('MUBARAK', 5, 'Available on Binance and CoinMarketCap'),
    ]

    created_ids = []

    print("📝 Создание тестовых сигналов...")
    print("=" * 60)

    try:
        with get_db_cursor() as cursor:
            for symbol, token_id, description in test_cases:
                cursor.execute("""
                               INSERT INTO signals_10min (symbol, token_id)
                               VALUES (%s, %s)
                               """, (symbol, token_id))

                signal_id = cursor.lastrowid
                created_ids.append(signal_id)
                print(f"✅ {symbol:<10} (ID: {signal_id}) - {description}")

        return created_ids

    except Exception as e:
        print(f"❌ Ошибка создания сигналов: {e}")
        return []


def check_enriched_results(signal_ids):
    """Check the enriched results in the database."""
    print("\n\n📊 Проверка обогащенных данных...")
    print("=" * 60)

    with get_db_cursor() as cursor:
        placeholders = ','.join(['%s'] * len(signal_ids))
        cursor.execute(f"""
            SELECT 
                e.*,
                s.symbol
            FROM signals_10min_enriched e
            JOIN signals_10min s ON s.id = e.signal_id
            WHERE e.signal_id IN ({placeholders})
            ORDER BY e.signal_id
        """, signal_ids)

        results = cursor.fetchall()

        if not results:
            print("❌ Нет обогащенных данных")
            return

        for row in results:
            print(f"\n{'=' * 50}")
            print(f"📈 {row['symbol']} (Signal ID: {row['signal_id']})")
            print(f"{'=' * 50}")

            # OI данные
            if row['oi_source_usdt']:
                print(f"\n💹 Open Interest (Источник: {row['oi_source_usdt']}):")
                if row['oi_usdt_current']:
                    print(f"   Текущий: ${float(row['oi_usdt_current']):,.2f}")
                if row['oi_usdt_average']:
                    print(f"   Средний: ${float(row['oi_usdt_average']):,.2f}")
                if row['oi_usdt_change_current_to_average'] is not None:
                    print(f"   Изменение к среднему: {float(row['oi_usdt_change_current_to_average']):.2f}%")
            else:
                print("\n💹 Open Interest: Нет данных")

            # Spot Volume данные
            if row['spot_volume_source_usdt']:
                print(f"\n📊 Spot Volume (Источник: {row['spot_volume_source_usdt']}):")
                if row['spot_volume_usdt_current']:
                    print(f"   Текущий: ${float(row['spot_volume_usdt_current']):,.2f}")
                if row['spot_volume_usdt_average']:
                    print(f"   Средний: ${float(row['spot_volume_usdt_average']):,.2f}")
                if row['spot_volume_usdt_yesterday']:
                    print(f"   Вчерашний: ${float(row['spot_volume_usdt_yesterday']):,.2f}")
                if row['spot_volume_usdt_change_current_to_yesterday'] is not None:
                    print(f"   Изменение к вчера: {float(row['spot_volume_usdt_change_current_to_yesterday']):.2f}%")
            else:
                print("\n📊 Spot Volume: Нет данных")

            # Spot Price данные
            if row['spot_price_source_usdt']:
                print(f"\n💰 Spot Price (Источник: {row['spot_price_source_usdt']}):")
                if row['spot_price_usdt_current']:
                    price = float(row['spot_price_usdt_current'])
                    if price > 1:
                        print(f"   Текущая: ${price:,.2f}")
                    else:
                        print(f"   Текущая: ${price:.8f}")
                if row['spot_price_usdt_change_24h'] is not None:
                    print(f"   Изменение 24h: {float(row['spot_price_usdt_change_24h']):.2f}%")
                if row['spot_price_usdt_change_7d'] is not None:
                    print(f"   Изменение 7d: {float(row['spot_price_usdt_change_7d']):.2f}%")
            else:
                print("\n💰 Spot Price: Нет данных")

        print(f"\n\n✅ Всего обработано записей: {len(results)} из {len(signal_ids)}")

        # Статистика по источникам
        sources = {
            'oi': {},
            'volume': {},
            'price': {}
        }

        for row in results:
            if row['oi_source_usdt']:
                sources['oi'][row['oi_source_usdt']] = sources['oi'].get(row['oi_source_usdt'], 0) + 1
            if row['spot_volume_source_usdt']:
                sources['volume'][row['spot_volume_source_usdt']] = sources['volume'].get(
                    row['spot_volume_source_usdt'], 0) + 1
            if row['spot_price_source_usdt']:
                sources['price'][row['spot_price_source_usdt']] = sources['price'].get(row['spot_price_source_usdt'],
                                                                                       0) + 1

        print("\n📊 Статистика по источникам:")
        print(f"   OI: {dict(sources['oi'])}")
        print(f"   Volume: {dict(sources['volume'])}")
        print(f"   Price: {dict(sources['price'])}")


def cleanup_test_data(signal_ids):
    """Clean up test data."""
    if not signal_ids:
        return

    print("\n\n🧹 Очистка тестовых данных...")

    try:
        with get_db_cursor() as cursor:
            placeholders = ','.join(['%s'] * len(signal_ids))

            # Удаляем из enriched
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


def run_integration_test():
    """Run the complete integration test."""
    print("🚀 ФИНАЛЬНЫЙ ИНТЕГРАЦИОННЫЙ ТЕСТ")
    print("=" * 60)
    print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    signal_ids = []

    try:
        # 1. Создаем тестовые сигналы
        signal_ids = create_test_signals()

        if not signal_ids:
            print("❌ Не удалось создать тестовые сигналы")
            return

        # 2. Небольшая пауза
        print("\n⏳ Ожидание 2 секунды...")
        time.sleep(2)

        # 3. Запускаем обработку
        print("\n🔄 Запуск обработки сигналов...")
        print("-" * 60)

        start_time = time.time()
        processor = SignalEnrichmentProcessor()
        processed_count = processor.process_new_signals()
        processing_time = time.time() - start_time

        print(f"\n✅ Обработка завершена:")
        print(f"   Обработано сигналов: {processed_count}")
        print(f"   Время обработки: {processing_time:.2f} сек")
        print(f"   Среднее время на сигнал: {processing_time / processed_count:.2f} сек" if processed_count > 0 else "")

        # 4. Проверяем результаты
        check_enriched_results(signal_ids)

    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 5. Очистка
        if signal_ids and input("\n\nОчистить тестовые данные? (y/n): ").lower() == 'y':
            cleanup_test_data(signal_ids)

    print("\n" + "=" * 60)
    print(f"Тест завершен: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_test()