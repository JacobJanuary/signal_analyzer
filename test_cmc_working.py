#!/usr/bin/env python3
"""Test CoinMarketCap integration with the working example."""
from api_clients.coinmarketcap_client import CoinMarketCapClient
from modules.spot_usdt_processor import SpotUSDTProcessor
from datetime import datetime, timezone


def test_cmc_direct():
    """Test CoinMarketCap API directly."""
    print("🔍 Прямой тест CoinMarketCap API")
    print("=" * 60)

    cmc = CoinMarketCapClient()

    if not cmc.is_configured():
        print("❌ API ключ не настроен")
        return

    test_symbols = ['KOMA', 'SKYAI', 'BTC']

    for symbol in test_symbols:
        print(f"\n📊 Тестирование {symbol}:")
        print("-" * 40)

        # 1. Получаем маппинг символа
        symbol_info = cmc.get_symbol_map(symbol)
        if symbol_info:
            print(f"✅ Найден: {symbol_info['name']} (ID: {symbol_info['id']})")
        else:
            print(f"❌ Не найден")
            continue

        # 2. Получаем последнюю котировку
        latest = cmc.get_latest_quote(symbol)
        if latest and 'quote' in latest and 'USD' in latest['quote']:
            usd = latest['quote']['USD']
            print(f"   Цена: ${usd.get('price', 0):.8f}")
            print(f"   Объем 24h: ${usd.get('volume_24h', 0):,.2f}")
            print(f"   Изменение 24h: {usd.get('percent_change_24h', 0):.2f}%")

        # 3. Получаем исторические данные
        market_data = cmc.get_market_data_for_symbol(symbol)
        if market_data:
            print(f"\n📈 Рыночные данные (30 дней):")
            print(f"   Средний объем: ${market_data['avg_volume_usdt']:,.2f}")
            print(f"   Текущий объем: ${market_data['current_volume_usdt']:,.2f}")
            print(f"   Вчерашний объем: ${market_data['yesterday_volume_usdt']:,.2f}")
            print(f"   Средняя цена: ${market_data['avg_price_usd']:.8f}")
            print(f"   Текущая цена: ${market_data['current_price_usd']:.8f}")
            print(f"   Точек данных: {market_data['data_points']}")
        else:
            print("❌ Не удалось получить рыночные данные")


def test_spot_processor():
    """Test spot processor with CoinMarketCap."""
    print("\n\n🔄 Тест процессора Spot USDT")
    print("=" * 60)

    processor = SpotUSDTProcessor()

    # Тестируем символы, которые должны использовать CoinMarketCap
    test_cases = [
        ('KOMA', 'Should use CoinMarketCap'),
        ('SKYAI', 'Should use CoinMarketCap'),
        ('BTC', 'Should use Binance/Bybit first')
    ]

    for symbol, description in test_cases:
        print(f"\n📊 {symbol} - {description}:")
        print("-" * 40)

        result = processor.process_symbol(symbol)

        if result.error:
            print(f"❌ Ошибка: {result.error}")
        else:
            print(f"✅ Источник: {result.source_exchange.value}")
            if result.avg_volume_usdt:
                print(f"   Средний объем: ${result.avg_volume_usdt:,.2f}")
            if result.current_volume_usdt:
                print(f"   Текущий объем: ${result.current_volume_usdt:,.2f}")
            if result.yesterday_volume_usdt:
                print(f"   Вчерашний объем: ${result.yesterday_volume_usdt:,.2f}")
            if result.volume_change_current_to_yesterday is not None:
                print(f"   Изменение объема (к вчера): {result.volume_change_current_to_yesterday:.2f}%")
            if result.current_price_usdt:
                print(f"   Текущая цена: ${result.current_price_usdt:.8f}")
            if result.price_change_24h is not None:
                print(f"   Изменение цены 24h: {result.price_change_24h:.2f}%")


def test_save_to_db():
    """Test saving CoinMarketCap data to database."""
    print("\n\n💾 Тест сохранения в БД")
    print("=" * 60)

    from database.models import EnrichedSignalData, signal_repository
    from database.connection import get_db_cursor

    # Создаем тестовый сигнал
    symbol = 'KOMA'
    signal_id = None

    try:
        with get_db_cursor() as cursor:
            cursor.execute("""
                           INSERT INTO signals_10min (symbol, token_id)
                           VALUES (%s, %s)
                           """, (symbol, 1))
            signal_id = cursor.lastrowid
            print(f"✅ Создан тестовый сигнал ID: {signal_id}")

        # Обрабатываем через процессор
        processor = SpotUSDTProcessor()
        result = processor.process_symbol(symbol)

        if not result.error and result.source_exchange.value == 'coinmarketcap':
            print(f"✅ Данные получены от CoinMarketCap")

            # Создаем запись для сохранения
            enriched = EnrichedSignalData(
                signal_id=signal_id,
                symbol=symbol,
                created_at=datetime.now(timezone.utc),
                spot_volume_usdt_average=result.avg_volume_usdt,
                spot_volume_usdt_current=result.current_volume_usdt,
                spot_volume_usdt_yesterday=result.yesterday_volume_usdt,
                spot_volume_usdt_change_current_to_yesterday=result.volume_change_current_to_yesterday,
                spot_volume_usdt_change_current_to_average=result.volume_change_current_to_average,
                spot_volume_source_usdt='coinmarketcap',
                spot_price_usdt_average=result.avg_price_usdt,
                spot_price_usdt_current=result.current_price_usdt,
                spot_price_usdt_yesterday=result.yesterday_price_usdt,
                spot_price_source_usdt='coinmarketcap'
            )

            # Сохраняем
            if signal_repository.save_enriched_data(enriched):
                print("✅ Данные успешно сохранены в БД")

                # Проверяем
                with get_db_cursor() as cursor:
                    cursor.execute("""
                                   SELECT *FROM signals_10min_enriched
                                   WHERE signal_id = %s
                    """, (signal_id,))
                    saved = cursor.fetchone()

                    if saved and saved['spot_volume_source_usdt'] == 'coinmarketcap':
                        print("✅ Проверка: данные от CoinMarketCap сохранены корректно")
                        print(f"   Объем: ${float(saved['spot_volume_usdt_current']):,.2f}")
                        print(f"   Цена: ${float(saved['spot_price_usdt_current']):.8f}")
            else:
                print("❌ Ошибка сохранения")
        else:
            print(f"❌ Не удалось получить данные от CoinMarketCap: {result.error}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Очистка
        if signal_id:
            try:
                with get_db_cursor() as cursor:
                    cursor.execute("DELETE FROM signals_10min_enriched WHERE signal_id = %s", (signal_id,))
                    cursor.execute("DELETE FROM signals_10min WHERE id = %s", (signal_id,))
                    print("\n✅ Тестовые данные очищены")
            except Exception as e:
                print(f"❌ Ошибка при очистке: {e}")


if __name__ == "__main__":
    print("🧪 Тестирование интеграции с CoinMarketCap")
    print("=" * 60)

    # 1. Прямой тест API
    test_cmc_direct()

    # 2. Тест через процессор
    test_spot_processor()

    # 3. Тест сохранения в БД
    if input("\n\nВыполнить тест сохранения в БД? (y/n): ").lower() == 'y':
        test_save_to_db()