#!/usr/bin/env python3
"""Test token_id functionality."""
from database.connection import get_db_cursor
from api_clients.coinmarketcap_client import CoinMarketCapClient
from modules.spot_usdt_processor import SpotUSDTProcessor


def check_tokens_table():
    """Check tokens table content."""
    print("📋 Проверка таблицы tokens")
    print("=" * 60)

    with get_db_cursor() as cursor:
        # Получаем несколько записей для примера
        cursor.execute("""
                       SELECT id, symbol, token_id
                       FROM tokens
                       WHERE token_id IS NOT NULL LIMIT 10
                       """)

        tokens = cursor.fetchall()

        print(f"Найдено токенов с token_id: {len(tokens)}\n")

        for token in tokens:
            print(f"ID: {token['id']:<5} Symbol: {token['symbol']:<10} Token_ID: {token['token_id']}")

        # Проверяем конкретные токены
        print("\n🔍 Проверка конкретных токенов:")

        test_symbols = ['KOMA', 'SKYAI', 'BTC', 'MUBARAK']

        for symbol in test_symbols:
            cursor.execute("""
                           SELECT symbol, token_id
                           FROM tokens
                           WHERE symbol = %s
                           """, (symbol,))

            result = cursor.fetchone()
            if result:
                print(f"✅ {symbol:<10} - token_id: {result['token_id']}")
            else:
                print(f"❌ {symbol:<10} - не найден в таблице tokens")


def test_cmc_with_token_id():
    """Test CoinMarketCap API with token_id."""
    print("\n\n🔍 Тест CoinMarketCap с token_id")
    print("=" * 60)

    cmc = CoinMarketCapClient()

    if not cmc.is_configured():
        print("❌ API ключ не настроен")
        return

    # Тестируем KOMA
    symbol = 'KOMA'

    # 1. Получаем token_id из БД
    token_id = cmc.get_token_id_from_db(symbol)

    if token_id:
        print(f"✅ Найден token_id для {symbol}: {token_id}")

        # 2. Получаем последнюю котировку
        latest = cmc.get_latest_quote(symbol)
        if latest and 'quote' in latest and 'USD' in latest['quote']:
            usd = latest['quote']['USD']
            print(f"\n📊 Последняя котировка:")
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
            print(f"   Точек данных: {market_data['data_points']}")
    else:
        print(f"❌ token_id не найден для {symbol}")


def test_full_processing():
    """Test full processing with token_id support."""
    print("\n\n🔄 Полный тест обработки с поддержкой token_id")
    print("=" * 60)

    processor = SpotUSDTProcessor()

    test_symbols = ['KOMA', 'SKYAI']

    for symbol in test_symbols:
        print(f"\n📊 Обработка {symbol}:")
        print("-" * 40)

        result = processor.process_symbol(symbol)

        if result.error:
            print(f"❌ Ошибка: {result.error}")
        else:
            print(f"✅ Источник: {result.source_exchange.value}")
            print(
                f"   Средний объем: ${result.avg_volume_usdt:,.2f}" if result.avg_volume_usdt else "   Средний объем: N/A")
            print(
                f"   Текущий объем: ${result.current_volume_usdt:,.2f}" if result.current_volume_usdt else "   Текущий объем: N/A")
            print(
                f"   Текущая цена: ${result.current_price_usdt:.8f}" if result.current_price_usdt else "   Текущая цена: N/A")


def update_missing_token_ids():
    """Check if we need to update token_ids in the database."""
    print("\n\n🔧 Проверка отсутствующих token_id")
    print("=" * 60)

    with get_db_cursor() as cursor:
        # Находим токены без token_id
        cursor.execute("""
                       SELECT id, symbol
                       FROM tokens
                       WHERE token_id IS NULL LIMIT 10
                       """)

        missing = cursor.fetchall()

        if missing:
            print(f"Найдено токенов без token_id: {len(missing)}")

            cmc = CoinMarketCapClient()
            if cmc.is_configured():
                print("\nПопытка получить token_id из CoinMarketCap:")

                for token in missing[:3]:  # Проверяем первые 3
                    symbol_info = cmc.get_symbol_map(token['symbol'])
                    if symbol_info and 'id' in symbol_info:
                        print(
                            f"✅ {token['symbol']:<10} - CMC ID: {symbol_info['id']} ({symbol_info.get('name', 'N/A')})")

                        # Можно обновить в БД
                        # cursor.execute("""
                        #     UPDATE tokens
                        #     SET token_id = %s
                        #     WHERE id = %s
                        # """, (symbol_info['id'], token['id']))
                    else:
                        print(f"❌ {token['symbol']:<10} - не найден в CoinMarketCap")
        else:
            print("✅ Все токены имеют token_id")


if __name__ == "__main__":
    # 1. Проверяем таблицу tokens
    check_tokens_table()

    # 2. Тестируем CoinMarketCap с token_id
    test_cmc_with_token_id()

    # 3. Тестируем полную обработку
    test_full_processing()

    # 4. Проверяем отсутствующие token_id
    update_missing_token_ids()