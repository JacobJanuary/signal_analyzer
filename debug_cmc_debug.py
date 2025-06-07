#!/usr/bin/env python3
"""Detailed debug of CoinMarketCap API issues."""
from api_clients.coinmarketcap_client import CoinMarketCapClient
from datetime import datetime, timedelta, timezone
import json


def debug_cmc_api():
    """Debug CoinMarketCap API calls."""
    print("🔍 Детальная отладка CoinMarketCap API")
    print("=" * 60)

    cmc = CoinMarketCapClient()

    if not cmc.is_configured():
        print("❌ API ключ не настроен")
        return

    # Тестируем KOMA
    symbol = 'KOMA'
    print(f"\n📊 Тестирование {symbol}:")

    # 1. Получаем информацию о символе
    print("\n1️⃣ Получение информации о символе:")
    symbol_info = cmc.get_symbol_map(symbol)
    if symbol_info:
        print(f"✅ Найден: {symbol_info['name']} (ID: {symbol_info['id']})")
        print(f"   Slug: {symbol_info.get('slug', 'N/A')}")
        print(f"   Rank: {symbol_info.get('rank', 'N/A')}")
        print(f"   Is Active: {symbol_info.get('is_active', 'N/A')}")

        # 2. Получаем последнюю котировку
        print("\n2️⃣ Получение последней котировки:")
        latest_quote = cmc.get_latest_quote(symbol)
        if latest_quote:
            print("✅ Котировка получена:")
            if 'quote' in latest_quote and 'USD' in latest_quote['quote']:
                usd_data = latest_quote['quote']['USD']
                print(f"   Цена: ${usd_data.get('price', 0):.8f}")
                print(f"   Объем 24h: ${usd_data.get('volume_24h', 0):,.2f}")
                print(f"   Market Cap: ${usd_data.get('market_cap', 0):,.2f}")
                print(f"   Изменение 24h: {usd_data.get('percent_change_24h', 0):.2f}%")
            else:
                print("❌ Нет данных USD в котировке")
                print(f"   Данные: {json.dumps(latest_quote, indent=2)}")
        else:
            print("❌ Не удалось получить котировку")

        # 3. Получаем исторические данные
        print("\n3️⃣ Получение исторических данных:")
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=30)

        historical_quotes = cmc.get_historical_quotes(
            symbol, start_time, end_time, interval='24h', count=30
        )

        if historical_quotes:
            print(f"✅ Получено записей: {len(historical_quotes)}")
            # Показываем последние 3 записи
            for i, quote in enumerate(historical_quotes[-3:]):
                timestamp = quote.get('timestamp', 'N/A')
                if 'quote' in quote and 'USD' in quote['quote']:
                    usd = quote['quote']['USD']
                    print(
                        f"   [{i + 1}] {timestamp}: ${usd.get('price', 0):.8f}, Vol: ${usd.get('volume_24h', 0):,.2f}")
        else:
            print("❌ Не удалось получить исторические данные")

        # 4. Тестируем метод get_market_data_for_symbol
        print("\n4️⃣ Тестирование get_market_data_for_symbol:")
        market_data = cmc.get_market_data_for_symbol(symbol)
        if market_data:
            print("✅ Рыночные данные получены:")
            for key, value in market_data.items():
                if isinstance(value, float):
                    print(f"   {key}: ${value:,.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ Не удалось получить рыночные данные")
    else:
        print(f"❌ Символ {symbol} не найден")


def test_direct_api_call():
    """Test direct API call to debug issues."""
    print("\n\n🔧 Прямой вызов API")
    print("=" * 60)

    cmc = CoinMarketCapClient()

    # Проверяем прямой вызов для KOMA
    print("\nПрямой запрос для KOMA (ID: 33405):")

    # Используем ID напрямую
    params = {
        'id': '33405',  # KOMA ID
        'convert': 'USD'
    }

    response = cmc._make_request('/v1/cryptocurrency/quotes/latest', params)
    if response:
        print("✅ Ответ получен:")
        print(json.dumps(response, indent=2)[:500] + "...")  # Первые 500 символов
    else:
        print("❌ Ошибка при прямом запросе")


def check_api_limits():
    """Check API rate limits and credits."""
    print("\n\n📊 Проверка лимитов API")
    print("=" * 60)

    cmc = CoinMarketCapClient()

    # Делаем запрос и смотрим заголовки ответа
    print("Проверка оставшихся кредитов...")

    # К сожалению, через requests мы не можем получить заголовки ответа
    # из нашей текущей реализации. Но можно проверить, работает ли API

    test_symbols = ['BTC', 'ETH', 'KOMA']
    for symbol in test_symbols:
        quote = cmc.get_latest_quote(symbol)
        if quote:
            print(f"✅ {symbol} - OK")
        else:
            print(f"❌ {symbol} - Failed")


if __name__ == "__main__":
    debug_cmc_api()
    test_direct_api_call()
    check_api_limits()