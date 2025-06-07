#!/usr/bin/env python3
"""Debug CoinMarketCap integration."""
from modules.spot_usdt_processor import SpotUSDTProcessor
from api_clients.coinmarketcap_client import CoinMarketCapClient
import json


def test_cmc_for_symbol(symbol: str):
    """Test CoinMarketCap for a specific symbol."""
    print(f"\n🔍 Тестирование CoinMarketCap для {symbol}")
    print("=" * 60)

    # Test CMC client directly
    cmc = CoinMarketCapClient()

    print(f"\n1. Проверка конфигурации API:")
    print(f"   API ключ настроен: {'✅ Да' if cmc.is_configured() else '❌ Нет'}")

    if not cmc.is_configured():
        print("\n❌ CoinMarketCap API ключ не настроен в .env файле")
        return

    print(f"\n2. Получение ID символа:")
    symbol_info = cmc.get_symbol_map(symbol)
    if symbol_info:
        print(f"   ✅ Найден: {symbol_info['name']} (ID: {symbol_info['id']})")
    else:
        print(f"   ❌ Символ {symbol} не найден на CoinMarketCap")
        return

    print(f"\n3. Получение рыночных данных:")
    market_data = cmc.get_market_data_for_symbol(symbol)
    if market_data:
        print(f"   ✅ Данные получены:")
        print(f"   - Средний объем (30д): ${market_data['avg_volume_usd']:,.2f}")
        print(f"   - Текущий объем: ${market_data['current_volume_usd']:,.2f}")
        print(f"   - Вчерашний объем: ${market_data.get('yesterday_volume_usd', 0):,.2f}")
        print(f"   - Средняя цена (30д): ${market_data['avg_price_usd']:,.2f}")
        print(f"   - Текущая цена: ${market_data['current_price_usd']:,.2f}")
        print(f"   - Количество точек данных: {market_data['data_points']}")
    else:
        print(f"   ❌ Не удалось получить рыночные данные")

    print(f"\n4. Тестирование через SpotUSDTProcessor:")
    processor = SpotUSDTProcessor()
    result = processor.process_symbol(symbol)

    if result.error:
        print(f"   ❌ Ошибка: {result.error}")
    else:
        print(f"   ✅ Обработка успешна!")
        print(f"   - Источник: {result.source_exchange.value}")
        print(
            f"   - Объем (средний): ${result.avg_volume_usdt:,.2f}" if result.avg_volume_usdt else "   - Объем (средний): None")
        print(
            f"   - Объем (текущий): ${result.current_volume_usdt:,.2f}" if result.current_volume_usdt else "   - Объем (текущий): None")
        print(
            f"   - Цена (средняя): ${result.avg_price_usdt:,.2f}" if result.avg_price_usdt else "   - Цена (средняя): None")
        print(
            f"   - Цена (текущая): ${result.current_price_usdt:,.2f}" if result.current_price_usdt else "   - Цена (текущая): None")


def test_problematic_symbols():
    """Test symbols that are failing."""
    symbols = ['KOMA', 'MUBARAK']

    for symbol in symbols:
        test_cmc_for_symbol(symbol)
        print("\n" + "-" * 80)


def test_latest_quote(symbol: str):
    """Test getting latest quote from CMC."""
    print(f"\n🔍 Тестирование latest quote для {symbol}")

    cmc = CoinMarketCapClient()
    if not cmc.is_configured():
        print("❌ API ключ не настроен")
        return

    quote = cmc.get_latest_quote(symbol)
    if quote:
        print(f"✅ Quote получен:")
        print(json.dumps(quote, indent=2))
    else:
        print(f"❌ Не удалось получить quote")


if __name__ == "__main__":
    print("🧪 Отладка интеграции с CoinMarketCap")
    print("=" * 80)

    # Test problematic symbols
    test_problematic_symbols()

    # Test a known symbol
    print("\n\n📊 Тест с известным символом (BTC):")
    test_cmc_for_symbol('BTC')

    # Test latest quote
    print("\n\n📊 Тест latest quote:")
    test_latest_quote('KOMA')