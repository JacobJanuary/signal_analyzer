#!/usr/bin/env python3
"""Test MLN token availability."""
from api_clients.binance_client import BinanceClient
from modules.spot_usdt_processor import SpotUSDTProcessor
import json


def test_mln_binance():
    """Test MLN availability on Binance."""
    print("🔍 Тестирование MLN на Binance Spot")
    print("=" * 50)

    client = BinanceClient()
    symbol = "MLNUSDT"

    # Test 1: Get klines
    print("\n1. Получение исторических данных (klines):")
    klines = client.get_spot_klines(symbol, "1d", limit=1)
    if klines:
        print(f"✅ Klines получены успешно")
        print(f"   Последняя свеча: {json.dumps(klines[0][:8], indent=2)}")
    else:
        print(f"❌ Не удалось получить klines")

    # Test 2: Get 24h ticker
    print("\n2. Получение 24h тикера:")
    original_base_url = client.base_url
    client.base_url = client.spot_base_url

    ticker = client._make_request("/api/v3/ticker/24hr", {"symbol": symbol})

    client.base_url = original_base_url

    if ticker:
        print(f"✅ Тикер получен успешно")
        print(f"   Последняя цена: ${float(ticker.get('lastPrice', 0)):,.2f}")
        print(f"   Объем 24h (USDT): ${float(ticker.get('quoteVolume', 0)):,.2f}")
        print(f"   Изменение 24h: {ticker.get('priceChangePercent')}%")
    else:
        print(f"❌ Не удалось получить тикер")

    # Test 3: Test with processor
    print("\n3. Тестирование через SpotUSDTProcessor:")
    processor = SpotUSDTProcessor()
    result = processor.process_symbol('MLN')

    if not result.error:
        print(f"✅ Обработка успешна")
        print(f"   Источник: {result.source_exchange.value}")
        print(f"   Средний объем (30д): ${result.avg_volume_usdt:,.2f}")
        print(f"   Текущий объем: ${result.current_volume_usdt:,.2f}")
        print(f"   Вчерашний объем: ${result.yesterday_volume_usdt:,.2f}")
        print(f"   Изменение сегодня/вчера: {result.today_yesterday_volume_change_pct:.2f}%")
        print(f"   Средняя цена (30д): ${result.avg_price_usdt:,.2f}")
        print(f"   Текущая цена: ${result.current_price_usdt:,.2f}")
    else:
        print(f"❌ Ошибка обработки: {result.error}")


if __name__ == "__main__":
    test_mln_binance()