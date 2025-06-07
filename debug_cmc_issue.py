#!/usr/bin/env python3
"""Debug exact issue with CoinMarketCap data processing."""
from api_clients.coinmarketcap_client import CoinMarketCapClient
from datetime import datetime, timedelta, timezone
import traceback


def test_step_by_step():
    """Test CoinMarketCap step by step."""
    print("🔍 Пошаговая отладка CoinMarketCap")
    print("=" * 60)

    cmc = CoinMarketCapClient()
    symbol = 'KOMA'

    print(f"Тестируем: {symbol}")
    print("-" * 40)

    # Шаг 1: Проверка конфигурации
    print("\n1️⃣ Проверка конфигурации:")
    if cmc.is_configured():
        print("✅ API ключ настроен")
    else:
        print("❌ API ключ не настроен")
        return

    # Шаг 2: Получение token_id
    print("\n2️⃣ Получение token_id из БД:")
    token_id = cmc.get_token_id_from_db(symbol)
    if token_id:
        print(f"✅ Token ID: {token_id}")
    else:
        print("❌ Token ID не найден в БД")

    # Шаг 3: Получение последней котировки
    print("\n3️⃣ Получение последней котировки:")
    try:
        latest = cmc.get_latest_quote(symbol)
        if latest:
            print("✅ Котировка получена")
            if 'quote' in latest and 'USD' in latest['quote']:
                usd = latest['quote']['USD']
                print(f"   Цена: ${usd.get('price', 0):.8f}")
                print(f"   Объем: ${usd.get('volume_24h', 0):,.2f}")
        else:
            print("❌ Котировка не получена")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()

    # Шаг 4: Получение исторических данных
    print("\n4️⃣ Получение исторических данных:")
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=31)

    try:
        # Сначала тестируем get_historical_quotes_v2 напрямую
        print("   Тест get_historical_quotes_v2:")
        historical_v2 = cmc.get_historical_quotes_v2(
            symbol, start_time, end_time, 'daily', token_id
        )
        if historical_v2:
            print(f"   ✅ Получено записей: {len(historical_v2)}")
        else:
            print("   ❌ Данные не получены")

        # Затем тестируем обертку get_historical_quotes
        print("\n   Тест get_historical_quotes:")
        historical = cmc.get_historical_quotes(
            symbol, start_time, end_time, 'daily', 31
        )
        if historical:
            print(f"   ✅ Получено записей: {len(historical)}")
        else:
            print("   ❌ Данные не получены")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()

    # Шаг 5: Получение market_data
    print("\n5️⃣ Получение market_data:")
    try:
        market_data = cmc.get_market_data_for_symbol(symbol)
        if market_data:
            print("✅ Market data получены:")
            for key, value in market_data.items():
                if isinstance(value, float):
                    print(f"   {key}: ${value:,.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ Market data не получены")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()


def test_spot_processor_debug():
    """Debug spot processor."""
    print("\n\n🔍 Отладка SpotUSDTProcessor")
    print("=" * 60)

    from modules.spot_usdt_processor import SpotUSDTProcessor

    processor = SpotUSDTProcessor()
    symbol = 'KOMA'

    print(f"Обработка {symbol}:")
    print("-" * 40)

    # Тестируем метод _process_coinmarketcap напрямую
    print("\nТест _process_coinmarketcap напрямую:")
    try:
        result = processor._process_coinmarketcap(symbol)
        if result.error:
            print(f"❌ Ошибка: {result.error}")
        else:
            print(f"✅ Успешно!")
            print(f"   Источник: {result.source_exchange.value}")
            print(
                f"   Текущий объем: ${result.current_volume_usdt:,.2f}" if result.current_volume_usdt else "   Текущий объем: None")
            print(
                f"   Текущая цена: ${result.current_price_usdt:.8f}" if result.current_price_usdt else "   Текущая цена: None")
    except Exception as e:
        print(f"❌ Исключение: {e}")
        traceback.print_exc()


def test_direct_v2_with_our_client():
    """Test v2 API directly with our base client."""
    print("\n\n🔍 Прямой тест v2 API через наш базовый клиент")
    print("=" * 60)

    from api_clients.base import BaseAPIClient
    from config.settings import settings

    # Создаем клиент с правильным базовым URL
    client = BaseAPIClient('https://pro-api.coinmarketcap.com')
    client.session.headers.update({
        'X-CMC_PRO_API_KEY': settings.COINMARKETCAP_API_KEY,
        'Accept': 'application/json'
    })

    symbol = 'KOMA'
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=31)

    params = {
        'symbol': symbol.upper(),
        'time_start': start_time.isoformat(),
        'time_end': end_time.isoformat(),
        'interval': 'daily',
        'convert': 'USD'
    }

    print(f"Запрос к: /v2/cryptocurrency/quotes/historical")
    print(f"Параметры: {params}")

    response = client._make_request('/v2/cryptocurrency/quotes/historical', params)

    if response:
        print("✅ Ответ получен")
        if 'data' in response:
            tokens = response['data'].get(symbol.upper(), [])
            print(f"Найдено токенов: {len(tokens)}")
            if tokens and len(tokens) > 0:
                token = tokens[0]
                print(f"Токен: {token.get('name')} (ID: {token.get('id')})")
                print(f"Котировок: {len(token.get('quotes', []))}")
        else:
            print("❌ Нет 'data' в ответе")
    else:
        print("❌ Ответ не получен")


if __name__ == "__main__":
    # 1. Пошаговая отладка
    test_step_by_step()

    # 2. Отладка процессора
    test_spot_processor_debug()

    # 3. Прямой тест через базовый клиент
    test_direct_v2_with_our_client()