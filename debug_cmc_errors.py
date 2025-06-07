#!/usr/bin/env python3
"""Debug CoinMarketCap API errors."""
import requests
from config.settings import settings
from datetime import datetime, timedelta, timezone
import json


def test_direct_api_v2():
    """Test direct API v2 call like in the working example."""
    print("🔍 Прямой тест API v2 (как в рабочем примере)")
    print("=" * 60)

    API_KEY = settings.COINMARKETCAP_API_KEY
    if not API_KEY:
        print("❌ API ключ не найден")
        return

    BASE_URL_V2 = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/"
    HEADERS = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    symbol = "KOMA"
    time_end = datetime.now(timezone.utc)
    time_start = time_end - timedelta(days=31)

    url = f"{BASE_URL_V2}quotes/historical"
    parameters = {
        "symbol": symbol.upper(),
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "interval": "daily",
        "convert": "USD",
    }

    print(f"URL: {url}")
    print(f"Parameters: {json.dumps(parameters, indent=2)}")

    try:
        response = requests.get(url, headers=HEADERS, params=parameters)
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            list_of_tokens = data.get("data", {}).get(symbol.upper(), [])

            print(f"✅ Успешный ответ!")
            print(f"Найдено токенов: {len(list_of_tokens)}")

            for i, token_data in enumerate(list_of_tokens):
                print(f"\nТокен {i + 1}:")
                print(f"  ID: {token_data.get('id')}")
                print(f"  Name: {token_data.get('name')}")
                print(f"  Is Active: {token_data.get('is_active')}")
                print(f"  Quotes count: {len(token_data.get('quotes', []))}")

        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"Response: {response.text[:500]}")

    except Exception as e:
        print(f"❌ Exception: {e}")


def test_our_client():
    """Test our client implementation."""
    print("\n\n🔍 Тест нашего клиента")
    print("=" * 60)

    from api_clients.base import BaseAPIClient

    class TestCMCClient(BaseAPIClient):
        def __init__(self):
            super().__init__('https://pro-api.coinmarketcap.com')
            self.session.headers.update({
                'X-CMC_PRO_API_KEY': settings.COINMARKETCAP_API_KEY,
                'Accept': 'application/json'
            })

        def test_v2_historical(self, symbol: str):
            time_end = datetime.now(timezone.utc)
            time_start = time_end - timedelta(days=31)

            params = {
                'symbol': symbol.upper(),
                'time_start': time_start.isoformat(),
                'time_end': time_end.isoformat(),
                'interval': 'daily',
                'convert': 'USD'
            }

            print(f"Making request to: /v2/cryptocurrency/quotes/historical")
            print(f"Base URL: {self.base_url}")
            print(f"Params: {json.dumps(params, indent=2)}")

            response = self._make_request('/v2/cryptocurrency/quotes/historical', params)

            if response:
                print("✅ Получен ответ от API")
                if 'data' in response:
                    data = response['data'].get(symbol.upper(), [])
                    print(f"Найдено токенов: {len(data)}")
                    if data and len(data) > 0:
                        print(f"Первый токен: {data[0].get('name')} (ID: {data[0].get('id')})")
                        print(f"Количество котировок: {len(data[0].get('quotes', []))}")
                else:
                    print("❌ Нет 'data' в ответе")
                    print(f"Response keys: {list(response.keys())}")
            else:
                print("❌ Нет ответа от API")

    client = TestCMCClient()
    client.test_v2_historical('KOMA')


def compare_urls():
    """Compare URL construction."""
    print("\n\n🔍 Сравнение URL")
    print("=" * 60)

    # Рабочий пример
    working_url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"

    # Наш клиент
    from api_clients.coinmarketcap_client import CoinMarketCapClient
    cmc = CoinMarketCapClient()

    print(f"Рабочий URL: {working_url}")
    print(f"Base URL нашего клиента: {cmc.base_url}")

    # Проверяем, как формируется полный URL
    test_endpoint = '/v2/cryptocurrency/quotes/historical'
    full_url = f"{cmc.base_url}{test_endpoint}"

    print(f"Полный URL нашего клиента: {full_url}")
    print(f"URLs совпадают: {working_url == full_url}")


if __name__ == "__main__":
    # 1. Тест прямого API как в рабочем примере
    test_direct_api_v2()

    # 2. Тест нашего клиента
    test_our_client()

    # 3. Сравнение URLs
    compare_urls()