import os
import sys
import json
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

API_KEY = os.getenv("COINMARKETCAP_API_KEY")
if not API_KEY:
    print("Ошибка: API-ключ не найден. Убедитесь, что он задан в файле .env.")
    sys.exit(1)

BASE_URL_V1 = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/"
BASE_URL_V2 = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/"
HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": API_KEY,
}


def get_historical_data(symbol: str, days: int) -> dict | None:
    """
    Получает исторические данные, находя правильный активный токен из списка,
    который может вернуть API.
    """
    print(f"Запрос исторических данных для '{symbol.upper()}' за последние {days} дней...")

    time_end = datetime.now(timezone.utc)
    time_start = time_end - timedelta(days=days)

    url = f"{BASE_URL_V2}quotes/historical"
    parameters = {
        "symbol": symbol.upper(),
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "interval": "daily",
        "convert": "USD",
    }

    try:
        response = requests.get(url, headers=HEADERS, params=parameters)
        response.raise_for_status()
        data = response.json()

        list_of_tokens = data.get("data", {}).get(symbol.upper(), [])
        if not list_of_tokens:
            print(f"Ошибка: Не удалось найти данные для символа '{symbol}'.")
            return None

        for token_data in list_of_tokens:
            if token_data.get("is_active") == 1 and token_data.get("quotes"):
                print(f"Найден активный токен: {token_data.get('name')} (ID: {token_data.get('id')})")
                return token_data

        print(f"Ошибка: Не найдено активных токенов с историческими данными для '{symbol}'.")
        return None

    except requests.exceptions.HTTPError as e:
        error_message = e.response.json().get('status', {}).get('error_message', str(e))
        print(f"Ошибка HTTP при запросе к API: {error_message}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API: {e}")
        return None


def get_latest_quote(symbol: str) -> dict | None:
    """
    Получает самые последние котировки.
    """
    print(f"Запрос последних котировок для '{symbol.upper()}'...")
    url = f"{BASE_URL_V1}quotes/latest"
    parameters = {"symbol": symbol.upper(), "convert": "USD"}
    try:
        response = requests.get(url, headers=HEADERS, params=parameters)
        response.raise_for_status()
        data = response.json()
        if "data" not in data or not data["data"]:
            print(f"Ошибка: Не удалось найти последние котировки для символа '{symbol}'.")
            return None
        return data["data"][symbol.upper()]
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API: {e}")
        return None


def process_and_display_data(symbol: str):
    """
    Обрабатывает полученные данные и выводит статистику.
    """
    historical_data = get_historical_data(symbol, days=31)
    if not historical_data:
        print("Недостаточно исторических данных для анализа.")
        return

    latest_data = get_latest_quote(symbol)
    if not latest_data:
        print("Не удалось получить последние котировки.")
        return

    # --- Подготовка данных ---
    quotes = sorted(historical_data['quotes'], key=lambda x: x['timestamp'])

    today_utc = datetime.now(timezone.utc).date()
    historical_quotes = [q for q in quotes if
                         datetime.fromisoformat(q['timestamp'].replace('Z', '+00:00')).date() < today_utc]

    if len(historical_quotes) < 1:
        print("Отсутствуют полные исторические данные за предыдущие дни.")
        return

    if len(historical_quotes) < 30:
        print(f"Предупреждение: получено данных только за {len(historical_quotes)} полных дней.")

    # --- Расчеты ---
    last_30_days_quotes = historical_quotes[-30:]

    # ИСПРАВЛЕНИЕ: Используем правильные ключи 'volume_24h' и 'price' из лога
    volumes_30d = [q['quote']['USD'].get('volume_24h', 0) for q in last_30_days_quotes]
    prices_30d = [q['quote']['USD'].get('price', 0) for q in last_30_days_quotes]

    quote_yesterday_usd = last_30_days_quotes[-1]['quote']['USD']
    volume_yesterday = quote_yesterday_usd.get('volume_24h', 0)
    price_yesterday = quote_yesterday_usd.get('price', 0)

    latest_quote_usd = latest_data.get('quote', {}).get('USD', {})
    volume_today_24h = latest_quote_usd.get('volume_24h', 0)
    current_price = latest_quote_usd.get('price', 0)

    avg_volume_30d = sum(volumes_30d) / len(volumes_30d) if volumes_30d else 0
    min_price_30d = min(prices_30d) if prices_30d else 0
    max_price_30d = max(prices_30d) if prices_30d else 0

    # --- Вывод результатов ---
    print("\n" + "=" * 40)
    print(f"Статистика для токена: {latest_data.get('name')} ({symbol.upper()})")
    print(f"Данные на: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    print("\n--- Объемы торгов (USD) ---")
    print(f"Объем за сегодня (скользящие 24ч): ${volume_today_24h:,.2f}")
    print(f"Объем за вчера (полный день):      ${volume_yesterday:,.2f}")
    print(f"Средний объем за 30 дней:          ${avg_volume_30d:,.2f}")

    print("\n--- Изменение объема (в сравнении с 24ч) ---")
    if volume_yesterday > 0:
        diff_vs_yesterday = ((volume_today_24h - volume_yesterday) / volume_yesterday) * 100
        print(f"Относительно вчерашнего дня: {diff_vs_yesterday:+.2f}%")
    else:
        print("Относительно вчерашнего дня: Н/Д (вчера не было объема)")

    if avg_volume_30d > 0:
        diff_vs_avg = ((volume_today_24h - avg_volume_30d) / avg_volume_30d) * 100
        print(f"Относительно среднего за 30 дней: {diff_vs_avg:+.2f}%")
    else:
        print("Относительно среднего за 30 дней: Н/Д (нет данных об объеме)")

    print("\n--- Цена (USD) ---")
    print(f"Текущая цена:            ${current_price:,.4f}")
    print(f"Цена закрытия вчера:     ${price_yesterday:,.4f}")
    print(f"Мин. цена за 30 дней:    ${min_price_30d:,.4f}")
    print(f"Макс. цена за 30 дней:   ${max_price_30d:,.4f}")

    print("\n--- Динамика цены (%) ---")
    print(f"За 1 час:    {latest_quote_usd.get('percent_change_1h', 0):+.2f}%")
    print(f"За 24 часа:  {latest_quote_usd.get('percent_change_24h', 0):+.2f}%")
    print(f"За 7 дней:   {latest_quote_usd.get('percent_change_7d', 0):+.2f}%")
    print(f"За 30 дней:  {latest_quote_usd.get('percent_change_30d', 0):+.2f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    token_symbol = input("Введите символ токена (например, BTC, ETH): ").strip()
    if not token_symbol:
        print("Символ токена не может быть пустым.")
    else:
        process_and_display_data(token_symbol)