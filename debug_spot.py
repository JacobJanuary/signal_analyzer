#!/usr/bin/env python3
"""Debug script to test spot market processing."""
from modules.spot_usdt_processor import SpotUSDTProcessor
from database.models import signal_repository
import json


def test_spot_processing(symbol: str = None):
    """Test spot processing for a symbol."""
    if symbol:
        print(f"\n🔍 Тестирование обработки спотовых данных для {symbol}")
        print("=" * 50)

        processor = SpotUSDTProcessor()
        result = processor.process_symbol(symbol)

        print(f"\n📊 Результаты:")
        print(f"- Ошибка: {result.error}")
        print(f"- Биржа: {result.source_exchange.value if result.source_exchange else 'None'}")

        if not result.error:
            print(f"\n📈 Объемы:")
            print(
                f"- Средний объем (30 дней): ${result.avg_volume_usdt:,.2f}" if result.avg_volume_usdt else "- Средний объем: None")
            print(
                f"- Текущий объем: ${result.current_volume_usdt:,.2f}" if result.current_volume_usdt else "- Текущий объем: None")
            print(
                f"- Изменение объема: {result.volume_change_percentage:.2f}%" if result.volume_change_percentage is not None else "- Изменение объема: None")

            print(f"\n💰 Цены:")
            print(
                f"- Средняя цена (30 дней): ${result.avg_price_usdt:,.2f}" if result.avg_price_usdt else "- Средняя цена: None")
            print(
                f"- Текущая цена: ${result.current_price_usdt:,.2f}" if result.current_price_usdt else "- Текущая цена: None")
            print(
                f"- Изменение цены: {result.price_change_percentage:.2f}%" if result.price_change_percentage is not None else "- Изменение цены: None")
    else:
        # Получаем символы из базы данных
        print("\n📋 Получение символов из базы данных...")
        try:
            signals = signal_repository.get_new_signals(last_processed_id=0)
            if signals:
                print(f"Найдено {len(signals)} сигналов")
                for signal in signals[:5]:  # Проверяем первые 5
                    print(f"\n--- Проверка символа: {signal.symbol} ---")
                    test_spot_processing(signal.symbol)
            else:
                print("Нет новых сигналов в базе данных")
        except Exception as e:
            print(f"❌ Ошибка при получении данных из БД: {e}")


def check_symbol_availability(symbol: str):
    """Check if symbol is available on spot markets."""
    print(f"\n🔍 Проверка доступности {symbol} на спотовых рынках")
    print("=" * 50)

    from api_clients.binance_client import BinanceClient
    from api_clients.bybit_client import BybitClient

    # Проверка Binance
    print("\n📊 Binance Spot:")
    binance = BinanceClient()
    trading_symbol = f"{symbol}USDT"

    # Пробуем получить одну свечу
    klines = binance.get_spot_klines(trading_symbol, "1d", limit=1)
    if klines:
        print(f"✅ {trading_symbol} доступен на Binance Spot")
        print(f"   Последняя цена: ${float(klines[0][4]):,.2f}")
    else:
        print(f"❌ {trading_symbol} НЕ доступен на Binance Spot")

    # Проверка Bybit
    print("\n📊 Bybit Spot:")
    bybit = BybitClient()

    klines = bybit.get_spot_klines(trading_symbol, "D",
                                   start_time=1704067200000,
                                   end_time=1704153600000,
                                   limit=1)
    if klines:
        print(f"✅ {trading_symbol} доступен на Bybit Spot")
        print(f"   Последняя цена: ${float(klines[0][4]):,.2f}")
    else:
        print(f"❌ {trading_symbol} НЕ доступен на Bybit Spot")


def list_common_spot_symbols():
    """List common spot trading symbols."""
    print("\n📋 Примеры популярных спотовых символов:")
    print("=" * 50)
    common_symbols = [
        "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "DOGE",
        "AVAX", "MATIC", "LINK", "UNI", "ATOM", "LTC", "ETC"
    ]

    print("Символы для тестирования:")
    for symbol in common_symbols:
        print(f"  - {symbol}")

    return common_symbols


if __name__ == "__main__":
    print("🔧 Отладка обработки спотовых данных")
    print("=" * 50)

    choice = input(
        "\nВыберите действие:\n1. Проверить конкретный символ\n2. Проверить символы из БД\n3. Проверить доступность символа\n4. Показать популярные символы\n\nВаш выбор (1-4): ")

    if choice == "1":
        symbol = input("Введите символ (например, BTC): ").strip().upper()
        if symbol:
            test_spot_processing(symbol)
    elif choice == "2":
        test_spot_processing()
    elif choice == "3":
        symbol = input("Введите символ для проверки доступности: ").strip().upper()
        if symbol:
            check_symbol_availability(symbol)
    elif choice == "4":
        symbols = list_common_spot_symbols()
        if input("\nПротестировать один из них? (y/n): ").lower() == 'y':
            symbol = input("Введите символ: ").strip().upper()
            if symbol:
                test_spot_processing(symbol)