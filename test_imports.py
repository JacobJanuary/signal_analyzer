"""Test imports to verify project structure."""
import sys
import os

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    try:
        # Тестируем импорты
        from config.settings import settings
        print("✓ config.settings imported successfully")

        from utils.logger import setup_logger
        print("✓ utils.logger imported successfully")

        from utils.helpers import get_timestamp_ms
        print("✓ utils.helpers imported successfully")

        from api_clients.base import BaseAPIClient
        print("✓ api_clients.base imported successfully")

        from api_clients.binance_client import BinanceClient
        print("✓ api_clients.binance_client imported successfully")

        from api_clients.bybit_client import BybitClient
        print("✓ api_clients.bybit_client imported successfully")

        from database.connection import DatabaseConnection
        print("✓ database.connection imported successfully")

        from database.models import SignalRepository
        print("✓ database.models imported successfully")

        from modules.oi_processor import OIProcessor
        print("✓ modules.oi_processor imported successfully")

        print("\n✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


if __name__ == "__main__":
    test_imports()