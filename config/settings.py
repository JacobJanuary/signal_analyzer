"""Configuration management module."""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings management."""

    # Database settings
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', '3306'))
    DB_USER: str = os.getenv('DB_USER', 'root')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', '')
    DB_NAME: str = os.getenv('DB_NAME', 'crypto_db')

    # API settings
    BINANCE_API_BASE_URL: str = os.getenv('BINANCE_API_BASE_URL', 'https://fapi.binance.com')
    BINANCE_SPOT_API_BASE_URL: str = os.getenv('BINANCE_SPOT_API_BASE_URL', 'https://api.binance.com')
    BYBIT_API_BASE_URL: str = os.getenv('BYBIT_API_BASE_URL', 'https://api.bybit.com')
    COINMARKETCAP_API_KEY: str = os.getenv('COINMARKETCAP_API_KEY', '')
    COINMARKETCAP_API_URL: str = os.getenv('COINMARKETCAP_API_URL', 'https://pro-api.coinmarketcap.com')

    # Logging settings
    LOG_FILE: str = os.getenv('LOG_FILE', 'crypto_signals_enrichment.log')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

    # Processing settings
    RETRY_ATTEMPTS: int = int(os.getenv('RETRY_ATTEMPTS', '3'))
    RETRY_DELAY: int = int(os.getenv('RETRY_DELAY', '1'))

    @classmethod
    def get_db_config(cls) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD,
            'database': cls.DB_NAME,
            'charset': 'utf8mb4'
        }


settings = Settings()