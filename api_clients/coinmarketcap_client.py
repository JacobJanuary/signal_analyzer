"""CoinMarketCap API client module."""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base import BaseAPIClient
from config.settings import settings
from utils.logger import setup_logger, log_with_context
from utils.helpers import safe_float_conversion

logger = setup_logger(__name__)


class CoinMarketCapClient(BaseAPIClient):
    """CoinMarketCap API client."""

    def __init__(self):
        """Initialize CoinMarketCap client."""
        super().__init__(settings.COINMARKETCAP_API_URL)
        self.api_key = settings.COINMARKETCAP_API_KEY

        if not self.api_key:
            logger.warning("CoinMarketCap API key not configured")

        # Set headers with API key
        self.session.headers.update({
            'X-CMC_PRO_API_KEY': self.api_key,
            'Accept': 'application/json'
        })

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def get_symbol_map(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get CMC ID and slug for a symbol."""
        if not self.is_configured():
            return None

        params = {
            'symbol': symbol.upper(),
            'limit': 1
        }

        try:
            response = self._make_request('/v1/cryptocurrency/map', params)
            if response and 'data' in response and response['data']:
                return response['data'][0]
            return None
        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error getting symbol map from CMC",
                symbol=symbol,
                error=str(e)
            )
            return None

    def get_historical_quotes(
        self,
        symbol: str,
        time_start: datetime,
        time_end: datetime,
        interval: str = '24h',
        count: int = 30
    ) -> Optional[List[Dict[str, Any]]]:
        """Get historical quotes using v3 API."""
        if not self.is_configured():
            return None

        # Get CMC ID for symbol
        symbol_info = self.get_symbol_map(symbol)
        if not symbol_info:
            log_with_context(
                logger, 'warning',
                "Symbol not found on CoinMarketCap",
                symbol=symbol
            )
            return None

        cmc_id = symbol_info['id']

        params = {
            'id': cmc_id,
            'time_start': int(time_start.timestamp()),
            'time_end': int(time_end.timestamp()),
            'interval': interval,
            'count': count,
            'convert': 'USD'
        }

        try:
            response = self._make_request('/v3/cryptocurrency/quotes/historical', params)
            if response and 'data' in response and 'quotes' in response['data']:
                return response['data']['quotes']
            return None
        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error getting historical quotes from CMC",
                symbol=symbol,
                error=str(e)
            )
            return None

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for a symbol."""
        if not self.is_configured():
            return None

        params = {
            'symbol': symbol.upper(),
            'convert': 'USD'
        }

        try:
            response = self._make_request('/v1/cryptocurrency/quotes/latest', params)
            if response and 'data' in response and symbol.upper() in response['data']:
                return response['data'][symbol.upper()]
            return None
        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error getting latest quote from CMC",
                symbol=symbol,
                error=str(e)
            )
            return None

    def get_market_data_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 30-day market data summary for a symbol."""
        if not self.is_configured():
            return None

        try:
            # Get historical data for last 30 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)

            historical_data = self.get_historical_quotes(
                symbol, start_time, end_time, interval='24h', count=30
            )

            if not historical_data:
                return None

            # Get latest quote
            latest_quote = self.get_latest_quote(symbol)
            if not latest_quote:
                return None

            # Process historical data
            volumes_usd = []
            prices = []

            for quote in historical_data:
                try:
                    if 'quote' in quote and 'USD' in quote['quote']:
                        usd_data = quote['quote']['USD']

                        volume = safe_float_conversion(usd_data.get('volume_24h'))
                        price = safe_float_conversion(usd_data.get('price'))

                        if volume > 0:
                            volumes_usd.append(volume)
                        if price > 0:
                            prices.append(price)
                except Exception as e:
                    log_with_context(
                        logger, 'warning',
                        "Error processing CMC quote",
                        error=str(e)
                    )

            if not volumes_usd or not prices:
                return None

            # Calculate averages
            avg_volume = sum(volumes_usd) / len(volumes_usd)
            avg_price = sum(prices) / len(prices)

            # Get current data from latest quote
            current_quote = latest_quote.get('quote', {}).get('USD', {})
            current_price = safe_float_conversion(current_quote.get('price', 0))
            current_volume = safe_float_conversion(current_quote.get('volume_24h', 0))

            # Get yesterday's volume (last item in historical data)
            yesterday_volume = volumes_usd[-1] if volumes_usd else current_volume

            return {
                'avg_volume_usd': avg_volume,
                'current_volume_usd': current_volume,
                'yesterday_volume_usd': yesterday_volume,
                'avg_price_usd': avg_price,
                'current_price_usd': current_price,
                'data_points': len(prices),
                'source': 'coinmarketcap'
            }

        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error getting market data from CMC",
                symbol=symbol,
                error=str(e),
                exc_info=True
            )
            return None