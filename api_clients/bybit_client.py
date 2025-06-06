"""Bybit API client module."""
from typing import Dict, Any, Optional, List
from .base import BaseAPIClient
from config.settings import settings
from utils.logger import setup_logger
from utils.helpers import safe_float_conversion


logger = setup_logger(__name__)


class BybitClient(BaseAPIClient):
    """Bybit API client."""

    def __init__(self):
        """Initialize Bybit client."""
        super().__init__(settings.BYBIT_API_BASE_URL)

    def _process_api_response(self, response: Optional[Dict]) -> Optional[Any]:
        """Process Bybit API response format."""
        if not response:
            return None

        if response.get("retCode") == 0:
            return response.get("result")

        logger.error(
            f"Bybit API error: {response.get('retMsg')} (Code: {response.get('retCode')})"
        )
        return None

    def get_historical_oi(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        category: str = "linear"
    ) -> Optional[List[Dict]]:
        """Get historical open interest data."""
        params = {
            "category": category,
            "symbol": symbol,
            "intervalTime": "1d",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 31
        }
        response = self._make_request("/v5/market/open-interest", params)
        return self._process_api_response(response)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        category: str = "linear",
        limit: int = 31
    ) -> Optional[List[List]]:
        """Get kline data."""
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": start_time,
            "end": end_time,
            "limit": limit
        }
        response = self._make_request("/v5/market/kline", params)
        result = self._process_api_response(response)

        if result and 'list' in result:
            return result['list']
        return None

    def get_ticker(self, symbol: str, category: str = "linear") -> Optional[Dict]:
        """Get ticker data."""
        params = {
            "category": category,
            "symbol": symbol
        }
        response = self._make_request("/v5/market/tickers", params)
        result = self._process_api_response(response)

        if result and 'list' in result and result['list']:
            return result['list'][0]
        return None

    def get_historical_oi_with_prices(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> Optional[List[Dict[str, float]]]:
        """Get historical OI data with USDT values calculated."""
        # Get OI data
        oi_data = self.get_historical_oi(symbol, start_time, end_time)
        if not oi_data or 'list' not in oi_data:
            return None

        # Get price data
        klines = self.get_klines(symbol, "D", start_time, end_time)
        if not klines:
            return None

        # Create price lookup
        prices_by_timestamp = {}
        for kline in klines:
            try:
                timestamp = int(kline[0])
                close_price = float(kline[4])
                prices_by_timestamp[timestamp] = close_price
            except (IndexError, ValueError, TypeError):
                continue

        # Combine OI with prices
        result = []
        for oi_record in oi_data['list']:
            try:
                timestamp = int(oi_record['timestamp'])
                oi_base = float(oi_record['openInterest'])

                if timestamp in prices_by_timestamp:
                    price = prices_by_timestamp[timestamp]
                    oi_usdt = oi_base * price
                    result.append({
                        'timestamp': timestamp,
                        'value_usdt': oi_usdt
                    })
            except (KeyError, ValueError, TypeError):
                continue

        return result[-30:] if len(result) > 30 else result

    def get_current_oi_usdt(self, symbol: str) -> Optional[float]:
        """Get current OI in USDT."""
        ticker = self.get_ticker(symbol)
        if not ticker:
            return None

        try:
            return float(ticker.get('openInterestValue', 0))
        except (ValueError, TypeError):
            return None

    def get_spot_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 31
    ) -> Optional[List[List]]:
        """Get spot market klines."""
        return self.get_klines(
            symbol, interval, start_time, end_time,
            category="spot", limit=limit
        )