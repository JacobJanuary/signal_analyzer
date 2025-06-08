"""Binance API client module."""
from typing import Dict, Any, Optional, List
from .base import BaseAPIClient
from config.settings import settings
from utils.logger import setup_logger


logger = setup_logger(__name__)


class BinanceClient(BaseAPIClient):
    """Binance API client."""

    def __init__(self):
        """Initialize Binance client."""
        super().__init__(settings.BINANCE_API_BASE_URL)
        self.spot_base_url = settings.BINANCE_SPOT_API_BASE_URL

    def get_historical_oi(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Get historical open interest data."""
        params = {
            "symbol": symbol,
            "period": "1d",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 30
        }
        return self._make_request("/futures/data/openInterestHist", params)

    def get_current_oi(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current open interest."""
        params = {"symbol": symbol}
        return self._make_request("/fapi/v1/openInterest", params)

    def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price."""
        params = {"symbol": symbol}
        return self._make_request("/fapi/v1/ticker/price", params)

    def get_spot_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Optional[List[List[Any]]]:
        """Get spot market klines."""
        # Temporarily change base URL for spot request
        original_base_url = self.base_url
        self.base_url = self.spot_base_url

        params = {"symbol": symbol, "interval": interval}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if limit:
            params["limit"] = limit

        result = self._make_request("/api/v3/klines", params)

        # Restore original base URL
        self.base_url = original_base_url

        return result

    def get_futures_klines(
            self,
            symbol: str,
            interval: str,
            start_time: Optional[int] = None,
            end_time: Optional[int] = None,
            limit: Optional[int] = None
    ) -> Optional[List[List[Any]]]:
        """Get futures market klines."""
        params = {"symbol": symbol, "interval": interval}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if limit:
            params["limit"] = limit

        # Используем эндпоинт фьючерсов
        return self._make_request("/fapi/v1/klines", params)