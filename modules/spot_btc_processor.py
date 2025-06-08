# Файл: modules/spot_btc_processor.py
# --- ИСПРАВЛЕННАЯ ВЕРСИЯ ---

"""Module: Spot BTC volume processing."""
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from api_clients.binance_client import BinanceClient
from api_clients.bybit_client import BybitClient
from utils.logger import setup_logger, log_with_context
from utils.helpers import (
    get_timestamp_ms,
    safe_float_conversion,
    calculate_percentage_change
)

logger = setup_logger(__name__)


class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    BYBIT = "bybit"


@dataclass
class SpotBTCResult:
    """Spot BTC market calculation result."""
    avg_volume_btc: Optional[float] = None
    current_volume_btc: Optional[float] = None
    yesterday_volume_btc: Optional[float] = None
    volume_change_current_to_yesterday: Optional[float] = None
    volume_change_current_to_average: Optional[float] = None
    source_exchange: Optional[Exchange] = None
    error: Optional[str] = None


class SpotBTCProcessor:
    """Process Spot BTC market data for signals."""

    def __init__(self):
        """Initialize Spot BTC processor."""
        self.binance_client = BinanceClient()
        self.bybit_client = BybitClient()

    def process_symbol(self, symbol: str) -> SpotBTCResult:
        """
        Process spot BTC data for a symbol.

        Args:
            symbol: Token symbol (e.g., 'ETH')

        Returns:
            SpotBTCResult with calculations or error
        """
        trading_symbol = f"{symbol}BTC"

        log_with_context(
            logger, 'info',
            "Starting spot BTC processing",
            symbol=symbol,
            trading_symbol=trading_symbol
        )

        # Try Binance first
        result = self._process_binance_spot_btc(trading_symbol)
        if not result.error:
            return result

        log_with_context(
            logger, 'warning',
            "Binance spot BTC processing failed, trying Bybit",
            symbol=symbol,
            trading_symbol=trading_symbol,
            binance_error=result.error
        )

        # Try Bybit if Binance fails
        result = self._process_bybit_spot_btc(trading_symbol)
        if not result.error:
            return result

        log_with_context(
            logger, 'error',
            "Spot BTC processing failed on all sources",
            symbol=symbol,
            trading_symbol=trading_symbol,
            final_error=result.error
        )

        return result

    def _process_binance_spot_btc(self, trading_symbol: str) -> SpotBTCResult:
        """Process spot BTC data from Binance."""
        try:
            log_with_context(logger, 'debug', "Starting Binance spot BTC processing", symbol=trading_symbol)

            utc_now = datetime.now(timezone.utc)
            end_time = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(days=31)

            klines = self.binance_client.get_spot_klines(
                trading_symbol,
                interval="1d",
                start_time=get_timestamp_ms(start_time),
                end_time=get_timestamp_ms(end_time),
                limit=31
            )

            if not klines:
                return SpotBTCResult(error=f"Failed to fetch Binance spot klines for {trading_symbol}")

            volumes_btc = []
            # Skip today's incomplete data if it's the last kline
            processed_klines = klines[:-1] if len(klines) > 30 else klines

            for kline in processed_klines:
                try:
                    # ---> ИЗМЕНЕНИЕ: Используем индекс 7 (quote_asset_volume) вместо 5 <---
                    btc_volume = safe_float_conversion(kline[7])
                    if btc_volume > 0:
                        volumes_btc.append(btc_volume)
                except (IndexError, ValueError) as e:
                    log_with_context(logger, 'warning', "Error processing Binance kline", error=str(e))

            if not volumes_btc:
                return SpotBTCResult(error=f"No valid spot BTC volume data from Binance for {trading_symbol}")

            avg_volume = sum(volumes_btc) / len(volumes_btc)

            # Get current data from 24h ticker
            current_ticker = self._get_binance_24h_ticker(trading_symbol)
            # ---> ИЗМЕНЕНИЕ: Используем 'quoteVolume' для получения объема в BTC <---
            current_volume = safe_float_conversion(current_ticker.get('quoteVolume', 0)) if current_ticker else \
            volumes_btc[-1]

            yesterday_volume = volumes_btc[-1] if len(volumes_btc) >= 1 else avg_volume

            volume_change_to_avg = calculate_percentage_change(current_volume, avg_volume)
            volume_change_to_yesterday = calculate_percentage_change(current_volume, yesterday_volume)

            log_with_context(logger, 'info', "Binance spot BTC processing successful", symbol=trading_symbol)

            return SpotBTCResult(
                avg_volume_btc=avg_volume,
                current_volume_btc=current_volume,
                yesterday_volume_btc=yesterday_volume,
                volume_change_current_to_yesterday=volume_change_to_yesterday,
                volume_change_current_to_average=volume_change_to_avg,
                source_exchange=Exchange.BINANCE
            )
        except Exception as e:
            log_with_context(logger, 'error', "Error in Binance spot BTC processing", symbol=trading_symbol,
                             error=str(e), exc_info=True)
            return SpotBTCResult(error=f"Binance spot BTC processing error for {trading_symbol}: {str(e)}")

    def _process_bybit_spot_btc(self, trading_symbol: str) -> SpotBTCResult:
        """Process spot BTC data from Bybit."""
        try:
            utc_now = datetime.now(timezone.utc)
            end_time = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(days=31)

            klines = self.bybit_client.get_spot_klines(
                trading_symbol,
                interval="D",
                start_time=get_timestamp_ms(start_time),
                end_time=get_timestamp_ms(end_time),
                limit=31
            )

            if not klines:
                return SpotBTCResult(error=f"Failed to fetch Bybit spot klines for {trading_symbol}")

            volumes_btc = []
            # Skip today's incomplete data
            processed_klines = klines[:-1] if len(klines) > 30 else klines

            for kline in processed_klines:
                try:
                    # ---> ИЗМЕНЕНИЕ: Используем индекс 6 (turnover) вместо 5 <---
                    btc_volume = safe_float_conversion(kline[6])
                    if btc_volume > 0:
                        volumes_btc.append(btc_volume)
                except (IndexError, ValueError) as e:
                    log_with_context(logger, 'warning', "Error processing Bybit kline", error=str(e))

            if not volumes_btc:
                return SpotBTCResult(error="No valid spot BTC volume data from Bybit")

            avg_volume = sum(volumes_btc) / len(volumes_btc)
            # For Bybit, daily kline represents the last 24h, so the last element is our "current" volume
            current_volume = volumes_btc[-1] if volumes_btc else 0
            yesterday_volume = volumes_btc[-1] if len(volumes_btc) >= 1 else avg_volume

            volume_change_to_avg = calculate_percentage_change(current_volume, avg_volume)
            volume_change_to_yesterday = calculate_percentage_change(current_volume, yesterday_volume)

            log_with_context(logger, 'info', "Bybit spot BTC processing successful", symbol=trading_symbol)

            return SpotBTCResult(
                avg_volume_btc=avg_volume,
                current_volume_btc=current_volume,
                yesterday_volume_btc=yesterday_volume,
                volume_change_current_to_yesterday=volume_change_to_yesterday,
                volume_change_current_to_average=volume_change_to_avg,
                source_exchange=Exchange.BYBIT
            )
        except Exception as e:
            log_with_context(logger, 'error', "Error in Bybit spot processing", error=str(e), exc_info=True)
            return SpotBTCResult(error=f"Bybit spot processing error: {str(e)}")

    def _get_binance_24h_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24h ticker data from Binance."""
        try:
            original_base_url = self.binance_client.base_url
            self.binance_client.base_url = self.binance_client.spot_base_url
            result = self.binance_client._make_request("/api/v3/ticker/24hr", {"symbol": symbol})
            self.binance_client.base_url = original_base_url
            return result
        except Exception as e:
            log_with_context(logger, 'warning', "Failed to get 24h ticker", symbol=symbol, error=str(e))
            return None