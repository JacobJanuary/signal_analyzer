"""Module 2.1: Open Interest history processing."""
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from api_clients.binance_client import BinanceClient
from api_clients.bybit_client import BybitClient
from utils.logger import setup_logger, log_with_context
from utils.helpers import (
    get_30_days_range,
    safe_float_conversion,
    calculate_percentage_change,
    format_date_from_timestamp
)


logger = setup_logger(__name__)


class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    BYBIT = "bybit"


@dataclass
class OIResult:
    """Open Interest calculation result."""
    average_oi_usdt: Optional[float] = None
    current_oi_usdt: Optional[float] = None
    yesterday_oi_usdt: Optional[float] = None
    change_current_to_yesterday: Optional[float] = None
    change_current_to_average: Optional[float] = None
    source_exchange: Optional[Exchange] = None
    error: Optional[str] = None


class OIProcessor:
    """Process Open Interest data for signals."""

    def __init__(self):
        """Initialize OI processor."""
        self.binance_client = BinanceClient()
        self.bybit_client = BybitClient()

    def process_symbol(self, symbol: str) -> OIResult:
        """
        Process OI data for a symbol.

        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')

        Returns:
            OIResult with calculations or error
        """
        trading_symbol = f"{symbol}USDT"

        log_with_context(
            logger, 'info',
            "Starting OI processing",
            symbol=symbol,
            trading_symbol=trading_symbol
        )

        # Try Binance first
        result = self._process_binance_oi(trading_symbol)
        if not result.error:
            return result

        log_with_context(
            logger, 'warning',
            "Binance OI processing failed, trying Bybit",
            symbol=symbol,
            binance_error=result.error
        )

        # Try Bybit if Binance fails
        result = self._process_bybit_oi(trading_symbol)
        if not result.error:
            return result

        log_with_context(
            logger, 'error',
            "OI processing failed on all exchanges",
            symbol=symbol
        )

        return result

    def _process_binance_oi(self, trading_symbol: str) -> OIResult:
        """Process OI data from Binance."""
        start_time, end_time = get_30_days_range()

        # Get historical OI data
        historical_data = self.binance_client.get_historical_oi(
            trading_symbol, start_time, end_time
        )

        if not historical_data:
            return OIResult(error="Failed to fetch Binance historical OI data")

        # Calculate average OI
        oi_values = []
        for record in historical_data:
            try:
                oi_usdt = safe_float_conversion(record.get('sumOpenInterestValue'))
                timestamp = int(record.get('timestamp', 0))
                if oi_usdt > 0:
                    oi_values.append((timestamp, oi_usdt))
            except (KeyError, ValueError) as e:
                log_with_context(
                    logger, 'warning',
                    "Error processing historical OI record",
                    error=str(e),
                    record=record
                )

        if not oi_values:
            return OIResult(error="No valid historical OI data from Binance")

        # Sort by timestamp to ensure chronological order
        oi_values.sort(key=lambda x: x[0])

        # Extract values for calculations
        oi_values_only = [v[1] for v in oi_values]
        average_oi = sum(oi_values_only) / len(oi_values_only)

        # Get yesterday's OI (last value in historical data)
        yesterday_oi = oi_values_only[-1] if oi_values_only else None

        # Get current OI
        current_oi_usdt = self._get_binance_current_oi(trading_symbol)
        if current_oi_usdt is None:
            return OIResult(error="Failed to fetch current OI from Binance")

        # Calculate percentage changes
        change_to_avg = calculate_percentage_change(current_oi_usdt, average_oi)
        change_to_yesterday = calculate_percentage_change(current_oi_usdt, yesterday_oi) if yesterday_oi else None

        log_with_context(
            logger, 'info',
            "Binance OI processing successful",
            symbol=trading_symbol,
            average_oi=average_oi,
            current_oi=current_oi_usdt,
            yesterday_oi=yesterday_oi,
            change_to_avg=change_to_avg,
            change_to_yesterday=change_to_yesterday
        )

        return OIResult(
            average_oi_usdt=average_oi,
            current_oi_usdt=current_oi_usdt,
            yesterday_oi_usdt=yesterday_oi,
            change_current_to_yesterday=change_to_yesterday,
            change_current_to_average=change_to_avg,
            source_exchange=Exchange.BINANCE
        )

    def _get_binance_current_oi(self, symbol: str) -> Optional[float]:
        """Get current OI from Binance in USDT."""
        # Get OI in base asset
        oi_data = self.binance_client.get_current_oi(symbol)
        if not oi_data:
            return None

        oi_base = safe_float_conversion(oi_data.get('openInterest'))
        if oi_base == 0:
            return 0.0

        # Get current price
        price_data = self.binance_client.get_current_price(symbol)
        if not price_data:
            return None

        current_price = safe_float_conversion(price_data.get('price'))
        if current_price == 0:
            return None

        # Calculate OI in USDT
        return oi_base * current_price

    def _process_bybit_oi(self, trading_symbol: str) -> OIResult:
        """Process OI data from Bybit."""
        start_time, end_time = get_30_days_range()

        # Get historical OI with prices
        historical_data = self.bybit_client.get_historical_oi_with_prices(
            trading_symbol, start_time, end_time
        )

        if not historical_data:
            return OIResult(error="Failed to fetch Bybit historical OI data")

        # Calculate average OI in USDT
        oi_values_usdt = []
        for record in historical_data:
            oi_usdt = record.get('value_usdt', 0)
            timestamp = record.get('timestamp', 0)
            if oi_usdt > 0:
                oi_values_usdt.append((timestamp, oi_usdt))

        if not oi_values_usdt:
            return OIResult(error="No valid historical OI data from Bybit")

        # Sort by timestamp
        oi_values_usdt.sort(key=lambda x: x[0])

        # Extract values
        oi_values_only = [v[1] for v in oi_values_usdt]
        average_oi = sum(oi_values_only) / len(oi_values_only)

        # Get yesterday's OI
        yesterday_oi = oi_values_only[-1] if oi_values_only else None

        # Get current OI
        current_oi_usdt = self.bybit_client.get_current_oi_usdt(trading_symbol)
        if current_oi_usdt is None:
            return OIResult(error="Failed to fetch current OI from Bybit")

        # Calculate percentage changes
        change_to_avg = calculate_percentage_change(current_oi_usdt, average_oi)
        change_to_yesterday = calculate_percentage_change(current_oi_usdt, yesterday_oi) if yesterday_oi else None

        log_with_context(
            logger, 'info',
            "Bybit OI processing successful",
            symbol=trading_symbol,
            average_oi=average_oi,
            current_oi=current_oi_usdt,
            yesterday_oi=yesterday_oi,
            change_to_avg=change_to_avg,
            change_to_yesterday=change_to_yesterday
        )

        return OIResult(
            average_oi_usdt=average_oi,
            current_oi_usdt=current_oi_usdt,
            yesterday_oi_usdt=yesterday_oi,
            change_current_to_yesterday=change_to_yesterday,
            change_current_to_average=change_to_avg,
            source_exchange=Exchange.BYBIT
        )