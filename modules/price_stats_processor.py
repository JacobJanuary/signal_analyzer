# Файл: modules/price_stats_processor.py
# --- НОВЫЙ ФАЙЛ ---

"""Module: Price statistics processing from futures markets."""
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import sys

from api_clients.binance_client import BinanceClient
from api_clients.bybit_client import BybitClient
from utils.logger import setup_logger, log_with_context
from utils.helpers import (
    get_timestamp_ms,
    safe_float_conversion,
    calculate_percentage_change
)

logger = setup_logger(__name__)


@dataclass
class PriceStatsResult:
    """Price statistics calculation result."""
    price_min_1h: Optional[float] = None
    price_max_1h: Optional[float] = None
    price_min_24h: Optional[float] = None
    price_max_24h: Optional[float] = None
    price_min_7d: Optional[float] = None
    price_max_7d: Optional[float] = None
    price_min_30d: Optional[float] = None
    price_max_30d: Optional[float] = None
    percent_change_1d: Optional[float] = None
    percent_change_7d: Optional[float] = None
    percent_change_30d: Optional[float] = None
    source_exchange: Optional[str] = None
    error: Optional[str] = None


class PriceStatsProcessor:
    """Process price statistics data for signals."""

    def __init__(self):
        self.binance_client = BinanceClient()
        self.bybit_client = BybitClient()

    def process_symbol(self, symbol: str) -> PriceStatsResult:
        trading_symbol = f"{symbol}USDT"
        log_with_context(logger, 'info', "Starting price stats processing", symbol=trading_symbol)

        result = self._process_binance_futures(trading_symbol)
        if not result.error:
            return result

        log_with_context(logger, 'warning', "Binance price stats failed, trying Bybit", symbol=trading_symbol,
                         binance_error=result.error)

        result = self._process_bybit_futures(trading_symbol)
        if not result.error:
            return result

        log_with_context(logger, 'error', "Price stats processing failed on all sources", symbol=trading_symbol,
                         final_error=result.error)
        return result

    def _process_binance_futures(self, symbol: str) -> PriceStatsResult:
        try:
            utc_now = datetime.now(timezone.utc)

            # --- 30-дневные данные (основа для 30d и 7d) ---
            klines_30d = self.binance_client.get_futures_klines(symbol, "1d", limit=30)
            if not klines_30d:
                return PriceStatsResult(error="Failed to fetch 30d klines from Binance Futures")

            price_min_30d = min(safe_float_conversion(k[3]) for k in klines_30d)
            price_max_30d = max(safe_float_conversion(k[2]) for k in klines_30d)
            price_30d_ago = safe_float_conversion(klines_30d[0][1])  # Open of the first candle
            current_price = safe_float_conversion(klines_30d[-1][4])  # Close of the last candle
            percent_change_30d = calculate_percentage_change(current_price, price_30d_ago)

            # --- 7-дневные данные ---
            klines_7d = klines_30d[-7:]
            price_min_7d = min(safe_float_conversion(k[3]) for k in klines_7d)
            price_max_7d = max(safe_float_conversion(k[2]) for k in klines_7d)
            price_7d_ago = safe_float_conversion(klines_7d[0][1])
            percent_change_7d = calculate_percentage_change(current_price, price_7d_ago)

            # --- 24-часовые данные ---
            klines_24h = self.binance_client.get_futures_klines(symbol, "1h", limit=24)
            if not klines_24h:
                return PriceStatsResult(error="Failed to fetch 24h klines from Binance Futures")

            price_min_24h = min(safe_float_conversion(k[3]) for k in klines_24h)
            price_max_24h = max(safe_float_conversion(k[2]) for k in klines_24h)
            price_24h_ago = safe_float_conversion(klines_24h[0][1])
            percent_change_24h = calculate_percentage_change(current_price, price_24h_ago)

            # --- 1-часовые данные ---
            klines_1h = self.binance_client.get_futures_klines(symbol, "1m", limit=60)
            if not klines_1h:
                return PriceStatsResult(error="Failed to fetch 1h klines from Binance Futures")

            price_min_1h = min(safe_float_conversion(k[3]) for k in klines_1h)
            price_max_1h = max(safe_float_conversion(k[2]) for k in klines_1h)

            return PriceStatsResult(
                price_min_1h=price_min_1h, price_max_1h=price_max_1h,
                price_min_24h=price_min_24h, price_max_24h=price_max_24h,
                price_min_7d=price_min_7d, price_max_7d=price_max_7d,
                price_min_30d=price_min_30d, price_max_30d=price_max_30d,
                percent_change_1d=percent_change_24h,
                percent_change_7d=percent_change_7d,
                percent_change_30d=percent_change_30d,
                source_exchange="binance"
            )
        except Exception as e:
            return PriceStatsResult(error=f"Binance Futures processing error: {str(e)}")

    def _process_bybit_futures(self, symbol: str) -> PriceStatsResult:
        try:
            utc_now = datetime.now(timezone.utc)

            # --- 30-дневные данные ---
            klines_30d = self.bybit_client.get_klines(symbol, "D",
                                                      start_time=get_timestamp_ms(utc_now - timedelta(days=30)),
                                                      end_time=get_timestamp_ms(utc_now), limit=30)
            if not klines_30d:
                return PriceStatsResult(error="Failed to fetch 30d klines from Bybit Futures")

            price_min_30d = min(safe_float_conversion(k[3]) for k in klines_30d)
            price_max_30d = max(safe_float_conversion(k[2]) for k in klines_30d)
            price_30d_ago = safe_float_conversion(klines_30d[0][1])
            current_price = safe_float_conversion(klines_30d[-1][4])
            percent_change_30d = calculate_percentage_change(current_price, price_30d_ago)

            # --- 7-дневные данные ---
            klines_7d = klines_30d[-7:]
            price_min_7d = min(safe_float_conversion(k[3]) for k in klines_7d)
            price_max_7d = max(safe_float_conversion(k[2]) for k in klines_7d)
            price_7d_ago = safe_float_conversion(klines_7d[0][1])
            percent_change_7d = calculate_percentage_change(current_price, price_7d_ago)

            # --- 24-часовые данные ---
            klines_24h = self.bybit_client.get_klines(symbol, "60",
                                                      start_time=get_timestamp_ms(utc_now - timedelta(hours=24)),
                                                      end_time=get_timestamp_ms(utc_now), limit=24)
            if not klines_24h:
                return PriceStatsResult(error="Failed to fetch 24h klines from Bybit Futures")

            price_min_24h = min(safe_float_conversion(k[3]) for k in klines_24h)
            price_max_24h = max(safe_float_conversion(k[2]) for k in klines_24h)
            price_24h_ago = safe_float_conversion(klines_24h[0][1])
            percent_change_24h = calculate_percentage_change(current_price, price_24h_ago)

            # --- 1-часовые данные ---
            klines_1h = self.bybit_client.get_klines(symbol, "1",
                                                     start_time=get_timestamp_ms(utc_now - timedelta(minutes=60)),
                                                     end_time=get_timestamp_ms(utc_now), limit=60)
            if not klines_1h:
                return PriceStatsResult(error="Failed to fetch 1h klines from Bybit Futures")

            price_min_1h = min(safe_float_conversion(k[3]) for k in klines_1h)
            price_max_1h = max(safe_float_conversion(k[2]) for k in klines_1h)

            return PriceStatsResult(
                price_min_1h=price_min_1h, price_max_1h=price_max_1h,
                price_min_24h=price_min_24h, price_max_24h=price_max_24h,
                price_min_7d=price_min_7d, price_max_7d=price_max_7d,
                price_min_30d=price_min_30d, price_max_30d=price_max_30d,
                percent_change_1d=percent_change_24h,
                percent_change_7d=percent_change_7d,
                percent_change_30d=percent_change_30d,
                source_exchange="bybit"
            )
        except Exception as e:
            return PriceStatsResult(error=f"Bybit Futures processing error: {str(e)}")