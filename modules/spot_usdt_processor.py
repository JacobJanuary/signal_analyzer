"""Module 2.2: Spot USDT volume and price processing."""
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from api_clients.binance_client import BinanceClient
from api_clients.bybit_client import BybitClient
from api_clients.coinmarketcap_client import CoinMarketCapClient
from utils.logger import setup_logger, log_with_context
from utils.helpers import (
    get_timestamp_ms,
    safe_float_conversion,
    calculate_percentage_change,
    format_date_from_timestamp
)


logger = setup_logger(__name__)


class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    BYBIT = "bybit"
    COINMARKETCAP = "coinmarketcap"


@dataclass
class SpotUSDTResult:
    """Spot USDT market calculation result."""
    avg_volume_usdt: Optional[float] = None
    current_volume_usdt: Optional[float] = None
    yesterday_volume_usdt: Optional[float] = None
    today_yesterday_volume_change_pct: Optional[float] = None
    volume_change_percentage: Optional[float] = None
    avg_price_usdt: Optional[float] = None
    current_price_usdt: Optional[float] = None
    price_change_percentage: Optional[float] = None
    source_exchange: Optional[Exchange] = None
    error: Optional[str] = None


class SpotUSDTProcessor:
    """Process Spot USDT market data for signals."""

    def __init__(self):
        """Initialize Spot USDT processor."""
        self.binance_client = BinanceClient()
        self.bybit_client = BybitClient()
        self.cmc_client = CoinMarketCapClient()

    def process_symbol(self, symbol: str) -> SpotUSDTResult:
        """
        Process spot USDT data for a symbol.

        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')

        Returns:
            SpotUSDTResult with calculations or error
        """
        trading_symbol = f"{symbol}USDT"

        log_with_context(
            logger, 'info',
            "Starting spot USDT processing",
            symbol=symbol,
            trading_symbol=trading_symbol
        )

        # Try Binance first
        result = self._process_binance_spot(trading_symbol)
        if not result.error:
            return result

        log_with_context(
            logger, 'warning',
            "Binance spot processing failed, trying Bybit",
            symbol=symbol,
            trading_symbol=trading_symbol,
            binance_error=result.error
        )

        # Try Bybit if Binance fails
        result = self._process_bybit_spot(trading_symbol)
        if not result.error:
            return result

        log_with_context(
            logger, 'warning',
            "Bybit spot processing failed, trying CoinMarketCap",
            symbol=symbol,
            trading_symbol=trading_symbol,
            bybit_error=result.error
        )

        # Try CoinMarketCap as last resort
        result = self._process_coinmarketcap(symbol)  # Note: using symbol, not trading_symbol
        if not result.error:
            return result

        log_with_context(
            logger, 'error',
            "Spot USDT processing failed on all sources",
            symbol=symbol,
            trading_symbol=trading_symbol,
            final_error=result.error
        )

        return result

    def _process_binance_spot(self, trading_symbol: str) -> SpotUSDTResult:
        """Process spot data from Binance."""
        try:
            log_with_context(
                logger, 'debug',
                "Starting Binance spot processing",
                symbol=trading_symbol
            )

            # Calculate time range for last 31 days to get yesterday's data
            utc_now = datetime.now(timezone.utc)
            end_time = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(days=31)

            # Get klines with proper timestamps
            klines = self.binance_client.get_spot_klines(
                trading_symbol,
                interval="1d",
                start_time=get_timestamp_ms(start_time),
                end_time=get_timestamp_ms(end_time),
                limit=31
            )

            if not klines:
                log_with_context(
                    logger, 'warning',
                    "No klines returned from Binance",
                    symbol=trading_symbol
                )
                return SpotUSDTResult(error=f"Failed to fetch Binance spot klines for {trading_symbol}")

            # Process klines data
            volumes_usdt = []
            prices = []

            for kline in klines:
                try:
                    # Kline structure: [open_time, open, high, low, close, volume, close_time, quote_volume, ...]
                    if len(kline) >= 8:
                        quote_volume = safe_float_conversion(kline[7])  # Volume in USDT
                        close_price = safe_float_conversion(kline[4])   # Close price

                        if quote_volume > 0:
                            volumes_usdt.append(quote_volume)
                        if close_price > 0:
                            prices.append(close_price)
                except (IndexError, ValueError) as e:
                    log_with_context(
                        logger, 'warning',
                        "Error processing kline",
                        error=str(e)
                    )

            if not volumes_usdt or not prices:
                return SpotUSDTResult(error=f"No valid spot data from Binance for {trading_symbol}")

            # Calculate averages (excluding today if we have 31 days)
            avg_data_volumes = volumes_usdt[:-1] if len(volumes_usdt) > 30 else volumes_usdt
            avg_data_prices = prices[:-1] if len(prices) > 30 else prices

            avg_volume = sum(avg_data_volumes) / len(avg_data_volumes)
            avg_price = sum(avg_data_prices) / len(avg_data_prices)

            # Get current data (last kline - today's incomplete data)
            # Note: for 24h volume, we might need to get ticker data instead
            current_ticker = self._get_binance_24h_ticker(trading_symbol)
            if current_ticker:
                current_volume = safe_float_conversion(current_ticker.get('quoteVolume', 0))
                current_price = safe_float_conversion(current_ticker.get('lastPrice', prices[-1]))
            else:
                current_volume = volumes_usdt[-1] if volumes_usdt else 0
                current_price = prices[-1] if prices else 0

            # Get yesterday's volume
            yesterday_volume = volumes_usdt[-2] if len(volumes_usdt) >= 2 else avg_volume

            # Calculate percentage changes
            volume_change_pct = calculate_percentage_change(current_volume, avg_volume)
            price_change_pct = calculate_percentage_change(current_price, avg_price)
            today_yesterday_change = calculate_percentage_change(current_volume, yesterday_volume)

            log_with_context(
                logger, 'info',
                "Binance spot processing successful",
                symbol=trading_symbol,
                avg_volume=avg_volume,
                current_volume=current_volume,
                yesterday_volume=yesterday_volume,
                avg_price=avg_price,
                current_price=current_price
            )

            return SpotUSDTResult(
                avg_volume_usdt=avg_volume,
                current_volume_usdt=current_volume,
                yesterday_volume_usdt=yesterday_volume,
                today_yesterday_volume_change_pct=today_yesterday_change,
                volume_change_percentage=volume_change_pct,
                avg_price_usdt=avg_price,
                current_price_usdt=current_price,
                price_change_percentage=price_change_pct,
                source_exchange=Exchange.BINANCE
            )

        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error in Binance spot processing",
                symbol=trading_symbol,
                error=str(e),
                exc_info=True
            )
            return SpotUSDTResult(error=f"Binance spot processing error for {trading_symbol}: {str(e)}")

    def _get_binance_24h_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24h ticker data from Binance."""
        try:
            # Временно меняем базовый URL для спотового запроса
            original_base_url = self.binance_client.base_url
            self.binance_client.base_url = self.binance_client.spot_base_url

            result = self.binance_client._make_request(
                "/api/v3/ticker/24hr",
                {"symbol": symbol}
            )

            # Восстанавливаем оригинальный URL
            self.binance_client.base_url = original_base_url

            return result
        except Exception as e:
            log_with_context(
                logger, 'warning',
                "Failed to get 24h ticker",
                symbol=symbol,
                error=str(e)
            )
            return None

    def _process_bybit_spot(self, trading_symbol: str) -> SpotUSDTResult:
        """Process spot data from Bybit."""
        try:
            # Calculate time range
            utc_now = datetime.now(timezone.utc)
            end_time = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(days=31)

            start_time_ms = get_timestamp_ms(start_time)
            end_time_ms = get_timestamp_ms(end_time)

            # Get spot klines from Bybit
            klines = self.bybit_client.get_spot_klines(
                trading_symbol,
                interval="D",
                start_time=start_time_ms,
                end_time=end_time_ms,
                limit=31
            )

            if not klines:
                return SpotUSDTResult(error="Failed to fetch Bybit spot klines")

            # Process klines data
            volumes_usdt = []
            prices = []

            for kline in klines:
                try:
                    # Bybit kline: [timestamp, open, high, low, close, volume, turnover]
                    if len(kline) >= 7:
                        turnover_usdt = safe_float_conversion(kline[6])  # Turnover in USDT
                        close_price = safe_float_conversion(kline[4])    # Close price

                        if turnover_usdt > 0:
                            volumes_usdt.append(turnover_usdt)
                        if close_price > 0:
                            prices.append(close_price)
                except (IndexError, ValueError) as e:
                    log_with_context(
                        logger, 'warning',
                        "Error processing Bybit kline",
                        error=str(e)
                    )

            # Limit to last 31 days
            volumes_usdt = volumes_usdt[-31:] if len(volumes_usdt) > 31 else volumes_usdt
            prices = prices[-31:] if len(prices) > 31 else prices

            if not volumes_usdt or not prices:
                return SpotUSDTResult(error="No valid spot data from Bybit")

            # Calculate averages (excluding today if we have 31 days)
            avg_data_volumes = volumes_usdt[:-1] if len(volumes_usdt) > 30 else volumes_usdt
            avg_data_prices = prices[:-1] if len(prices) > 30 else prices

            avg_volume = sum(avg_data_volumes) / len(avg_data_volumes)
            avg_price = sum(avg_data_prices) / len(avg_data_prices)

            # Get current data
            current_volume = volumes_usdt[-1] if volumes_usdt else 0
            current_price = prices[-1] if prices else 0

            # Get yesterday's volume
            yesterday_volume = volumes_usdt[-2] if len(volumes_usdt) >= 2 else avg_volume

            # Calculate percentage changes
            volume_change_pct = calculate_percentage_change(current_volume, avg_volume)
            price_change_pct = calculate_percentage_change(current_price, avg_price)
            today_yesterday_change = calculate_percentage_change(current_volume, yesterday_volume)

            log_with_context(
                logger, 'info',
                "Bybit spot processing successful",
                symbol=trading_symbol,
                avg_volume=avg_volume,
                current_volume=current_volume,
                yesterday_volume=yesterday_volume,
                avg_price=avg_price,
                current_price=current_price
            )

            return SpotUSDTResult(
                avg_volume_usdt=avg_volume,
                current_volume_usdt=current_volume,
                yesterday_volume_usdt=yesterday_volume,
                today_yesterday_volume_change_pct=today_yesterday_change,
                volume_change_percentage=volume_change_pct,
                avg_price_usdt=avg_price,
                current_price_usdt=current_price,
                price_change_percentage=price_change_pct,
                source_exchange=Exchange.BYBIT
            )

        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error in Bybit spot processing",
                error=str(e),
                exc_info=True
            )
            return SpotUSDTResult(error=f"Bybit spot processing error: {str(e)}")
            usdt[-30:] if len(volumes_usdt) > 30 else volumes_usdt
            prices = prices[-30:] if len(prices) > 30 else prices

            if not volumes_usdt or not prices:
                return SpotUSDTResult(error="No valid spot data from Bybit")

            # Calculate averages
            avg_volume = sum(volumes_usdt) / len(volumes_usdt)
            avg_price = sum(prices) / len(prices)

            # Get current data
            current_volume = volumes_usdt[-1] if volumes_usdt else 0
            current_price = prices[-1] if prices else 0

            # Calculate percentage changes
            volume_change_pct = calculate_percentage_change(current_volume, avg_volume)
            price_change_pct = calculate_percentage_change(current_price, avg_price)

            log_with_context(
                logger, 'info',
                "Bybit spot processing successful",
                symbol=trading_symbol,
                avg_volume=avg_volume,
                current_volume=current_volume,
                avg_price=avg_price,
                current_price=current_price
            )

            return SpotUSDTResult(
                avg_volume_usdt=avg_volume,
                current_volume_usdt=current_volume,
                volume_change_percentage=volume_change_pct,
                avg_price_usdt=avg_price,
                current_price_usdt=current_price,
                price_change_percentage=price_change_pct,
                source_exchange=Exchange.BYBIT
            )

        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error in Bybit spot processing",
                error=str(e),
                exc_info=True
            )
            return SpotUSDTResult(error=f"Bybit spot processing error: {str(e)}")

    def _process_coinmarketcap(self, symbol: str) -> SpotUSDTResult:
        """Process market data from CoinMarketCap."""
        try:
            if not self.cmc_client.is_configured():
                return SpotUSDTResult(error="CoinMarketCap API key not configured")

            log_with_context(
                logger, 'info',
                "Attempting CoinMarketCap data retrieval",
                symbol=symbol
            )

            # Get market data from CMC
            market_data = self.cmc_client.get_market_data_for_symbol(symbol)

            if not market_data:
                return SpotUSDTResult(error=f"No data available from CoinMarketCap for {symbol}")

            # Extract data
            avg_volume = market_data['avg_volume_usd']
            current_volume = market_data['current_volume_usd']
            yesterday_volume = market_data.get('yesterday_volume_usd', avg_volume)
            avg_price = market_data['avg_price_usd']
            current_price = market_data['current_price_usd']

            # Calculate percentage changes
            volume_change_pct = calculate_percentage_change(current_volume, avg_volume)
            price_change_pct = calculate_percentage_change(current_price, avg_price)
            today_yesterday_change = calculate_percentage_change(current_volume, yesterday_volume)

            log_with_context(
                logger, 'info',
                "CoinMarketCap processing successful",
                symbol=symbol,
                avg_volume=avg_volume,
                current_volume=current_volume,
                yesterday_volume=yesterday_volume,
                avg_price=avg_price,
                current_price=current_price,
                data_points=market_data['data_points']
            )

            return SpotUSDTResult(
                avg_volume_usdt=avg_volume,
                current_volume_usdt=current_volume,
                yesterday_volume_usdt=yesterday_volume,
                today_yesterday_volume_change_pct=today_yesterday_change,
                volume_change_percentage=volume_change_pct,
                avg_price_usdt=avg_price,
                current_price_usdt=current_price,
                price_change_percentage=price_change_pct,
                source_exchange=Exchange.COINMARKETCAP
            )

        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error in CoinMarketCap processing",
                symbol=symbol,
                error=str(e),
                exc_info=True
            )
            return SpotUSDTResult(error=f"CoinMarketCap processing error: {str(e)}")