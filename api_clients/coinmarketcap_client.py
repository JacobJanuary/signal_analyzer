"""CoinMarketCap API client module."""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
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
            'limit': 10  # Увеличиваем лимит для получения всех токенов с этим символом
        }

        try:
            response = self._make_request('/v1/cryptocurrency/map', params)
            if response and 'data' in response and response['data']:
                # Ищем активный токен
                for token in response['data']:
                    if token.get('is_active') == 1:
                        log_with_context(
                            logger, 'info',
                            "Active symbol found on CoinMarketCap",
                            symbol=symbol,
                            cmc_id=token['id'],
                            name=token['name']
                        )
                        return token

                # Если нет активных, возвращаем первый
                log_with_context(
                    logger, 'warning',
                    "No active token found, using first result",
                    symbol=symbol
                )
                return response['data'][0]
            else:
                log_with_context(
                    logger, 'warning',
                    "Symbol not found on CoinMarketCap",
                    symbol=symbol
                )
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
        interval: str = 'daily',
        count: int = 30
    ) -> Optional[List[Dict[str, Any]]]:
        """Get historical quotes using v2 API."""
        if not self.is_configured():
            return None

        # Используем v2 API как в рабочем примере
        url = '/v2/cryptocurrency/quotes/historical'

        params = {
            'symbol': symbol.upper(),
            'time_start': time_start.isoformat(),
            'time_end': time_end.isoformat(),
            'interval': interval,
            'convert': 'USD'
        }

        try:
            # Используем полный URL для v2 API
            original_base_url = self.base_url
            self.base_url = 'https://pro-api.coinmarketcap.com'

            response = self._make_request('/v2/cryptocurrency/quotes/historical', params)

            # Восстанавливаем URL
            self.base_url = original_base_url

            if response and 'data' in response:
                # v2 API возвращает данные в формате {symbol: [list of tokens]}
                token_list = response['data'].get(symbol.upper(), [])

                if not token_list:
                    return None

                # Ищем активный токен с историческими данными
                for token_data in token_list:
                    if token_data.get('is_active') == 1 and token_data.get('quotes'):
                        log_with_context(
                            logger, 'info',
                            "Found active token with historical data",
                            symbol=symbol,
                            name=token_data.get('name'),
                            id=token_data.get('id'),
                            quotes_count=len(token_data.get('quotes', []))
                        )
                        return token_data['quotes']

                # Если нет активных, пробуем первый токен
                if token_list[0].get('quotes'):
                    return token_list[0]['quotes']

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
            # Get historical data for last 31 days
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=31)

            historical_data = self.get_historical_quotes(
                symbol, start_time, end_time, interval='daily', count=31
            )

            if not historical_data:
                log_with_context(
                    logger, 'warning',
                    "No historical data from CoinMarketCap",
                    symbol=symbol
                )
                return None

            historical_data.sort(key=lambda x: x.get('timestamp'))
            # Get latest quote
            latest_quote = self.get_latest_quote(symbol)
            if not latest_quote:
                log_with_context(
                    logger, 'warning',
                    "No latest quote from CoinMarketCap",
                    symbol=symbol
                )
                return None

            # Фильтруем данные - исключаем сегодняшний день для расчета средних
            today_utc = datetime.now(timezone.utc).date()
            historical_quotes = []

            for quote in historical_data:
                try:
                    # Парсим timestamp
                    timestamp_str = quote.get('timestamp', '')
                    if timestamp_str:
                        # Убираем 'Z' и добавляем '+00:00' для корректного парсинга
                        timestamp_str = timestamp_str.replace('Z', '+00:00')
                        quote_date = datetime.fromisoformat(timestamp_str).date()

                        # Берем только данные до сегодняшнего дня
                        if quote_date < today_utc:
                            historical_quotes.append(quote)
                except Exception as e:
                    log_with_context(
                        logger, 'warning',
                        "Error parsing timestamp",
                        timestamp=quote.get('timestamp'),
                        error=str(e)
                    )

            if not historical_quotes:
                log_with_context(
                    logger, 'warning',
                    "No valid historical quotes after filtering",
                    symbol=symbol
                )
                return None

            # Берем последние 30 дней
            last_30_days = historical_quotes[-30:] if len(historical_quotes) > 30 else historical_quotes

            # Process historical data
            volumes_usd = []
            prices = []

            for quote in last_30_days:
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

            # Get yesterday's data (последний элемент в historical_quotes)
            yesterday_volume = volumes_usd[-1] if volumes_usd else avg_volume
            yesterday_price = prices[-1] if prices else avg_price

            # Get current data from latest quote
            current_quote = latest_quote.get('quote', {}).get('USD', {})
            current_price = safe_float_conversion(current_quote.get('price', 0))
            current_volume = safe_float_conversion(current_quote.get('volume_24h', 0))

            log_with_context(
                logger, 'info',
                "Successfully processed CoinMarketCap data",
                symbol=symbol,
                historical_days=len(last_30_days),
                avg_volume=avg_volume,
                current_volume=current_volume,
                yesterday_volume=yesterday_volume
            )

            return {
                'avg_volume_usdt': avg_volume,
                'current_volume_usdt': current_volume,
                'yesterday_volume_usdt': yesterday_volume,
                'avg_price_usd': avg_price,
                'current_price_usd': current_price,
                'yesterday_price_usd': yesterday_price,
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

    def get_latest_quotes_by_ids(self, ids: List[int]) -> Optional[Dict[str, Any]]:
        """Get latest market data for a list of CoinMarketCap IDs."""
        if not self.is_configured() or not ids:
            return None

        # Convert list of IDs to comma-separated string
        ids_str = ','.join(map(str, ids))

        params = {
            'id': ids_str,
            'convert': 'USD'
        }

        try:
            # Using v1 endpoint which is suitable for this
            response = self._make_request('/v1/cryptocurrency/quotes/latest', params)
            if response and 'data' in response:
                return response['data']
            return None
        except Exception as e:
            log_with_context(
                logger, 'error',
                "Error getting latest quotes by ID from CMC",
                ids=ids_str,
                error=str(e)
            )
            return None