"""Database models and operations module."""
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from database.connection import get_db_cursor
from utils.logger import setup_logger, log_with_context


logger = setup_logger(__name__)


@dataclass
class SignalRecord:
    """Signal record from signals_10min table."""
    id: int
    symbol: str
    # Add other fields as needed based on actual table structure

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalRecord':
        """Create SignalRecord from dictionary."""
        return cls(
            id=data['id'],
            symbol=data['symbol']
        )


@dataclass
class EnrichedSignalData:
    """Enriched signal data for signals_10min_enriched table."""
    signal_id: int
    symbol: str
    created_at: datetime

    # OI USDT fields
    oi_usdt_average: Optional[float] = None
    oi_usdt_current: Optional[float] = None
    oi_usdt_yesterday: Optional[float] = None
    oi_usdt_change_current_to_yesterday: Optional[float] = None
    oi_usdt_change_current_to_average: Optional[float] = None
    oi_source_usdt: Optional[str] = None

    # Spot USDT volume fields
    spot_volume_usdt_average: Optional[float] = None
    spot_volume_usdt_current: Optional[float] = None
    spot_volume_usdt_yesterday: Optional[float] = None
    spot_volume_usdt_change_current_to_yesterday: Optional[float] = None
    spot_volume_usdt_change_current_to_average: Optional[float] = None
    spot_volume_source_usdt: Optional[str] = None

    # Spot BTC volume fields
    spot_volume_btc_average: Optional[float] = None
    spot_volume_btc_current: Optional[float] = None
    spot_volume_btc_yesterday: Optional[float] = None
    spot_volume_btc_change_current_to_yesterday: Optional[float] = None
    spot_volume_btc_change_current_to_average: Optional[float] = None
    spot_volume_source_btc: Optional[str] = None

    # Spot USDT price fields
    spot_price_usdt_average: Optional[float] = None
    spot_price_usdt_current: Optional[float] = None
    spot_price_usdt_yesterday: Optional[float] = None
    spot_price_usdt_change_1h: Optional[float] = None
    spot_price_usdt_change_24h: Optional[float] = None
    spot_price_usdt_change_7d: Optional[float] = None
    spot_price_usdt_change_30d: Optional[float] = None
    spot_price_source_usdt: Optional[str] = None

    # CoinMarketCap price statistics
    cmc_price_min_1h: Optional[float] = None
    cmc_price_max_1h: Optional[float] = None
    cmc_price_min_24h: Optional[float] = None
    cmc_price_max_24h: Optional[float] = None
    cmc_price_min_7d: Optional[float] = None
    cmc_price_max_7d: Optional[float] = None
    cmc_price_min_30d: Optional[float] = None
    cmc_price_max_30d: Optional[float] = None

    # CoinMarketCap percent changes
    cmc_percent_change_1d: Optional[float] = None
    cmc_percent_change_7d: Optional[float] = None
    cmc_percent_change_30d: Optional[float] = None


class SignalRepository:
    """Repository for signal database operations."""

    def get_new_signals(self, last_processed_id: int = 0) -> List[SignalRecord]:
        """Get new signals that haven't been processed yet."""
        query = """
            SELECT s.* 
            FROM signals_10min s
            LEFT JOIN signals_10min_enriched e ON s.id = e.signal_id
            WHERE s.id > %s AND e.id IS NULL
            ORDER BY s.id ASC
            LIMIT 100
        """

        with get_db_cursor() as cursor:
            cursor.execute(query, (last_processed_id,))
            results = cursor.fetchall()

            log_with_context(
                logger, 'info',
                "Fetched new signals",
                count=len(results),
                last_processed_id=last_processed_id
            )

            return [SignalRecord.from_dict(row) for row in results]

    def save_enriched_data(self, enriched_data: EnrichedSignalData) -> bool:
        """Save enriched signal data to database."""
        # Convert dataclass to dict and remove None values
        data_dict = {k: v for k, v in asdict(enriched_data).items() if v is not None}

        # Build dynamic INSERT query
        fields = list(data_dict.keys())
        placeholders = ["%s"] * len(fields)

        query = f"""
            INSERT INTO signals_10min_enriched ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
        """

        try:
            with get_db_cursor() as cursor:
                cursor.execute(query, list(data_dict.values()))

                log_with_context(
                    logger, 'info',
                    "Saved enriched signal data",
                    signal_id=enriched_data.signal_id,
                    symbol=enriched_data.symbol
                )

                return True

        except Exception as e:
            log_with_context(
                logger, 'error',
                "Failed to save enriched signal data",
                signal_id=enriched_data.signal_id,
                symbol=enriched_data.symbol,
                error=str(e)
            )
            return False

    def create_enriched_table_if_not_exists(self) -> bool:
        """Create the enriched signals table if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS signals_10min_enriched (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            signal_id BIGINT NOT NULL,
            symbol VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- OI, USDT
            oi_usdt_average DECIMAL(18,2),
            oi_usdt_current DECIMAL(18,2),
            oi_usdt_yesterday DECIMAL(18,2),
            oi_usdt_change_current_to_yesterday DECIMAL(10,2),
            oi_usdt_change_current_to_average DECIMAL(10,2),
            oi_source_usdt ENUM('binance', 'bybit'),
            
            -- Spot USDT volume
            spot_volume_usdt_average DECIMAL(18,2),
            spot_volume_usdt_current DECIMAL(18,2),
            spot_volume_usdt_yesterday DECIMAL(18,2),
            spot_volume_usdt_change_current_to_yesterday DECIMAL(10,2),
            spot_volume_usdt_change_current_to_average DECIMAL(10,2),
            spot_volume_source_usdt ENUM('binance', 'bybit', 'coinmarketcap'),
            
            -- Spot BTC volume
            spot_volume_btc_average DECIMAL(18,8),
            spot_volume_btc_current DECIMAL(18,8),
            spot_volume_btc_yesterday DECIMAL(18,8),
            spot_volume_btc_change_current_to_yesterday DECIMAL(10,2),
            spot_volume_btc_change_current_to_average DECIMAL(10,2),
            spot_volume_source_btc ENUM('binance', 'bybit', 'coinmarketcap'),
            
            -- Spot USDT price
            spot_price_usdt_average DECIMAL(18,8),
            spot_price_usdt_current DECIMAL(18,8),
            spot_price_usdt_yesterday DECIMAL(18,8),
            spot_price_usdt_change_1h DECIMAL(10,2),
            spot_price_usdt_change_24h DECIMAL(10,2),
            spot_price_usdt_change_7d DECIMAL(10,2),
            spot_price_usdt_change_30d DECIMAL(10,2),
            spot_price_source_usdt ENUM('binance', 'bybit', 'coinmarketcap'),
            
            -- Coinmarketcap: min/max за периоды
            cmc_price_min_1h DECIMAL(18,8),
            cmc_price_max_1h DECIMAL(18,8),
            cmc_price_min_24h DECIMAL(18,8),
            cmc_price_max_24h DECIMAL(18,8),
            cmc_price_min_7d DECIMAL(18,8),
            cmc_price_max_7d DECIMAL(18,8),
            cmc_price_min_30d DECIMAL(18,8),
            cmc_price_max_30d DECIMAL(18,8),
            
            -- Coinmarketcap: изменения цен
            cmc_percent_change_1d DECIMAL(10,2),
            cmc_percent_change_7d DECIMAL(10,2),
            cmc_percent_change_30d DECIMAL(10,2),
            
            INDEX idx_signal_id (signal_id),
            INDEX idx_symbol (symbol),
            INDEX idx_created_at (created_at),
            FOREIGN KEY (signal_id) REFERENCES signals_10min(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        try:
            with get_db_cursor() as cursor:
                cursor.execute(create_table_query)
                logger.info("Enriched signals table created or already exists")
                return True
        except Exception as e:
            log_with_context(
                logger, 'error',
                "Failed to create enriched signals table",
                error=str(e)
            )
            return False


# Global repository instance
signal_repository = SignalRepository()