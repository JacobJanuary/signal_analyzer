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
    oi_average_usdt: Optional[float] = None
    oi_now_usdt: Optional[float] = None
    oi_change_pct_usdt: Optional[float] = None
    oi_source_usdt: Optional[str] = None

    # Spot USDT fields
    spot_avg_volume_usdt: Optional[float] = None
    spot_current_volume_usdt: Optional[float] = None
    spot_yesterday_volume_usdt: Optional[float] = None
    spot_today_yesterday_volume_change_pct: Optional[float] = None
    spot_volume_change_pct_usdt: Optional[float] = None
    spot_avg_price_usdt: Optional[float] = None
    spot_current_price_usdt: Optional[float] = None
    spot_price_change_pct_usdt: Optional[float] = None
    spot_source_usdt: Optional[str] = None

    # Spot BTC fields
    spot_avg_volume_btc: Optional[float] = None
    spot_current_volume_btc: Optional[float] = None
    spot_volume_change_pct_btc: Optional[float] = None
    spot_avg_price_btc: Optional[float] = None
    spot_current_price_btc: Optional[float] = None
    spot_price_change_pct_btc: Optional[float] = None
    spot_source_btc: Optional[str] = None

    # Price statistics fields
    current_price_usdt: Optional[float] = None
    min_price_1h: Optional[float] = None
    max_price_1h: Optional[float] = None
    min_price_24h: Optional[float] = None
    max_price_24h: Optional[float] = None
    min_price_7d: Optional[float] = None
    max_price_7d: Optional[float] = None
    min_price_30d: Optional[float] = None
    max_price_30d: Optional[float] = None
    price_change_pct_1d: Optional[float] = None
    price_change_pct_7d: Optional[float] = None
    price_change_pct_30d: Optional[float] = None
    price_stat_source: Optional[str] = None


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
            oi_average_usdt DECIMAL(18,2),
            oi_now_usdt DECIMAL(18,2),
            oi_change_pct_usdt DECIMAL(10,2),
            oi_source_usdt ENUM('binance', 'bybit'),
            
            -- Spot USDT
            spot_avg_volume_usdt DECIMAL(18,2),
            spot_current_volume_usdt DECIMAL(18,2),
            spot_yesterday_volume_usdt DECIMAL(18,2),
            spot_today_yesterday_volume_change_pct DECIMAL(10,2),
            spot_volume_change_pct_usdt DECIMAL(10,2),
            spot_avg_price_usdt DECIMAL(18,8),
            spot_current_price_usdt DECIMAL(18,8),
            spot_price_change_pct_usdt DECIMAL(10,2),
            spot_source_usdt ENUM('binance', 'bybit', 'coinmarketcap'),
            
            -- Spot BTC
            spot_avg_volume_btc DECIMAL(18,8),
            spot_current_volume_btc DECIMAL(18,8),
            spot_volume_change_pct_btc DECIMAL(10,2),
            spot_avg_price_btc DECIMAL(18,8),
            spot_current_price_btc DECIMAL(18,8),
            spot_price_change_pct_btc DECIMAL(10,2),
            spot_source_btc ENUM('binance', 'bybit', 'coinmarketcap'),
            
            -- Price statistics
            current_price_usdt DECIMAL(18,8),
            min_price_1h DECIMAL(18,8),
            max_price_1h DECIMAL(18,8),
            min_price_24h DECIMAL(18,8),
            max_price_24h DECIMAL(18,8),
            min_price_7d DECIMAL(18,8),
            max_price_7d DECIMAL(18,8),
            min_price_30d DECIMAL(18,8),
            max_price_30d DECIMAL(18,8),
            price_change_pct_1d DECIMAL(10,2),
            price_change_pct_7d DECIMAL(10,2),
            price_change_pct_30d DECIMAL(10,2),
            price_stat_source ENUM('binance', 'bybit', 'coinmarketcap'),
            
            INDEX idx_signal_id (signal_id),
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