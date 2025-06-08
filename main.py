"""Main entry point for crypto signals enrichment."""
import sys
from datetime import datetime, timezone
from typing import Optional

from database.models import signal_repository, SignalRecord, EnrichedSignalData
from modules.oi_processor import OIProcessor
from modules.spot_usdt_processor import SpotUSDTProcessor
from modules.spot_btc_processor import SpotBTCProcessor
from modules.price_stats_processor import PriceStatsProcessor
from utils.logger import setup_logger, log_with_context
from config.settings import settings


logger = setup_logger(__name__)


class SignalEnrichmentProcessor:
    """Main processor for enriching crypto signals."""

    def __init__(self):
        """Initialize the processor."""
        self.oi_processor = OIProcessor()
        self.spot_usdt_processor = SpotUSDTProcessor()
        self.spot_btc_processor = SpotBTCProcessor()
        self.price_stats_processor = PriceStatsProcessor()
        # Future: Initialize other processors here
        # self.spot_btc_processor = SpotBTCProcessor()
        # self.price_stats_processor = PriceStatsProcessor()

        # Ensure the enriched table exists
        if not signal_repository.create_enriched_table_if_not_exists():
            raise RuntimeError("Failed to create enriched signals table")

    def process_new_signals(self) -> int:
        """
        Process all new signals.

        Returns:
            Number of signals processed
        """
        log_with_context(
            logger, 'info',
            "Starting signal enrichment process",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Get new signals
        new_signals = signal_repository.get_new_signals()

        if not new_signals:
            logger.info("No new signals to process")
            return 0

        log_with_context(
            logger, 'info',
            "Found new signals to process",
            count=len(new_signals)
        )

        processed_count = 0

        for signal in new_signals:
            try:
                enriched_data = self._process_single_signal(signal)
                if enriched_data:
                    if signal_repository.save_enriched_data(enriched_data):
                        processed_count += 1
                    else:
                        log_with_context(
                            logger, 'error',
                            "Failed to save enriched data",
                            signal_id=signal.id,
                            symbol=signal.symbol
                        )
            except Exception as e:
                log_with_context(
                    logger, 'error',
                    "Error processing signal",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    error=str(e),
                    exc_info=True
                )

        log_with_context(
            logger, 'info',
            "Signal enrichment process completed",
            total_signals=len(new_signals),
            processed_count=processed_count,
            failed_count=len(new_signals) - processed_count
        )

        return processed_count

    def _process_single_signal(self, signal: SignalRecord) -> Optional[EnrichedSignalData]:
        """
        Process a single signal through all modules.

        Args:
            signal: Signal record to process

        Returns:
            Enriched signal data or None if processing failed
        """
        log_with_context(
            logger, 'info',
            "Processing signal",
            signal_id=signal.id,
            symbol=signal.symbol
        )

        # Initialize enriched data
        enriched = EnrichedSignalData(
            signal_id=signal.id,
            symbol=signal.symbol,
            created_at=datetime.now(timezone.utc)
        )

        # Step 2.1: Process OI data
        try:
            oi_result = self.oi_processor.process_symbol(signal.symbol)
            if not oi_result.error:
                enriched.oi_usdt_average = oi_result.average_oi_usdt
                enriched.oi_usdt_current = oi_result.current_oi_usdt
                enriched.oi_usdt_yesterday = oi_result.yesterday_oi_usdt
                enriched.oi_usdt_change_current_to_yesterday = oi_result.change_current_to_yesterday
                enriched.oi_usdt_change_current_to_average = oi_result.change_current_to_average
                enriched.oi_source_usdt = oi_result.source_exchange.value if oi_result.source_exchange else None

                log_with_context(
                    logger, 'info',
                    "OI processing successful",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    source=enriched.oi_source_usdt,
                    average_oi=enriched.oi_usdt_average,
                    current_oi=enriched.oi_usdt_current,
                    yesterday_oi=enriched.oi_usdt_yesterday
                )
            else:
                log_with_context(
                    logger, 'warning',
                    "OI processing failed",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    error=oi_result.error
                )
        except Exception as e:
            log_with_context(
                logger, 'error',
                "Exception in OI processing",
                signal_id=signal.id,
                symbol=signal.symbol,
                error=str(e),
                exc_info=True
            )

        # Step 2.2: Process Spot USDT data
        try:
            spot_usdt_result = self.spot_usdt_processor.process_symbol(signal.symbol)
            if not spot_usdt_result.error:
                # Volume data
                enriched.spot_volume_usdt_average = spot_usdt_result.avg_volume_usdt
                enriched.spot_volume_usdt_current = spot_usdt_result.current_volume_usdt
                enriched.spot_volume_usdt_yesterday = spot_usdt_result.yesterday_volume_usdt
                enriched.spot_volume_usdt_change_current_to_yesterday = spot_usdt_result.volume_change_current_to_yesterday
                enriched.spot_volume_usdt_change_current_to_average = spot_usdt_result.volume_change_current_to_average
                enriched.spot_volume_source_usdt = spot_usdt_result.source_exchange.value if spot_usdt_result.source_exchange else None

                # Price data
                enriched.spot_price_usdt_average = spot_usdt_result.avg_price_usdt
                enriched.spot_price_usdt_current = spot_usdt_result.current_price_usdt
                enriched.spot_price_usdt_yesterday = spot_usdt_result.yesterday_price_usdt
                enriched.spot_price_usdt_change_1h = spot_usdt_result.price_change_1h
                enriched.spot_price_usdt_change_24h = spot_usdt_result.price_change_24h
                enriched.spot_price_usdt_change_7d = spot_usdt_result.price_change_7d
                enriched.spot_price_usdt_change_30d = spot_usdt_result.price_change_30d
                enriched.spot_price_source_usdt = spot_usdt_result.source_exchange.value if spot_usdt_result.source_exchange else None

                log_with_context(
                    logger, 'info',
                    "Spot USDT processing successful",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    source=enriched.spot_volume_source_usdt,  # Исправлено
                    avg_volume=enriched.spot_volume_usdt_average,  # Исправлено
                    current_volume=enriched.spot_volume_usdt_current,  # Исправлено
                    avg_price=enriched.spot_price_usdt_average,
                    current_price=enriched.spot_price_usdt_current
                )
            else:
                log_with_context(
                    logger, 'warning',
                    "Spot USDT processing failed",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    error=spot_usdt_result.error
                )
        except Exception as e:
            log_with_context(
                logger, 'error',
                "Exception in Spot USDT processing",
                signal_id=signal.id,
                symbol=signal.symbol,
                error=str(e),
                exc_info=True
            )

        # Step 2.3: Process Spot BTC data
        # Step 2.3: Process Spot BTC data
        try:
            spot_btc_result = self.spot_btc_processor.process_symbol(signal.symbol)
            if not spot_btc_result.error:
                enriched.spot_volume_btc_average = spot_btc_result.avg_volume_btc
                enriched.spot_volume_btc_current = spot_btc_result.current_volume_btc
                enriched.spot_volume_btc_yesterday = spot_btc_result.yesterday_volume_btc
                enriched.spot_volume_btc_change_current_to_yesterday = spot_btc_result.volume_change_current_to_yesterday
                enriched.spot_volume_btc_change_current_to_average = spot_btc_result.volume_change_current_to_average
                enriched.spot_volume_source_btc = spot_btc_result.source_exchange.value if spot_btc_result.source_exchange else None

                log_with_context(
                    logger, 'info',
                    "Spot BTC processing successful",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    source=enriched.spot_volume_source_btc,
                    avg_volume_btc=enriched.spot_volume_btc_average,
                    current_volume_btc=enriched.spot_volume_btc_current
                )
            else:
                log_with_context(
                    logger, 'warning',
                    "Spot BTC processing failed",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    error=spot_btc_result.error
                )
        except Exception as e:
            log_with_context(
                logger, 'error',
                "Exception in Spot BTC processing",
                signal_id=signal.id,
                symbol=signal.symbol,
                error=str(e),
                exc_info=True
            )

        # Step 2.4: Process price statistics
        try:
            stats_result = self.price_stats_processor.process_symbol(signal.symbol)
            if not stats_result.error:
                enriched.cmc_price_min_1h = stats_result.price_min_1h
                enriched.cmc_price_max_1h = stats_result.price_max_1h
                enriched.cmc_price_min_24h = stats_result.price_min_24h
                enriched.cmc_price_max_24h = stats_result.price_max_24h
                enriched.cmc_price_min_7d = stats_result.price_min_7d
                enriched.cmc_price_max_7d = stats_result.price_max_7d
                enriched.cmc_price_min_30d = stats_result.price_min_30d
                enriched.cmc_price_max_30d = stats_result.price_max_30d
                enriched.cmc_percent_change_1d = stats_result.percent_change_1d
                enriched.cmc_percent_change_7d = stats_result.percent_change_7d
                enriched.cmc_percent_change_30d = stats_result.percent_change_30d
                # Источник можно сохранить в одно из полей, если нужно,
                # но в схеме EnrichedSignalData отдельного поля для этого нет
                log_with_context(logger, 'info', "Price stats processing successful", signal_id=signal.id,
                                 symbol=signal.symbol, source=stats_result.source_exchange)
            else:
                log_with_context(logger, 'warning', "Price stats processing failed", signal_id=signal.id,
                                 symbol=signal.symbol, error=stats_result.error)
        except Exception as e:
            log_with_context(logger, 'error', "Exception in Price stats processing", signal_id=signal.id, error=str(e),
                             exc_info=True)

        return enriched


def main():
    """Main function for cron execution."""
    try:
        processor = SignalEnrichmentProcessor()
        processed = processor.process_new_signals()

        log_with_context(
            logger, 'info',
            "Main process completed successfully",
            processed_signals=processed
        )

        return 0
    except Exception as e:
        log_with_context(
            logger, 'error',
            "Fatal error in main process",
            error=str(e),
            exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())