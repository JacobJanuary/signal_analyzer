"""Main entry point for crypto signals enrichment."""
import sys
from datetime import datetime, timezone
from typing import Optional

from database.models import signal_repository, SignalRecord, EnrichedSignalData
from modules.oi_processor import OIProcessor
from modules.spot_usdt_processor import SpotUSDTProcessor
from utils.logger import setup_logger, log_with_context
from config.settings import settings


logger = setup_logger(__name__)


class SignalEnrichmentProcessor:
    """Main processor for enriching crypto signals."""

    def __init__(self):
        """Initialize the processor."""
        self.oi_processor = OIProcessor()
        self.spot_usdt_processor = SpotUSDTProcessor()
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
                enriched.oi_average_usdt = oi_result.average_oi_usdt
                enriched.oi_now_usdt = oi_result.current_oi_usdt
                enriched.oi_change_pct_usdt = oi_result.change_percentage
                enriched.oi_source_usdt = oi_result.source_exchange.value if oi_result.source_exchange else None

                log_with_context(
                    logger, 'info',
                    "OI processing successful",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    source=enriched.oi_source_usdt,
                    average_oi=enriched.oi_average_usdt,
                    current_oi=enriched.oi_now_usdt
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
                enriched.spot_avg_volume_usdt = spot_usdt_result.avg_volume_usdt
                enriched.spot_current_volume_usdt = spot_usdt_result.current_volume_usdt
                enriched.spot_volume_change_pct_usdt = spot_usdt_result.volume_change_percentage
                enriched.spot_avg_price_usdt = spot_usdt_result.avg_price_usdt
                enriched.spot_current_price_usdt = spot_usdt_result.current_price_usdt
                enriched.spot_price_change_pct_usdt = spot_usdt_result.price_change_percentage
                enriched.spot_source_usdt = spot_usdt_result.source_exchange.value if spot_usdt_result.source_exchange else None

                # Also save current price for price statistics
                enriched.current_price_usdt = spot_usdt_result.current_price_usdt

                log_with_context(
                    logger, 'info',
                    "Spot USDT processing successful",
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    source=enriched.spot_source_usdt,
                    avg_volume=enriched.spot_avg_volume_usdt,
                    current_volume=enriched.spot_current_volume_usdt
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
        # Step 2.4: Process price statistics

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