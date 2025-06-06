"""Unit tests for Spot USDT processor module."""
import pytest
from unittest.mock import Mock, patch
from modules.spot_usdt_processor import SpotUSDTProcessor, SpotUSDTResult, Exchange


class TestSpotUSDTProcessor:
    """Test cases for Spot USDT processor."""

    @pytest.fixture
    def processor(self):
        """Create Spot USDT processor instance."""
        return SpotUSDTProcessor()

    @pytest.fixture
    def mock_binance_klines(self):
        """Mock Binance klines data."""
        return [
            # [open_time, open, high, low, close, volume, close_time, quote_volume, ...]
            [1704067200000, "40000", "41000", "39000", "40500", "1000", 1704153599999, "40500000", 8],
            [1704153600000, "40500", "41500", "40000", "41000", "1100", 1704239999999, "45100000", 8],
            [1704240000000, "41000", "42000", "40500", "41500", "1200", 1704326399999, "49800000", 8],
        ]

    @pytest.fixture
    def mock_bybit_klines(self):
        """Mock Bybit klines data."""
        return [
            # [timestamp, open, high, low, close, volume, turnover]
            ["1704067200000", "40000", "41000", "39000", "40500", "1000", "40500000"],
            ["1704153600000", "40500", "41500", "40000", "41000", "1100", "45100000"],
            ["1704240000000", "41000", "42000", "40500", "41500", "1200", "49800000"],
        ]

    def test_process_symbol_binance_success(self, processor, mock_binance_klines):
        """Test successful Binance spot processing."""
        with patch.object(
                processor.binance_client,
                'get_spot_klines',
                return_value=mock_binance_klines
        ):
            result = processor.process_symbol('BTC')

            assert result.error is None
            assert result.source_exchange == Exchange.BINANCE

            # Check volume calculations
            # Average volume: (40500000 + 45100000 + 49800000) / 3 = 45133333.33
            assert result.avg_volume_usdt == pytest.approx(45133333.33, rel=0.01)
            assert result.current_volume_usdt == 49800000  # Last volume

            # Check price calculations
            # Average price: (40500 + 41000 + 41500) / 3 = 41000
            assert result.avg_price_usdt == pytest.approx(41000, rel=0.01)
            assert result.current_price_usdt == 41500  # Last price

            # Check percentage changes
            # Volume change: (49800000 - 45133333.33) / 45133333.33 * 100 = 10.34%
            assert result.volume_change_percentage == pytest.approx(10.34, rel=0.01)
            # Price change: (41500 - 41000) / 41000 * 100 = 1.22%
            assert result.price_change_percentage == pytest.approx(1.22, rel=0.01)

    def test_process_symbol_binance_no_data(self, processor):
        """Test Binance spot processing with no data."""
        with patch.object(
                processor.binance_client,
                'get_spot_klines',
                return_value=None
        ):
            # Mock Bybit to also fail
            with patch.object(
                    processor.bybit_client,
                    'get_spot_klines',
                    return_value=None
            ):
                result = processor.process_symbol('BTC')

                assert result.error is not None
                assert "Failed to fetch" in result.error

    def test_process_symbol_fallback_to_bybit(self, processor, mock_bybit_klines):
        """Test fallback to Bybit when Binance fails."""
        with patch.object(
                processor.binance_client,
                'get_spot_klines',
                return_value=None
        ), patch.object(
            processor.bybit_client,
            'get_spot_klines',
            return_value=mock_bybit_klines
        ):
            result = processor.process_symbol('ETH')

            assert result.error is None
            assert result.source_exchange == Exchange.BYBIT
            assert result.avg_volume_usdt > 0
            assert result.current_volume_usdt > 0
            assert result.avg_price_usdt > 0
            assert result.current_price_usdt > 0

    def test_process_symbol_empty_klines(self, processor):
        """Test processing with empty klines data."""
        with patch.object(
                processor.binance_client,
                'get_spot_klines',
                return_value=[]
        ):
            result = processor._process_binance_spot('BTCUSDT')

            assert result.error is not None
            assert result.avg_volume_usdt is None
            assert result.current_volume_usdt is None

    def test_process_symbol_invalid_klines(self, processor):
        """Test processing with invalid klines data."""
        invalid_klines = [
            [1704067200000, "invalid", "41000", "39000", "40500", "1000", 1704153599999, "invalid"],
            [1704153600000, "40500", "41500", "40000", "41000", "1100", 1704239999999, "0"],
        ]

        with patch.object(
                processor.binance_client,
                'get_spot_klines',
                return_value=invalid_klines
        ):
            result = processor._process_binance_spot('BTCUSDT')

            # Should handle invalid data gracefully
            assert result.error is not None or result.avg_volume_usdt == 0

    def test_calculate_percentage_change_zero_average(self, processor):
        """Test percentage calculation with zero average."""
        mock_klines = [
            [1704067200000, "40000", "41000", "39000", "40500", "0", 1704153599999, "0", 8],
        ]

        with patch.object(
                processor.binance_client,
                'get_spot_klines',
                return_value=mock_klines
        ):
            result = processor._process_binance_spot('TESTUSDT')

            # Should handle zero volume gracefully
            assert result.error is not None or result.avg_volume_usdt == 0


class TestSpotUSDTProcessorIntegration:
    """Integration tests for Spot USDT processor."""

    @pytest.mark.integration
    def test_real_api_call_binance(self):
        """Test with real Binance API (requires network)."""
        processor = SpotUSDTProcessor()
        result = processor.process_symbol('BTC')

        # Just verify structure, not specific values
        assert isinstance(result, SpotUSDTResult)
        if not result.error:
            assert result.avg_volume_usdt > 0
            assert result.current_volume_usdt > 0
            assert result.avg_price_usdt > 0
            assert result.current_price_usdt > 0
            assert result.source_exchange in [Exchange.BINANCE, Exchange.BYBIT]