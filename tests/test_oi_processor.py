"""Unit tests for OI processor module."""
import pytest
from unittest.mock import Mock, patch
from modules.oi_processor import OIProcessor, OIResult, Exchange


class TestOIProcessor:
    """Test cases for OI processor."""

    @pytest.fixture
    def processor(self):
        """Create OI processor instance."""
        return OIProcessor()

    @pytest.fixture
    def mock_binance_historical_data(self):
        """Mock Binance historical OI data."""
        return [
            {'sumOpenInterestValue': '1000000.0', 'timestamp': 1704067200000},
            {'sumOpenInterestValue': '1100000.0', 'timestamp': 1704153600000},
            {'sumOpenInterestValue': '1200000.0', 'timestamp': 1704240000000},
        ]

    @pytest.fixture
    def mock_binance_current_oi(self):
        """Mock Binance current OI data."""
        return {'openInterest': '50000'}

    @pytest.fixture
    def mock_binance_price(self):
        """Mock Binance price data."""
        return {'price': '25.5'}

    def test_process_symbol_binance_success(
        self,
        processor,
        mock_binance_historical_data,
        mock_binance_current_oi,
        mock_binance_price
    ):
        """Test successful Binance OI processing."""
        with patch.object(
            processor.binance_client,
            'get_historical_oi',
            return_value=mock_binance_historical_data
        ), patch.object(
            processor.binance_client,
            'get_current_oi',
            return_value=mock_binance_current_oi
        ), patch.object(
            processor.binance_client,
            'get_current_price',
            return_value=mock_binance_price
        ):
            result = processor.process_symbol('BTC')

            assert result.error is None
            assert result.source_exchange == Exchange.BINANCE
            assert result.average_oi_usdt == 1100000.0  # (1M + 1.1M + 1.2M) / 3
            assert result.current_oi_usdt == 1275000.0  # 50000 * 25.5
            assert result.change_percentage == pytest.approx(15.91, rel=0.01)

    def test_process_symbol_binance_no_data(self, processor):
        """Test Binance OI processing with no data."""
        with patch.object(
            processor.binance_client,
            'get_historical_oi',
            return_value=None
        ):
            # Mock Bybit to also fail for complete failure test
            with patch.object(
                processor.bybit_client,
                'get_historical_oi_with_prices',
                return_value=None
            ):
                result = processor.process_symbol('BTC')

                assert result.error is not None
                assert "Failed to fetch" in result.error

    def test_process_symbol_fallback_to_bybit(self, processor):
        """Test fallback to Bybit when Binance fails."""
        mock_bybit_data = [
            {'timestamp': 1704067200000, 'value_usdt': 1000000.0},
            {'timestamp': 1704153600000, 'value_usdt': 1100000.0},
        ]

        with patch.object(
            processor.binance_client,
            'get_historical_oi',
            return_value=None
        ), patch.object(
            processor.bybit_client,
            'get_historical_oi_with_prices',
            return_value=mock_bybit_data
        ), patch.object(
            processor.bybit_client,
            'get_current_oi_usdt',
            return_value=1200000.0
        ):
            result = processor.process_symbol('ETH')

            assert result.error is None
            assert result.source_exchange == Exchange.BYBIT
            assert result.average_oi_usdt == 1050000.0  # (1M + 1.1M) / 2
            assert result.current_oi_usdt == 1200000.0
            assert result.change_percentage == pytest.approx(14.29, rel=0.01)

    def test_calculate_percentage_change_zero_average(self, processor):
        """Test percentage calculation with zero average."""
        with patch.object(
            processor.binance_client,
            'get_historical_oi',
            return_value=[{'sumOpenInterestValue': '0'}]
        ), patch.object(
            processor.binance_client,
            'get_current_oi',
            return_value={'openInterest': '1000'}
        ), patch.object(
            processor.binance_client,
            'get_current_price',
            return_value={'price': '10'}
        ):
            result = processor.process_symbol('TEST')

            # Should fall back to Bybit due to invalid data
            assert result.error is not None


class TestOIProcessorIntegration:
    """Integration tests for OI processor."""

    @pytest.mark.integration
    def test_real_api_call_binance(self):
        """Test with real Binance API (requires network)."""
        processor = OIProcessor()
        result = processor.process_symbol('BTC')

        # Just verify structure, not specific values
        assert isinstance(result, OIResult)
        if not result.error:
            assert result.average_oi_usdt > 0
            assert result.current_oi_usdt > 0
            assert result.source_exchange in [Exchange.BINANCE, Exchange.BYBIT]