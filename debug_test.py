#!/usr/bin/env python3
"""Debug why tests are calling real API."""
from unittest.mock import patch, MagicMock
from modules.spot_usdt_processor import SpotUSDTProcessor


def test_mock_behavior():
    """Test mock behavior."""
    processor = SpotUSDTProcessor()

    mock_klines = [
        [1704067200000, "40000", "41000", "39000", "40500", "1000", 1704153599999, "40500000", 8],
        [1704153600000, "40500", "41500", "40000", "41000", "1100", 1704239999999, "45100000", 8],
        [1704240000000, "41000", "42000", "40500", "41500", "1200", 1704326399999, "49800000", 8],
    ]

    print("Test 1: Direct mock test")
    with patch.object(processor.binance_client, 'get_spot_klines') as mock_get_klines:
        mock_get_klines.return_value = mock_klines

        # Also mock the internal _make_request to prevent real API calls
        with patch.object(processor.binance_client, '_make_request') as mock_request:
            mock_request.return_value = {
                'quoteVolume': '49800000',
                'lastPrice': '41500',
                'priceChangePercent': '1.22'
            }

            # Mock price changes calculation
            with patch.object(processor, '_calculate_price_changes') as mock_changes:
                mock_changes.return_value = {'1h': 0.5, '24h': None, '7d': 5.0, '30d': 10.0}

                result = processor.process_symbol('BTC')
                print(f"Result: {result}")
                print(f"Error: {result.error}")
                print(f"Current volume: {result.current_volume_usdt}")
                print(f"Current price: {result.current_price_usdt}")


if __name__ == "__main__":
    test_mock_behavior()