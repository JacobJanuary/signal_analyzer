#!/usr/bin/env python3
"""Script to verify all fields are correctly mapped."""
from modules.spot_usdt_processor import SpotUSDTProcessor, SpotUSDTResult
from modules.oi_processor import OIProcessor, OIResult
from database.models import EnrichedSignalData
import inspect


def check_spot_usdt_result_fields():
    """Check SpotUSDTResult fields."""
    print("📋 SpotUSDTResult fields:")
    print("-" * 50)

    # Get all fields from dataclass
    fields = SpotUSDTResult.__dataclass_fields__
    for field_name, field_info in fields.items():
        print(f"  - {field_name}: {field_info.type}")

    print("\n📋 Expected fields in main.py:")
    expected_fields = [
        'avg_volume_usdt',
        'current_volume_usdt',
        'yesterday_volume_usdt',
        'volume_change_current_to_yesterday',
        'volume_change_current_to_average',
        'avg_price_usdt',
        'current_price_usdt',
        'yesterday_price_usdt',
        'price_change_1h',
        'price_change_24h',
        'price_change_7d',
        'price_change_30d',
        'source_exchange',
        'error'
    ]

    result_fields = set(fields.keys())
    expected_set = set(expected_fields)

    missing = expected_set - result_fields
    extra = result_fields - expected_set

    if missing:
        print(f"\n❌ Missing fields in SpotUSDTResult: {missing}")
    if extra:
        print(f"\n⚠️  Extra fields in SpotUSDTResult: {extra}")
    if not missing and not extra:
        print("\n✅ All fields match!")


def check_oi_result_fields():
    """Check OIResult fields."""
    print("\n\n📋 OIResult fields:")
    print("-" * 50)

    fields = OIResult.__dataclass_fields__
    for field_name, field_info in fields.items():
        print(f"  - {field_name}: {field_info.type}")

    print("\n📋 Expected fields in main.py:")
    expected_fields = [
        'average_oi_usdt',
        'current_oi_usdt',
        'yesterday_oi_usdt',
        'change_current_to_yesterday',
        'change_current_to_average',
        'source_exchange',
        'error'
    ]

    result_fields = set(fields.keys())
    expected_set = set(expected_fields)

    missing = expected_set - result_fields
    extra = result_fields - expected_set

    if missing:
        print(f"\n❌ Missing fields in OIResult: {missing}")
    if extra:
        print(f"\n⚠️  Extra fields in OIResult: {extra}")
    if not missing and not extra:
        print("\n✅ All fields match!")


def check_enriched_data_fields():
    """Check EnrichedSignalData fields."""
    print("\n\n📋 EnrichedSignalData fields (spot related):")
    print("-" * 50)

    fields = EnrichedSignalData.__dataclass_fields__
    spot_fields = [f for f in fields.keys() if 'spot' in f]

    for field_name in sorted(spot_fields):
        field_info = fields[field_name]
        print(f"  - {field_name}: {field_info.type}")


def test_spot_processing():
    """Test actual spot processing."""
    print("\n\n🧪 Testing actual spot processing for DEXE:")
    print("-" * 50)

    processor = SpotUSDTProcessor()
    result = processor.process_symbol('DEXE')

    if result.error:
        print(f"❌ Error: {result.error}")
    else:
        print("✅ Processing successful!")
        print(f"\nResult object attributes:")
        for attr in dir(result):
            if not attr.startswith('_'):
                value = getattr(result, attr)
                if not callable(value):
                    print(f"  - {attr}: {value}")


if __name__ == "__main__":
    check_spot_usdt_result_fields()
    check_oi_result_fields()
    check_enriched_data_fields()
    test_spot_processing()