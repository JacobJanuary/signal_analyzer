#!/usr/bin/env python3
"""Check field mapping between result objects and EnrichedSignalData."""
from modules.spot_usdt_processor import SpotUSDTResult
from modules.oi_processor import OIResult
from database.models import EnrichedSignalData


def check_mappings():
    """Check all field mappings."""
    print("🔍 Проверка соответствия полей\n")

    # OI mappings
    print("📊 OI поля:")
    print("-" * 80)
    oi_mappings = [
        ("EnrichedSignalData", "OIResult", "Описание"),
        ("oi_usdt_average", "average_oi_usdt", "Средний OI за 30 дней"),
        ("oi_usdt_current", "current_oi_usdt", "Текущий OI"),
        ("oi_usdt_yesterday", "yesterday_oi_usdt", "Вчерашний OI"),
        ("oi_usdt_change_current_to_yesterday", "change_current_to_yesterday", "Изменение текущий/вчера"),
        ("oi_usdt_change_current_to_average", "change_current_to_average", "Изменение текущий/средний"),
        ("oi_source_usdt", "source_exchange", "Источник данных"),
    ]

    for mapping in oi_mappings:
        print(f"{mapping[0]:<40} <- {mapping[1]:<30} # {mapping[2]}")

    # Spot USDT Volume mappings
    print("\n\n📊 Spot USDT Volume поля:")
    print("-" * 80)
    volume_mappings = [
        ("EnrichedSignalData", "SpotUSDTResult", "Описание"),
        ("spot_volume_usdt_average", "avg_volume_usdt", "Средний объем за 30 дней"),
        ("spot_volume_usdt_current", "current_volume_usdt", "Текущий объем"),
        ("spot_volume_usdt_yesterday", "yesterday_volume_usdt", "Вчерашний объем"),
        ("spot_volume_usdt_change_current_to_yesterday", "volume_change_current_to_yesterday",
         "Изменение объема текущий/вчера"),
        ("spot_volume_usdt_change_current_to_average", "volume_change_current_to_average",
         "Изменение объема текущий/средний"),
        ("spot_volume_source_usdt", "source_exchange", "Источник данных объема"),
    ]

    for mapping in volume_mappings:
        print(f"{mapping[0]:<50} <- {mapping[1]:<35} # {mapping[2]}")

    # Spot USDT Price mappings
    print("\n\n📊 Spot USDT Price поля:")
    print("-" * 80)
    price_mappings = [
        ("EnrichedSignalData", "SpotUSDTResult", "Описание"),
        ("spot_price_usdt_average", "avg_price_usdt", "Средняя цена за 30 дней"),
        ("spot_price_usdt_current", "current_price_usdt", "Текущая цена"),
        ("spot_price_usdt_yesterday", "yesterday_price_usdt", "Вчерашняя цена"),
        ("spot_price_usdt_change_1h", "price_change_1h", "Изменение цены за 1 час"),
        ("spot_price_usdt_change_24h", "price_change_24h", "Изменение цены за 24 часа"),
        ("spot_price_usdt_change_7d", "price_change_7d", "Изменение цены за 7 дней"),
        ("spot_price_usdt_change_30d", "price_change_30d", "Изменение цены за 30 дней"),
        ("spot_price_source_usdt", "source_exchange", "Источник данных цены"),
    ]

    for mapping in price_mappings:
        print(f"{mapping[0]:<40} <- {mapping[1]:<30} # {mapping[2]}")

    # Check actual fields exist
    print("\n\n✅ Проверка существования полей:")
    print("-" * 80)

    enriched_fields = set(EnrichedSignalData.__dataclass_fields__.keys())
    oi_fields = set(OIResult.__dataclass_fields__.keys())
    spot_fields = set(SpotUSDTResult.__dataclass_fields__.keys())

    # Check OI fields
    print("\nOI поля:")
    for db_field, result_field, _ in oi_mappings[1:]:  # Skip header
        db_exists = db_field in enriched_fields
        result_exists = result_field in oi_fields or result_field == "source_exchange"
        status = "✅" if db_exists and result_exists else "❌"
        print(f"{status} {db_field} <- {result_field}")
        if not db_exists:
            print(f"   ❌ {db_field} отсутствует в EnrichedSignalData")
        if not result_exists and result_field != "source_exchange":
            print(f"   ❌ {result_field} отсутствует в OIResult")

    # Check Volume fields
    print("\nSpot Volume поля:")
    for db_field, result_field, _ in volume_mappings[1:]:  # Skip header
        db_exists = db_field in enriched_fields
        result_exists = result_field in spot_fields or result_field == "source_exchange"
        status = "✅" if db_exists and result_exists else "❌"
        print(f"{status} {db_field} <- {result_field}")
        if not db_exists:
            print(f"   ❌ {db_field} отсутствует в EnrichedSignalData")
        if not result_exists and result_field != "source_exchange":
            print(f"   ❌ {result_field} отсутствует в SpotUSDTResult")

    # Check Price fields
    print("\nSpot Price поля:")
    for db_field, result_field, _ in price_mappings[1:]:  # Skip header
        db_exists = db_field in enriched_fields
        result_exists = result_field in spot_fields or result_field == "source_exchange"
        status = "✅" if db_exists and result_exists else "❌"
        print(f"{status} {db_field} <- {result_field}")
        if not db_exists:
            print(f"   ❌ {db_field} отсутствует в EnrichedSignalData")
        if not result_exists and result_field != "source_exchange":
            print(f"   ❌ {result_field} отсутствует в SpotUSDTResult")


if __name__ == "__main__":
    check_mappings()