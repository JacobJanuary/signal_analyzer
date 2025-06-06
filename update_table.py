#!/usr/bin/env python3
"""Script to update existing table structure."""
from database.connection import get_db_cursor
from utils.logger import setup_logger

logger = setup_logger(__name__)


def add_new_columns():
    """Add new columns to existing table."""
    columns_to_add = [
        ("oi_now_usdt", "DECIMAL(18,2)", "AFTER oi_average_usdt"),
        ("spot_current_volume_usdt", "DECIMAL(18,2)", "AFTER spot_avg_volume_usdt"),
        ("spot_yesterday_volume_usdt", "DECIMAL(18,2)", "AFTER spot_current_volume_usdt"),
        ("spot_today_yesterday_volume_change_pct", "DECIMAL(10,2)", "AFTER spot_yesterday_volume_usdt"),
        ("spot_current_price_usdt", "DECIMAL(18,8)", "AFTER spot_avg_price_usdt"),
        ("spot_current_volume_btc", "DECIMAL(18,8)", "AFTER spot_avg_volume_btc"),
        ("spot_current_price_btc", "DECIMAL(18,8)", "AFTER spot_avg_price_btc"),
        ("current_price_usdt", "DECIMAL(18,8)", "AFTER spot_source_btc"),
    ]

    try:
        with get_db_cursor() as cursor:
            # Получаем существующие колонки
            cursor.execute("""
                           SELECT COLUMN_NAME
                           FROM INFORMATION_SCHEMA.COLUMNS
                           WHERE TABLE_SCHEMA = DATABASE()
                             AND TABLE_NAME = 'signals_10min_enriched'
                           """)
            existing_columns = {row['COLUMN_NAME'] for row in cursor.fetchall()}

            for column_name, column_type, position in columns_to_add:
                if column_name not in existing_columns:
                    alter_query = f"""
                    ALTER TABLE signals_10min_enriched 
                    ADD COLUMN {column_name} {column_type} {position}
                    """
                    cursor.execute(alter_query)
                    logger.info(f"Column {column_name} added successfully")
                    print(f"✅ Колонка {column_name} успешно добавлена")
                else:
                    logger.info(f"Column {column_name} already exists")
                    print(f"ℹ️  Колонка {column_name} уже существует")

            # Обновляем ENUM для поддержки coinmarketcap
            update_enum_queries = [
                """ALTER TABLE signals_10min_enriched
                    MODIFY COLUMN spot_source_usdt ENUM('binance', 'bybit', 'coinmarketcap')""",
                """ALTER TABLE signals_10min_enriched
                    MODIFY COLUMN spot_source_btc ENUM('binance', 'bybit', 'coinmarketcap')""",
                """ALTER TABLE signals_10min_enriched
                    MODIFY COLUMN price_stat_source ENUM('binance', 'bybit', 'coinmarketcap')"""
            ]

            for query in update_enum_queries:
                try:
                    cursor.execute(query)
                    print("✅ ENUM обновлен для поддержки CoinMarketCap")
                except Exception as e:
                    # Игнорируем если ENUM уже обновлен
                    pass

    except Exception as e:
        logger.error(f"Error updating table: {str(e)}")
        print(f"❌ Ошибка при обновлении таблицы: {str(e)}")
        raise


if __name__ == "__main__":
    print("🔄 Обновление структуры таблицы...")
    add_new_columns()
    print("✅ Готово!")