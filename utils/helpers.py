"""Helper functions module."""
from datetime import datetime, timezone, timedelta
from typing import Optional, Union


def get_timestamp_ms(dt_object: datetime) -> int:
    """Convert datetime object to milliseconds timestamp."""
    return int(dt_object.timestamp() * 1000)


def format_date_from_timestamp(timestamp_ms: int) -> str:
    """Convert milliseconds timestamp to YYYY-MM-DD string."""
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc
    ).strftime('%Y-%m-%d')


def get_utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def get_30_days_range() -> tuple[int, int]:
    """Get timestamp range for last 30 days (up to yesterday)."""
    utc_now = get_utc_now()
    # End date is yesterday at 00:00:00 UTC
    end_date = utc_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    # Start date is 29 days before end date (30 days inclusive)
    start_date = end_date - timedelta(days=29)

    return get_timestamp_ms(start_date), get_timestamp_ms(end_date)


def safe_float_conversion(value: Union[str, float, None], default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def calculate_percentage_change(current: float, average: float) -> Optional[float]:
    """Calculate percentage change from average."""
    if average == 0:
        return None if current == 0 else float('inf')
    return ((current - average) / average) * 100