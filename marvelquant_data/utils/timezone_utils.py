"""
Timezone Utility Module for Nexus Backtester

This module provides timezone conversion utilities for IST (Indian Standard Time)
and UTC (Coordinated Universal Time) handling, ensuring consistent timezone
management across the entire Nexus backtester system.

Architecture Standards:
    - Display Layer: ALL timestamps in IST (UTC+5:30)
    - Storage Layer: ALL timestamps in UTC
    - Parquet Files: ALL timestamps in UTC with timezone metadata

Regulatory Compliance:
    - SEBI requirements: IST timestamps for regulatory reporting
    - Data integrity: UTC storage for global synchronization

Usage Example:
    ```python
    from marvelquant_data.utils.timezone_utils import utc_to_ist, ist_to_utc, format_ist_display
    from datetime import datetime

    # Convert UTC to IST for display
    utc_dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    ist_dt = utc_to_ist(utc_dt)
    print(format_ist_display(utc_dt))  # "2024-01-15 15:30:00 IST"

    # Convert IST to UTC for storage
    ist_dt = datetime(2024, 1, 15, 15, 30, 0, tzinfo=IST_TZ)
    utc_dt = ist_to_utc(ist_dt)
    ```

Reference:
    Architecture documentation: /docs/bmad/architecture.md
    Registry metadata: /registry/domains/*/
"""

from datetime import datetime, timezone, timedelta, time, date
from typing import Optional
import pandas as pd

# IST Timezone Definition
IST_OFFSET = timedelta(hours=5, minutes=30)
IST_TZ = timezone(IST_OFFSET, name='IST')

# NSE Trading Hours (IST)
NSE_MARKET_OPEN = time(9, 15)    # 09:15 IST
NSE_MARKET_CLOSE = time(15, 30)  # 15:30 IST

# MCX Trading Hours (IST)
MCX_MARKET_OPEN = time(9, 0)     # 09:00 IST
MCX_MARKET_CLOSE = time(23, 30)  # 23:30 IST


# ============================================================================
# Core Timezone Conversion Functions
# ============================================================================

def utc_to_ist(utc_dt: datetime) -> datetime:
    """
    Convert UTC datetime to IST (Indian Standard Time) for display.

    Args:
        utc_dt: UTC datetime (must have tzinfo=timezone.utc or be naive)

    Returns:
        IST datetime with tzinfo=IST_TZ

    Example:
        >>> from datetime import datetime, timezone
        >>> utc_dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        >>> ist_dt = utc_to_ist(utc_dt)
        >>> print(ist_dt)
        2024-01-15 15:30:00+05:30

    Raises:
        ValueError: If input datetime has non-UTC timezone
    """
    if utc_dt.tzinfo is None:
        # Assume naive datetime is UTC
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    elif utc_dt.tzinfo != timezone.utc:
        raise ValueError(f"Input datetime must be UTC, got {utc_dt.tzinfo}")

    return utc_dt.astimezone(IST_TZ)


def ist_to_utc(ist_dt: datetime) -> datetime:
    """
    Convert IST datetime to UTC for storage.

    Args:
        ist_dt: IST datetime (must have tzinfo=IST_TZ or be naive IST)

    Returns:
        UTC datetime with tzinfo=timezone.utc

    Example:
        >>> from datetime import datetime
        >>> ist_dt = datetime(2024, 1, 15, 15, 30, 0, tzinfo=IST_TZ)
        >>> utc_dt = ist_to_utc(ist_dt)
        >>> print(utc_dt)
        2024-01-15 10:00:00+00:00

    Raises:
        ValueError: If input datetime has non-IST timezone (unless naive)
    """
    if ist_dt.tzinfo is None:
        # Assume naive datetime is IST
        ist_dt = ist_dt.replace(tzinfo=IST_TZ)
    elif ist_dt.tzinfo != IST_TZ:
        # Check if offset matches IST
        if ist_dt.utcoffset() != IST_OFFSET:
            raise ValueError(f"Input datetime must be IST (UTC+5:30), got {ist_dt.tzinfo}")

    return ist_dt.astimezone(timezone.utc)


def format_ist_display(utc_dt: datetime) -> str:
    """
    Format UTC timestamp for IST display in standard format.

    Args:
        utc_dt: UTC datetime

    Returns:
        Formatted string: "YYYY-MM-DD HH:MM:SS IST"

    Example:
        >>> from datetime import datetime, timezone
        >>> utc_dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        >>> print(format_ist_display(utc_dt))
        2024-01-15 15:30:00 IST
    """
    ist_dt = utc_to_ist(utc_dt)
    return ist_dt.strftime('%Y-%m-%d %H:%M:%S IST')


def format_utc_storage(dt: datetime) -> str:
    """
    Format datetime for UTC storage in ISO 8601 format.

    Args:
        dt: datetime (any timezone, will be converted to UTC)

    Returns:
        ISO 8601 formatted string: "YYYY-MM-DDTHH:MM:SSZ"

    Example:
        >>> from datetime import datetime
        >>> ist_dt = datetime(2024, 1, 15, 15, 30, 0, tzinfo=IST_TZ)
        >>> print(format_utc_storage(ist_dt))
        2024-01-15T10:00:00Z
    """
    if dt.tzinfo is None:
        raise ValueError("Input datetime must have timezone info")

    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')


# ============================================================================
# Market Hours Conversion
# ============================================================================

def convert_nse_time_to_utc(ist_time: time, date_obj: date) -> datetime:
    """
    Convert NSE trading time (IST) to UTC datetime.

    Args:
        ist_time: IST time object (e.g., time(9, 15) for market open)
        date_obj: Date for the trading day

    Returns:
        UTC datetime

    Example:
        >>> from datetime import time, date
        >>> market_open_utc = convert_nse_time_to_utc(time(9, 15), date(2024, 1, 15))
        >>> print(market_open_utc)
        2024-01-15 03:45:00+00:00
    """
    ist_dt = datetime.combine(date_obj, ist_time, tzinfo=IST_TZ)
    return ist_to_utc(ist_dt)


def convert_mcx_time_to_utc(ist_time: time, date_obj: date) -> datetime:
    """
    Convert MCX trading time (IST) to UTC datetime.

    Args:
        ist_time: IST time object (e.g., time(9, 0) for market open)
        date_obj: Date for the trading day

    Returns:
        UTC datetime

    Example:
        >>> from datetime import time, date
        >>> market_open_utc = convert_mcx_time_to_utc(time(9, 0), date(2024, 1, 15))
        >>> print(market_open_utc)
        2024-01-15 03:30:00+00:00
    """
    ist_dt = datetime.combine(date_obj, ist_time, tzinfo=IST_TZ)
    return ist_to_utc(ist_dt)


def get_nse_market_hours_utc(date_obj: date) -> tuple[datetime, datetime]:
    """
    Get NSE market hours in UTC for a given date.

    Args:
        date_obj: Trading date

    Returns:
        Tuple of (market_open_utc, market_close_utc)

    Example:
        >>> from datetime import date
        >>> open_utc, close_utc = get_nse_market_hours_utc(date(2024, 1, 15))
        >>> print(f"NSE: {open_utc} to {close_utc}")
        NSE: 2024-01-15 03:45:00+00:00 to 2024-01-15 10:00:00+00:00
    """
    market_open = convert_nse_time_to_utc(NSE_MARKET_OPEN, date_obj)
    market_close = convert_nse_time_to_utc(NSE_MARKET_CLOSE, date_obj)
    return market_open, market_close


def get_mcx_market_hours_utc(date_obj: date) -> tuple[datetime, datetime]:
    """
    Get MCX market hours in UTC for a given date.

    Args:
        date_obj: Trading date

    Returns:
        Tuple of (market_open_utc, market_close_utc)

    Example:
        >>> from datetime import date
        >>> open_utc, close_utc = get_mcx_market_hours_utc(date(2024, 1, 15))
        >>> print(f"MCX: {open_utc} to {close_utc}")
        MCX: 2024-01-15 03:30:00+00:00 to 2024-01-15 18:00:00+00:00
    """
    market_open = convert_mcx_time_to_utc(MCX_MARKET_OPEN, date_obj)
    market_close = convert_mcx_time_to_utc(MCX_MARKET_CLOSE, date_obj)
    return market_open, market_close


# ============================================================================
# Pandas DataFrame Timezone Conversion
# ============================================================================

def convert_df_utc_to_ist(df: pd.DataFrame, timestamp_col: str = 'ts_event') -> pd.DataFrame:
    """
    Convert UTC timestamps in DataFrame to IST for display.

    Args:
        df: DataFrame with UTC timestamps
        timestamp_col: Name of timestamp column (default: 'ts_event')

    Returns:
        DataFrame with IST timestamps

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'ts_event': [1705308000000000000]})  # UTC nanoseconds
        >>> df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
        >>> df_ist = convert_df_utc_to_ist(df)
        >>> print(df_ist['ts_event'])
        0   2024-01-15 15:30:00+05:30
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(IST_TZ)
    return df


def convert_df_ist_to_utc(df: pd.DataFrame, timestamp_col: str = 'ts_event') -> pd.DataFrame:
    """
    Convert IST timestamps in DataFrame to UTC for storage.

    Args:
        df: DataFrame with IST timestamps
        timestamp_col: Name of timestamp column (default: 'ts_event')

    Returns:
        DataFrame with UTC timestamps

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'ts_event': ['2024-01-15 15:30:00']})
        >>> df['ts_event'] = pd.to_datetime(df['ts_event']).dt.tz_localize(IST_TZ)
        >>> df_utc = convert_df_ist_to_utc(df)
        >>> print(df_utc['ts_event'])
        0   2024-01-15 10:00:00+00:00
    """
    df = df.copy()
    if df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(IST_TZ)
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(timezone.utc)
    return df


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_utc_timestamp(dt: datetime) -> bool:
    """
    Validate that datetime is in UTC timezone.

    Args:
        dt: datetime to validate

    Returns:
        True if UTC, False otherwise

    Example:
        >>> from datetime import datetime, timezone
        >>> utc_dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        >>> print(validate_utc_timestamp(utc_dt))
        True
    """
    return dt.tzinfo == timezone.utc


def validate_ist_timestamp(dt: datetime) -> bool:
    """
    Validate that datetime is in IST timezone.

    Args:
        dt: datetime to validate

    Returns:
        True if IST, False otherwise

    Example:
        >>> from datetime import datetime
        >>> ist_dt = datetime(2024, 1, 15, 15, 30, 0, tzinfo=IST_TZ)
        >>> print(validate_ist_timestamp(ist_dt))
        True
    """
    if dt.tzinfo is None:
        return False
    return dt.utcoffset() == IST_OFFSET


def validate_parquet_timestamp_metadata(df: pd.DataFrame, timestamp_col: str = 'ts_event') -> tuple[bool, str]:
    """
    Validate that parquet DataFrame has UTC timezone metadata.

    Args:
        df: DataFrame to validate
        timestamp_col: Timestamp column name

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'ts_event': [1705308000000000000]})
        >>> df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
        >>> valid, msg = validate_parquet_timestamp_metadata(df)
        >>> print(valid)
        True
    """
    if timestamp_col not in df.columns:
        return False, f"Timestamp column '{timestamp_col}' not found"

    if df[timestamp_col].dt.tz is None:
        return False, f"Timestamp column '{timestamp_col}' has no timezone (must be UTC)"

    if df[timestamp_col].dt.tz != timezone.utc:
        return False, f"Timestamp column '{timestamp_col}' is not UTC (got {df[timestamp_col].dt.tz})"

    return True, "Timestamp metadata valid (UTC)"


# ============================================================================
# Module Constants (for external use)
# ============================================================================

__all__ = [
    # Timezone constants
    'IST_TZ',
    'IST_OFFSET',
    'NSE_MARKET_OPEN',
    'NSE_MARKET_CLOSE',
    'MCX_MARKET_OPEN',
    'MCX_MARKET_CLOSE',

    # Core conversion functions
    'utc_to_ist',
    'ist_to_utc',
    'format_ist_display',
    'format_utc_storage',

    # Market hours conversion
    'convert_nse_time_to_utc',
    'convert_mcx_time_to_utc',
    'get_nse_market_hours_utc',
    'get_mcx_market_hours_utc',

    # DataFrame conversion
    'convert_df_utc_to_ist',
    'convert_df_ist_to_utc',

    # Validation utilities
    'validate_utc_timestamp',
    'validate_ist_timestamp',
    'validate_parquet_timestamp_metadata',
]
