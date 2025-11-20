#!/usr/bin/env python3
"""
Official Nautilus Data Transformation Pattern

Follows the verified pattern from nautilus_trader/examples/backtest/
- example_01_load_bars_from_custom_csv/run_example.py
- example_04_using_data_catalog/run_example.py

Transforms NSE data (index, futures, options) to Nautilus catalog format.
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.data import BarType, QuoteTick
from nautilus_trader.model.instruments import Equity, IndexInstrument, OptionContract, FuturesContract
from nautilus_trader.model.objects import Price, Quantity, Currency
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler, QuoteTickDataWrangler

# Import our contract generators
from marvelquant_data.utils.contract_generators import (
    create_options_contract,
    create_futures_contract,
    parse_nse_option_symbol
)

# Import custom data types
from marvelquant_data.data_types import OptionOI, FutureOI
from marvelquant_data.data_types.option_greeks import OptionGreeks

# Import Utils
from marvelquant_data.utils.greeks import OptionPricing
from marvelquant_data.utils.rates import InterestRateProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# IST offset (5 hours 30 minutes)
IST_OFFSET = timedelta(hours=5, minutes=30)


# Venue Mapping Configuration
# Maps underlying symbols to their primary exchange venue
VENUE_MAP = {
    # NSE Indices
    "NIFTY": "NSE",
    "BANKNIFTY": "NSE",
    "FINNIFTY": "NSE",
    "MIDCPNIFTY": "NSE",
    
    # BSE Indices
    "SENSEX": "BSE",
    "BANKEX": "BSE",
    
    # MCX Commodities
    "CRUDEOIL": "MCX",
    "NATURALGAS": "MCX",
    "GOLD": "MCX",
    "SILVER": "MCX",
    "COPPER": "MCX",
    "ZINC": "MCX",
    "LEAD": "MCX",
    "ALUMINIUM": "MCX",
    "NICKEL": "MCX",
}

# Default fallback venue
DEFAULT_VENUE = "NSE"

def get_venue(symbol: str) -> str:
    """
    Get the correct exchange venue for a symbol.
    
    Args:
        symbol: Underlying symbol (e.g., "NIFTY", "SENSEX", "SBIN")
        
    Returns:
        Venue string (e.g., "NSE", "BSE", "MCX")
    """
    # Normalize symbol for lookup
    norm_symbol = symbol.upper()
    return VENUE_MAP.get(norm_symbol, DEFAULT_VENUE)

def get_instrument_id_string(symbol: str, instrument_type: str) -> str:
    """
    Generate standardized InstrumentId string based on type and venue.
    
    Format:
    - Index: {SYMBOL}-INDEX.{VENUE} (e.g., NIFTY-INDEX.NSE)
    - Futures: {SYMBOL}-I.{VENUE} (e.g., NIFTY-I.NSE for continuous)
    - Equity: {SYMBOL}.{VENUE} (e.g., SBIN.NSE)
    - Options: handled by contract generators using specific option symbols
    
    Args:
        symbol: Base symbol (e.g., "NIFTY", "SBIN")
        instrument_type: "index", "future", "equity"
        
    Returns:
        Formatted InstrumentId string
    """
    venue = get_venue(symbol)
    norm_symbol = symbol.upper()
    
    if instrument_type == "index":
        return f"{norm_symbol}-INDEX.{venue}"
    elif instrument_type == "future":
        return f"{norm_symbol}-I.{venue}"
    elif instrument_type == "equity":
        return f"{norm_symbol}.{venue}"
    else:
        return f"{norm_symbol}.{venue}"  # Default fallthrough

def bars_to_quote_ticks(bars, instrument):
    """
    Convert Bar data to QuoteTicks for Greeks calculation.

    Creates QuoteTicks where bid=ask=close price.
    This is required for NautilusTrader Greeks calculator.
    """
    quote_ticks = []

    for bar in bars:
        # Create QuoteTick using close price as both bid and ask
        price = Price(bar.close.as_double(), instrument.price_precision)
        size = Quantity(1, instrument.size_precision)

        tick = QuoteTick(
            instrument_id=instrument.id,
            bid_price=price,
            ask_price=price,
            bid_size=size,
            ask_size=size,
            ts_event=bar.ts_event,
            ts_init=bar.ts_init,
        )
        quote_ticks.append(tick)

    return quote_ticks


def yyyymmdd_seconds_to_datetime(date_int, time_int: int) -> datetime:
    """
    Convert YYYYMMDD integer + seconds to datetime in UTC.
    
    Args:
        date_int: Date as YYYYMMDD (e.g., 20240102) or datetime.date object
        time_int: Time as seconds since midnight (e.g., 33300 = 09:15:00)
    
    Returns:
        datetime in UTC
    """
    # Handle both int and datetime.date types
    if isinstance(date_int, (datetime, pd.Timestamp)):
        date_int = int(date_int.strftime('%Y%m%d'))
    elif hasattr(date_int, 'year'):  # datetime.date object
        date_int = date_int.year * 10000 + date_int.month * 100 + date_int.day
    
    # Parse date
    year = date_int // 10000
    month = (date_int % 10000) // 100
    day = date_int % 100
    
    # Parse time
    hours = time_int // 3600
    minutes = (time_int % 3600) // 60
    seconds = time_int % 60
    
    # Create IST datetime (naive)
    ist_dt = datetime(year, month, day, hours, minutes, seconds)
    
    # Convert to UTC
    utc_dt = ist_dt - IST_OFFSET
    
    return utc_dt


def transform_index_bars(
    input_dir: Path,
    catalog: ParquetDataCatalog,
    symbol: str,
    start_date: str,
    end_date: str
) -> int:
    """
    Transform index data to Nautilus Bar format (OFFICIAL PATTERN).
    
    Args:
        input_dir: Directory containing raw parquet files
        catalog: Nautilus ParquetDataCatalog instance
        symbol: Symbol name (e.g., "NIFTY", "BANKNIFTY")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Number of bars created
    """
    logger.info(f"Transforming {symbol} index bars...")
    
    # Find all parquet files for this symbol
    symbol_dir = input_dir / "index" / symbol.lower()
    parquet_files = list(symbol_dir.rglob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {symbol_dir}")
        return 0
    
    # Read all files into one DataFrame
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        logger.error("No data loaded")
        return 0
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert date + time to datetime timestamp
    combined_df['timestamp'] = combined_df.apply(
        lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
        axis=1
    )
    
    logger.info(f"Data range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    # Filter by date range (account for IST->UTC conversion: IST dates start at UTC-5:30)
    start = pd.to_datetime(start_date) - pd.Timedelta(hours=6)  # Buffer for IST conversion
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    combined_df = combined_df[(combined_df['timestamp'] >= start) & 
                               (combined_df['timestamp'] < end)]
    
    if combined_df.empty:
        logger.warning(f"No data in date range {start_date} to {end_date}")
        return 0
    
    # OFFICIAL PATTERN: Prepare DataFrame for BarDataWrangler
    # Required: columns ['open', 'high', 'low', 'close', 'volume'] with 'timestamp' as INDEX
    bar_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # NOTE: Index data is already in rupees (not paise), NO conversion needed
    # Prices are correct as-is: 21476.00, not 214.76

    # CRITICAL: Deduplicate by timestamp (keep last value) before setting index
    bar_df = bar_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    bar_df = bar_df.set_index('timestamp')  # CRITICAL: Set timestamp as index!
    bar_df = bar_df.sort_index()  # Sort by timestamp
    
    # Determine Venue
    venue_str = get_venue(symbol)
    venue = Venue(venue_str)
    
    # Create InstrumentId
    # Use helper to ensure consistent naming (e.g. NIFTY-INDEX.NSE)
    instrument_id_str = get_instrument_id_string(symbol, "index")
    instrument_id = InstrumentId.from_str(instrument_id_str)
    
    # Use IndexInstrument (not Equity) for index instruments
    instrument = IndexInstrument(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        currency=Currency.from_str("INR"),
        price_precision=2,
        price_increment=Price(0.05, 2),
        size_precision=0,
        size_increment=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )
    
    # Create bar type
    bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")
    
    # OFFICIAL PATTERN: Use BarDataWrangler
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(
        data=bar_df,
        default_volume=0.0,  # Index data has no real volume
        ts_init_delta=0
    )
    
    # OFFICIAL PATTERN: Write to catalog
    catalog.write_data([instrument])  # Write instrument first
    catalog.write_data(bars, skip_disjoint_check=True)  # Skip check for overlapping data

    # Generate and write QuoteTicks for Greeks calculation
    quote_ticks = bars_to_quote_ticks(bars, instrument)
    catalog.write_data(quote_ticks, skip_disjoint_check=True)
    logger.info(f"✅ {symbol}: Created {len(bars):,} bars + {len(quote_ticks):,} QuoteTicks")

    return len(bars)


def transform_futures_bars(
    input_dir: Path,
    catalog: ParquetDataCatalog,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path = None
) -> tuple[int, None]:
    """
    Transform futures data to Nautilus Bar format + separate OI DataFrame.
    
    Returns:
        (bar_count, oi_dataframe)
    """
    logger.info(f"Transforming {symbol} futures bars...")
    
    symbol_dir = input_dir / "futures" / symbol.lower()
    parquet_files = list(symbol_dir.rglob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {symbol_dir}")
        return 0, pd.DataFrame()

    # CRITICAL: Only use dated files (nifty_future_YYYYMMDD.parquet) which are in RUPEES
    # Exclude data.parquet (in paise) and futures_data.parquet (corrupt)
    dated_files = [f for f in parquet_files if f.stem.startswith(f"{symbol.lower()}_future_")]

    if not dated_files:
        logger.warning(f"No dated futures files found in {symbol_dir}")
        return 0, pd.DataFrame()

    logger.info(f"Using {len(dated_files)} dated futures files (already in rupees)")

    dfs = []
    for file in dated_files:
        try:
            df = pd.read_parquet(file)
            # Handle mixed date formats
            if df['date'].dtype == 'object':
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(int)
            # Ensure time is int
            if df['time'].dtype == 'object':
                df['time'] = df['time'].astype(int)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        return 0, pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert to timestamp
    combined_df['timestamp'] = combined_df.apply(
        lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
        axis=1
    )
    
    logger.info(f"Futures data range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    # Filter by date range (account for IST->UTC conversion)
    start = pd.to_datetime(start_date) - pd.Timedelta(hours=6)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    combined_df = combined_df[(combined_df['timestamp'] >= start) & 
                               (combined_df['timestamp'] < end)]
    
    if combined_df.empty:
        return 0, pd.DataFrame()
    
    # Prepare for BarDataWrangler (OHLCV only, NO OI!)
    bar_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # NOTE: Dated futures files (nifty_future_YYYYMMDD.parquet) are ALREADY in RUPEES
    # No conversion needed!

    # Data quality fixes
    bar_df['volume'] = bar_df['volume'].clip(lower=0)  # Handle negative volumes
    
    # Fix invalid OHLC relationships (Nautilus validates: high >= close, low <= close)
    bar_df['high'] = bar_df[['high', 'close']].max(axis=1)
    bar_df['low'] = bar_df[['low', 'close']].min(axis=1)
    
    bar_df = bar_df.set_index('timestamp')
    bar_df = bar_df.sort_index()
    
    # Determine Venue
    venue_str = get_venue(symbol)
    
    # Create FuturesContract (use proper Nautilus instrument type)
    # Symbol for continuous future: {SYMBOL}-I (e.g. NIFTY-I)
    future_symbol = f"{symbol.upper()}-I"
    
    instrument = create_futures_contract(
        symbol=future_symbol,  # -I for continuous futures
        expiry_date="continuous",  # Continuous contract
        underlying=symbol,
        venue=venue_str
    )
    
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    
    # Create bars
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(bar_df)
    
    # Write to catalog
    catalog.write_data([instrument])
    catalog.write_data(bars, skip_disjoint_check=True)

    # Generate and write QuoteTicks for Greeks calculation
    quote_ticks = bars_to_quote_ticks(bars, instrument)
    catalog.write_data(quote_ticks, skip_disjoint_check=True)

    # Create FutureOI custom data (Arrow serialization registered)
    oi_data_list = []
    prev_oi = 0
    for idx, row in combined_df.iterrows():
        current_oi = int(row["oi"])
        coi = current_oi - prev_oi
        prev_oi = current_oi
        ts_ns = int(row["timestamp"].timestamp() * 1_000_000_000)
        
        oi_data = FutureOI(
            instrument_id=instrument.id,
            oi=current_oi,
            coi=coi,
            ts_event=ts_ns,
            ts_init=ts_ns
        )
        oi_data_list.append(oi_data)
    
    # Write FutureOI to catalog (Arrow registered)
    if oi_data_list:
        oi_data_list.sort(key=lambda x: x.ts_init)
        catalog.write_data(oi_data_list)
        logger.info(f"✅ Saved {len(oi_data_list):,} FutureOI records")
    
    logger.info(f"✅ {symbol} futures: Created {len(bars):,} bars + {len(quote_ticks):,} QuoteTicks")
    return len(bars), None  # No longer returning DataFrame


def transform_options_bars(
    input_dir: Path,
    catalog: ParquetDataCatalog,
    symbol: str,
    start_date: str,
    end_date: str
) -> int:
    """
    Transform options data to Nautilus Bar format + calculate Greeks (OFFICIAL PATTERN).
    """
    logger.info(f"Transforming {symbol} options bars with Greeks...")
    
    # Initialize Interest Rate Provider
    ir_xml_path = PROJECT_ROOT / "data/interest_rates/india_91day_tbill_rates_2018_2025_nautilus.xml"
    ir_provider = InterestRateProvider(ir_xml_path)

    # Load underlying index/spot data for the period to calculate Greeks
    # We'll load raw index files again to create a quick lookup DataFrame
    index_dir = input_dir / "index" / symbol.lower()
    index_files = list(index_dir.rglob("*.parquet"))
    spot_df = pd.DataFrame()
    
    if index_files:
        dfs = []
        for f in index_files:
            try:
                # Simple date filtering on filename or content could speed this up
                # For now read all and filter
                temp_df = pd.read_parquet(f)
                dfs.append(temp_df)
            except Exception:
                pass
        if dfs:
            spot_raw = pd.concat(dfs, ignore_index=True)
            spot_raw['timestamp'] = spot_raw.apply(
                lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']), axis=1
            )
            # Filter spot data to relevant range
            s_start = pd.to_datetime(start_date) - pd.Timedelta(hours=6)
            s_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            spot_raw = spot_raw[(spot_raw['timestamp'] >= s_start) & (spot_raw['timestamp'] < s_end)]
            
            # Create lookup: timestamp -> close price
            # CRITICAL: Deduplicate by keeping last value for each timestamp
            spot_df = spot_raw[['timestamp', 'close']].rename(columns={'close': 'spot_price'})
            spot_df = spot_df.drop_duplicates(subset=['timestamp'], keep='last')
            spot_df = spot_df.set_index('timestamp').sort_index()
            logger.info(f"Loaded {len(spot_df)} spot price records for Greeks calculation (deduplicated)")

    symbol_dir = input_dir / "option" / symbol.lower()
    if not symbol_dir.exists():
        logger.warning(f"No options directory found: {symbol_dir}")
        return 0
    
    # CRITICAL: Only use dated files (nifty_call/put_YYYYMMDD.parquet) which are in RUPEES
    all_files = list(symbol_dir.rglob("*.parquet"))

    # Filter for dated call/put files (already in rupees)
    symbol_lower = symbol.lower()
    dated_call_files = [f for f in all_files if f.stem.startswith(f"{symbol_lower}_call_")]
    dated_put_files = [f for f in all_files if f.stem.startswith(f"{symbol_lower}_put_")]
    parquet_files = dated_call_files + dated_put_files

    if not parquet_files:
        logger.warning(f"No dated option files found in {symbol_dir}")
        return 0

    logger.info(f"Using {len(parquet_files)} dated option files (already in rupees)")

    total_bars = 0
    total_greeks = 0

    # Process dated option files
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            
            if 'symbol' not in df.columns or df.empty:
                continue
            
            # Convert timestamp
            df['timestamp'] = df.apply(
                lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
                axis=1
            )
            
            # Filter by date
            start = pd.to_datetime(start_date) - pd.Timedelta(hours=6)
            end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
            
            if df.empty:
                continue
            
            # Group by option symbol
            for option_symbol, group in df.groupby('symbol'):
                try:
                    # 1. Prepare Bars
                    bar_df = group[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                    bar_df['volume'] = bar_df['volume'].clip(lower=0)
                    bar_df['high'] = bar_df[['high', 'close', 'open']].max(axis=1)
                    bar_df['low'] = bar_df[['low', 'close', 'open']].min(axis=1)
                    # CRITICAL: Deduplicate bars by timestamp (keep last value)
                    bar_df = bar_df.drop_duplicates(subset=['timestamp'], keep='last')
                    bar_df = bar_df.set_index('timestamp').sort_index()
                    
                    # Parse contract details
                    try:
                        parsed = parse_nse_option_symbol(option_symbol)
                        expiry_date = parsed['expiry']
                        strike_price = parsed['strike']
                        option_type = parsed['option_type']  # "CALL" or "PUT"
                        opt_type_code = 'CE' if option_type == 'CALL' else 'PE'
                        underlying_symbol = parsed['underlying']
                        
                        # Determine Venue based on Underlying
                        venue_str = get_venue(underlying_symbol)
                        
                        contract = create_options_contract(
                            symbol=option_symbol,
                            underlying=underlying_symbol,
                            strike=strike_price,
                            expiry=expiry_date,
                            option_kind=option_type,
                            venue=venue_str
                        )
                    except Exception:
                         # Fallback if parsing fails (skip greeks for unknown formats)
                         continue

                    # Create Bars
                    bar_type = BarType.from_str(f"{contract.id}-1-MINUTE-LAST-EXTERNAL")
                    wrangler = BarDataWrangler(bar_type, contract)
                    
                    # Validate data before processing
                    if bar_df.empty:
                        logger.warning(f"Skipping {option_symbol}: empty bar_df after deduplication")
                        continue
                    
                    # Check for zero prices (data quality check)
                    zero_price_count = ((bar_df[['open', 'high', 'low', 'close']] == 0).all(axis=1)).sum()
                    if zero_price_count > 0:
                        logger.warning(f"{option_symbol}: {zero_price_count} rows with all zero prices (out of {len(bar_df)})")
                    
                    bars = wrangler.process(bar_df)
                    
                    # Validate bars after processing
                    if bars and len(bars) > 0:
                        sample_bar = bars[0]
                        if sample_bar.open.as_double() == 0 and sample_bar.close.as_double() == 0:
                            logger.error(f"{option_symbol}: All bars have zero prices after wrangler processing!")
                            logger.error(f"  Sample bar_df row: open={bar_df.iloc[0]['open']:.2f}, close={bar_df.iloc[0]['close']:.2f}")
                    
                    # Write Bars & Instrument
                    catalog.write_data([contract])
                    catalog.write_data(bars, skip_disjoint_check=True)
                    
                    # 2. Calculate Greeks
                    if not spot_df.empty:
                        # Merge bars with spot prices
                        # Reset index to merge on timestamp column
                        calc_df = bar_df.reset_index().merge(spot_df.reset_index(), on='timestamp', how='inner')
                        
                        # CRITICAL: Deduplicate after merge (keep last value per timestamp)
                        calc_df = calc_df.drop_duplicates(subset=['timestamp'], keep='last')
                        
                        greeks_list = []
                        
                        for idx, row in calc_df.iterrows():
                            ts = row['timestamp']
                            spot = row['spot_price']
                            price = row['close']
                            
                            # Validate prices (must be positive)
                            if spot <= 0 or price <= 0:
                                continue
                            
                            # Time to expiry in years
                            expiry_ts = pd.Timestamp(expiry_date).tz_localize('UTC') + pd.Timedelta(hours=15, minutes=30) # 15:30 IST approx expiry
                            # Ensure timestamp is tz-aware (UTC)
                            if ts.tzinfo is None:
                                ts = ts.tz_localize('UTC')
                            
                            tte = (expiry_ts - ts).total_seconds() / (365.0 * 24 * 3600)
                            
                            if tte <= 0:
                                continue
                                
                            # Get Risk Free Rate
                            r = ir_provider.get_risk_free_rate(ts.date())
                            
                            # Calculate
                            pricer = OptionPricing(S=spot, K=strike_price, r=r, T=tte)
                            iv = pricer.ImplVolWithBrent(price, opt_type_code)
                            
                            # Skip if IV is at minimum bound (likely invalid price data)
                            if iv <= pricer.IV_LOWER_BOUND + 1e-5:
                                continue
                            
                            # Calculate Greeks using implied volatility
                            delta = pricer.Delta(iv, opt_type_code)
                            gamma = pricer.Gamma(iv)
                            theta = pricer.Theta(iv, opt_type_code)
                            vega = pricer.Vega(iv)
                            rho = pricer.Rho(iv, opt_type_code)
                            
                            # Create OptionGreeks object
                            ts_ns = int(ts.timestamp() * 1_000_000_000)
                            
                            greeks = OptionGreeks(
                                instrument_id=contract.id,
                                iv=iv,
                                delta=delta,
                                gamma=gamma,
                                theta=theta,
                                vega=vega,
                                rho=rho,
                                ts_event=ts_ns,
                                ts_init=ts_ns
                            )
                            greeks_list.append(greeks)
                            
                        if greeks_list:
                            # Write Greeks to Catalog
                            catalog.write_data(greeks_list)
                            total_greeks += len(greeks_list)

                    # 3. Option OI
                    if "oi" in group.columns:
                        oi_data_list = []
                        prev_oi = 0
                        for idx, oi_row in group.iterrows():
                            current_oi = int(oi_row["oi"])
                            coi = current_oi - prev_oi
                            prev_oi = current_oi
                            ts_ns = int(oi_row["timestamp"].timestamp() * 1_000_000_000)

                            oi_data = OptionOI(
                                instrument_id=contract.id,
                                oi=current_oi,
                                coi=coi,
                                ts_event=ts_ns,
                                ts_init=ts_ns
                            )
                            oi_data_list.append(oi_data)

                        if oi_data_list:
                            oi_data_list.sort(key=lambda x: x.ts_init)
                            catalog.write_data(oi_data_list)

                    total_bars += len(bars)
                    
                except Exception as e:
                    logger.warning(f"Error processing option {option_symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error reading {file}: {e}")
            continue

    logger.info(f"✅ {symbol} options: Created {total_bars:,} bars + {total_greeks:,} Greeks records")
    return total_bars


def main():
    parser = argparse.ArgumentParser(
        description="Transform NSE data to Nautilus catalog (Official Pattern)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "original_source" / "raw_data",
        help="Input directory with raw data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "nautilus_data",
        help="Output directory for Nautilus catalog (default: nautilus_data)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["NIFTY", "BANKNIFTY"],
        help="Symbols to transform"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before starting"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["index", "futures", "options"],
        default=["index", "futures", "options"],
        help="Data types to transform"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("NAUTILUS DATA TRANSFORMATION - OFFICIAL PATTERN + GREEKS")
    logger.info("="*80)
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Types: {args.types}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info("="*80)
    
    # Clean output directory if requested
    if args.clean:
        if args.output_dir.exists():
            logger.warning(f"Cleaning output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create catalog
    catalog = ParquetDataCatalog(path=str(args.output_dir))
    
    total_bars = 0
    
    # Transform index data
    if "index" in args.types:
        for symbol in args.symbols:
            try:
                count = transform_index_bars(
                    args.input_dir,
                    catalog,
                    symbol,
                    args.start_date,
                    args.end_date
                )
                total_bars += count
            except Exception as e:
                logger.error(f"Error transforming {symbol} index: {e}", exc_info=True)
    
    # Transform futures data
    if "futures" in args.types:
        for symbol in args.symbols:
            try:
                count, _ = transform_futures_bars(
                    args.input_dir,
                    catalog,
                    symbol,
                    args.start_date,
                    args.end_date
                )
                total_bars += count
            except Exception as e:
                logger.error(f"Error transforming {symbol} futures: {e}", exc_info=True)
    
    # Transform options data
    if "options" in args.types:
        for symbol in args.symbols:
            try:
                count = transform_options_bars(
                    args.input_dir,
                    catalog,
                    symbol,
                    args.start_date,
                    args.end_date
                )
                total_bars += count
            except Exception as e:
                logger.error(f"Error transforming {symbol} options: {e}", exc_info=True)
    
    # Summary
    print("\n" + "="*80)
    print("TRANSFORMATION COMPLETE")
    print("="*80)
    print(f"Total bars created: {total_bars:,}")
    print(f"Catalog location: {args.output_dir}")
    print("="*80)
    print("\nData structure:")
    print(f"  Bar data: {args.output_dir}/bar/")
    print(f"  Greeks data: {args.output_dir}/custom/option_greeks/")
    print(f"  OI data: {args.output_dir}/custom/option_oi/")
    print(f"  Instruments: {args.output_dir}/instrument/")
    print("="*80)


if __name__ == "__main__":
    main()
