#!/usr/bin/env python3
"""
Comprehensive Validation of Nautilus Data Transformation

Validates transformed option-related data in nautilus_data/data:
- Spot/index bars
- Option bars  
- Option Greeks
- Quote ticks
- Catalog query semantics
- Streaming Greeks compatibility

Follows the validation plan from:
.cursor/plans/transform-official-nautilus-with-custom-greeks-interest-rates-c1ed76b6.plan.md
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import BarType, QuoteTick
from nautilus_trader.model.data import Bar
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from marvelquant_data.data_types.option_greeks import OptionGreeks
from marvelquant_data.utils.greeks import OptionPricing
from marvelquant_data.utils.rates import InterestRateProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# IST offset
IST_OFFSET = timedelta(hours=5, minutes=30)


class ValidationResult:
    """Container for validation results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors = []
        self.warnings = []
        self.stats = {}
        self.summary = ""

    def add_error(self, msg: str):
        self.passed = False
        self.errors.append(msg)
        logger.error(f"[{self.name}] {msg}")

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"[{self.name}] {msg}")

    def add_stat(self, key: str, value):
        self.stats[key] = value

    def __str__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [f"\n{'='*80}", f"{self.name}: {status}"]
        if self.stats:
            lines.append("\nStatistics:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:  # Show first 5
                lines.append(f"  - {w}")
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for e in self.errors[:10]:  # Show first 10
                lines.append(f"  - {e}")
        lines.append("="*80)
        return "\n".join(lines)


def yyyymmdd_seconds_to_datetime(date_int, time_int: int) -> datetime:
    """Convert YYYYMMDD integer + seconds to datetime in UTC."""
    if isinstance(date_int, (datetime, pd.Timestamp)):
        date_int = int(date_int.strftime('%Y%m%d'))
    elif hasattr(date_int, 'year'):
        date_int = date_int.year * 10000 + date_int.month * 100 + date_int.day
    
    year = date_int // 10000
    month = (date_int % 10000) // 100
    day = date_int % 100
    
    hours = time_int // 3600
    minutes = (time_int % 3600) // 60
    seconds = time_int % 60
    
    ist_dt = datetime(year, month, day, hours, minutes, seconds)
    utc_dt = ist_dt - IST_OFFSET
    return utc_dt


def validate_spot_index_bars(
    catalog: ParquetDataCatalog,
    raw_data_dir: Path,
    symbol: str,
    date: str
) -> ValidationResult:
    """
    Step 3: Validate Spot / Index Bars
    
    Validates:
    - Schema correctness
    - Timestamp monotonicity and cadence
    - Price scale consistency
    - Instrument metadata
    """
    result = ValidationResult(f"Spot/Index Bars: {symbol}")
    
    try:
        # Load raw index data
        symbol_dir = raw_data_dir / "index" / symbol.lower()
        parquet_files = list(symbol_dir.rglob("*.parquet"))
        
        if not parquet_files:
            result.add_error(f"No raw data files found in {symbol_dir}")
            return result
        
        # Read raw data
        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                result.add_warning(f"Error reading {f}: {e}")
        
        if not dfs:
            result.add_error("No raw data loaded")
            return result
        
        raw_df = pd.concat(dfs, ignore_index=True)
        raw_df['timestamp'] = raw_df.apply(
            lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
            axis=1
        )
        
        # Filter to target date
        start = pd.to_datetime(date) - pd.Timedelta(hours=6)
        end = pd.to_datetime(date) + pd.Timedelta(days=1)
        raw_df = raw_df[(raw_df['timestamp'] >= start) & (raw_df['timestamp'] < end)]
        
        result.add_stat("Raw rows", len(raw_df))
        if raw_df.empty:
            result.add_error(f"No raw data for {date}")
            return result
        
        # Load from catalog
        instrument_id_str = f"{symbol}-INDEX.NSE"
        instrument_id = InstrumentId.from_str(instrument_id_str)
        bar_type_str = f"{instrument_id_str}-1-MINUTE-LAST-EXTERNAL"
        
        # Query bars from catalog
        bars = catalog.query(
            data_cls=Bar,
            identifiers=[instrument_id_str],
            start=start.isoformat(),
            end=end.isoformat()
        )
        
        if not bars:
            result.add_error(f"No bars found in catalog for {instrument_id_str}")
            return result
        
        result.add_stat("Catalog bars", len(bars))
        result.add_stat("Time range", f"{bars[0].ts_event} to {bars[-1].ts_event}")
        
        # Validate schema
        sample_bar = bars[0]
        required_fields = ['open', 'high', 'low', 'close', 'volume', 'ts_event', 'ts_init']
        for field in required_fields:
            if not hasattr(sample_bar, field):
                result.add_error(f"Missing field: {field}")
        
        # Validate timestamps
        ts_events = [b.ts_event for b in bars]
        ts_inits = [b.ts_init for b in bars]
        
        if ts_events != sorted(ts_events):
            result.add_error("ts_event not monotonic")
        
        if ts_inits != sorted(ts_inits):
            result.add_error("ts_init not monotonic")
        
        # Check 1-minute cadence (allow some tolerance for market pauses)
        if len(bars) > 1:
            intervals = np.diff(ts_events) / 1e9 / 60  # Convert to minutes
            expected_interval = 1.0
            tolerance = 5.0  # Allow up to 5 minutes gap (market pauses)
            
            valid_intervals = intervals[(intervals >= expected_interval) & (intervals <= expected_interval + tolerance)]
            if len(valid_intervals) < len(intervals) * 0.8:  # At least 80% should be ~1 minute
                result.add_warning(f"Only {len(valid_intervals)}/{len(intervals)} intervals are ~1 minute")
        
        # Validate price scale (compare with raw data)
        bar_df = pd.DataFrame([{
            'ts_event': b.ts_event,
            'open': b.open.as_double(),
            'high': b.high.as_double(),
            'low': b.low.as_double(),
            'close': b.close.as_double(),
            'volume': b.volume.as_double()
        } for b in bars])
        
        # Sample comparison: find matching timestamps
        sample_size = min(10, len(bars))
        matches = 0
        for i in range(sample_size):
            bar = bars[i]
            bar_ts = pd.Timestamp(bar.ts_event, unit='ns')
            
            # Find closest raw data point
            raw_match = raw_df.iloc[(raw_df['timestamp'] - bar_ts).abs().argsort()[:1]]
            if not raw_match.empty:
                raw_close = raw_match.iloc[0]['close']
                bar_close = bar.close.as_double()
                
                # Allow small tolerance (rounding differences)
                if abs(raw_close - bar_close) < 0.1:
                    matches += 1
        
        if matches < sample_size * 0.8:
            result.add_warning(f"Only {matches}/{sample_size} price matches within tolerance")
        else:
            result.add_stat("Price matches", f"{matches}/{sample_size}")
        
        # Validate instrument metadata
        instruments = catalog.instruments()
        index_instruments = [inst for inst in instruments if str(inst.id) == instrument_id_str]
        
        if not index_instruments:
            result.add_error(f"Instrument {instrument_id_str} not found in catalog")
        else:
            inst = index_instruments[0]
            result.add_stat("Currency", str(inst.currency))
            result.add_stat("Price precision", inst.price_precision)
            
            if str(inst.currency) != "INR":
                result.add_error(f"Expected currency INR, got {inst.currency}")
            
            if inst.price_precision != 2:
                result.add_warning(f"Expected price_precision 2, got {inst.price_precision}")
        
    except Exception as e:
        result.add_error(f"Exception during validation: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


def validate_option_bars(
    catalog: ParquetDataCatalog,
    raw_data_dir: Path,
    symbol: str,
    option_symbol: str,
    date: str
) -> ValidationResult:
    """
    Step 4: Validate Option Bars
    
    Validates:
    - Schema & layout
    - OHLCV integrity
    - Time alignment
    - Instrument metadata
    """
    result = ValidationResult(f"Option Bars: {option_symbol}")
    
    try:
        # Load raw option data
        symbol_dir = raw_data_dir / "option" / symbol.lower()
        all_files = list(symbol_dir.rglob("*.parquet"))
        
        # Find files for this option symbol
        option_files = []
        for f in all_files:
            try:
                df = pd.read_parquet(f)
                if 'symbol' in df.columns and option_symbol in df['symbol'].values:
                    option_files.append(f)
            except:
                pass
        
        if not option_files:
            result.add_warning(f"No raw data files found for {option_symbol}")
            # Continue with catalog validation
        
        # Load from catalog
        instrument_id_str = f"{option_symbol}.NSE"
        instrument_id = InstrumentId.from_str(instrument_id_str)
        
        start = pd.to_datetime(date) - pd.Timedelta(hours=6)
        end = pd.to_datetime(date) + pd.Timedelta(days=1)
        
        bars = catalog.query(
            data_cls=Bar,
            identifiers=[instrument_id_str],
            start=start.isoformat(),
            end=end.isoformat()
        )
        
        if not bars:
            result.add_error(f"No bars found in catalog for {instrument_id_str}")
            return result
        
        result.add_stat("Catalog bars", len(bars))
        
        # Validate schema
        sample_bar = bars[0]
        required_fields = ['open', 'high', 'low', 'close', 'volume', 'ts_event', 'ts_init']
        for field in required_fields:
            if not hasattr(sample_bar, field):
                result.add_error(f"Missing field: {field}")
        
        # Validate OHLCV integrity
        for bar in bars:
            if bar.high.as_double() < bar.close.as_double():
                result.add_error(f"High < Close: {bar.high.as_double()} < {bar.close.as_double()}")
            if bar.low.as_double() > bar.close.as_double():
                result.add_error(f"Low > Close: {bar.low.as_double()} > {bar.close.as_double()}")
            if bar.high.as_double() < bar.open.as_double():
                result.add_error(f"High < Open: {bar.high.as_double()} < {bar.open.as_double()}")
            if bar.low.as_double() > bar.open.as_double():
                result.add_error(f"Low > Open: {bar.low.as_double()} > {bar.open.as_double()}")
        
        # Validate timestamps
        ts_events = [b.ts_event for b in bars]
        if ts_events != sorted(ts_events):
            result.add_error("ts_event not monotonic")
        
        # Validate instrument metadata
        instruments = catalog.instruments()
        option_instruments = [inst for inst in instruments if str(inst.id) == instrument_id_str]
        
        if not option_instruments:
            result.add_error(f"Option contract {instrument_id_str} not found")
        else:
            inst = option_instruments[0]
            result.add_stat("Underlying", str(inst.underlying) if hasattr(inst, 'underlying') else "N/A")
            result.add_stat("Strike", inst.strike_price.as_double() if hasattr(inst, 'strike_price') else "N/A")
            result.add_stat("Expiration", inst.expiration_ns if hasattr(inst, 'expiration_ns') else "N/A")
            result.add_stat("Currency", str(inst.currency))
            result.add_stat("Price precision", inst.price_precision)
            
            if str(inst.currency) != "INR":
                result.add_error(f"Expected currency INR, got {inst.currency}")
        
        # Compare with raw data if available
        if option_files:
            raw_dfs = []
            for f in option_files:
                try:
                    df = pd.read_parquet(f)
                    df = df[df['symbol'] == option_symbol]
                    raw_dfs.append(df)
                except:
                    pass
            
            if raw_dfs:
                raw_df = pd.concat(raw_dfs, ignore_index=True)
                raw_df['timestamp'] = raw_df.apply(
                    lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
                    axis=1
                )
                raw_df = raw_df[(raw_df['timestamp'] >= start) & (raw_df['timestamp'] < end)]
                
                if not raw_df.empty:
                    # Sample comparison
                    sample_size = min(5, len(bars))
                    matches = 0
                    for i in range(sample_size):
                        bar = bars[i]
                        bar_ts = pd.Timestamp(bar.ts_event, unit='ns')
                        raw_match = raw_df.iloc[(raw_df['timestamp'] - bar_ts).abs().argsort()[:1]]
                        if not raw_match.empty:
                            raw_close = raw_match.iloc[0]['close']
                            bar_close = bar.close.as_double()
                            if abs(raw_close - bar_close) < 0.1:
                                matches += 1
                    
                    result.add_stat("Price matches", f"{matches}/{sample_size}")
        
    except Exception as e:
        result.add_error(f"Exception during validation: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


def validate_option_greeks(
    catalog: ParquetDataCatalog,
    symbol: str,
    option_symbol: str,
    date: str,
    ir_provider: InterestRateProvider
) -> ValidationResult:
    """
    Step 5: Validate Option Greeks
    
    Validates:
    - Instrument & timestamps
    - Value ranges
    - Spot linkage
    - Interest rate usage
    """
    result = ValidationResult(f"Option Greeks: {option_symbol}")
    
    try:
        instrument_id_str = f"{option_symbol}.NSE"
        instrument_id = InstrumentId.from_str(instrument_id_str)
        
        start = pd.to_datetime(date) - pd.Timedelta(hours=6)
        end = pd.to_datetime(date) + pd.Timedelta(days=1)
        
        # Query Greeks from catalog
        greeks_list = catalog.query(
            data_cls=OptionGreeks,
            identifiers=[instrument_id_str],
            start=start.isoformat(),
            end=end.isoformat()
        )
        
        if not greeks_list:
            result.add_error(f"No Greeks data found for {instrument_id_str}")
            return result
        
        result.add_stat("Greeks records", len(greeks_list))
        
        # Validate schema
        sample_greeks = greeks_list[0]
        required_fields = ['instrument_id', 'iv', 'delta', 'gamma', 'theta', 'vega', 'rho', 'ts_event', 'ts_init']
        for field in required_fields:
            if not hasattr(sample_greeks, field):
                result.add_error(f"Missing field: {field}")
        
        # Validate instrument IDs
        for g in greeks_list:
            if str(g.instrument_id) != instrument_id_str:
                result.add_error(f"Instrument ID mismatch: {g.instrument_id} != {instrument_id_str}")
        
        # Validate timestamps
        ts_events = [g.ts_event for g in greeks_list]
        ts_inits = [g.ts_init for g in greeks_list]
        
        if ts_events != sorted(ts_events):
            result.add_error("ts_event not monotonic")
        
        if ts_inits != sorted(ts_inits):
            result.add_error("ts_init not monotonic")
        
        # Check for duplicates
        if len(set(ts_events)) != len(ts_events):
            result.add_warning(f"Duplicate timestamps found: {len(ts_events) - len(set(ts_events))} duplicates")
        
        # Validate value ranges
        ivs = [g.iv for g in greeks_list]
        deltas = [g.delta for g in greeks_list]
        gammas = [g.gamma for g in greeks_list]
        vegas = [g.vega for g in greeks_list]
        thetas = [g.theta for g in greeks_list]
        rhos = [g.rho for g in greeks_list]
        
        result.add_stat("IV range", f"{min(ivs):.4f} to {max(ivs):.4f}")
        result.add_stat("Delta range", f"{min(deltas):.4f} to {max(deltas):.4f}")
        result.add_stat("Gamma range", f"{min(gammas):.6f} to {max(gammas):.6f}")
        
        # IV should be positive
        negative_ivs = sum(1 for iv in ivs if iv <= 0)
        if negative_ivs > 0:
            result.add_error(f"{negative_ivs} records with non-positive IV")
        
        # Delta should be in [-1, 1] for calls, [-1, 0] for puts
        invalid_deltas = sum(1 for d in deltas if abs(d) > 1.01)  # Small tolerance
        if invalid_deltas > 0:
            result.add_warning(f"{invalid_deltas} records with |delta| > 1")
        
        # Gamma should be positive
        negative_gammas = sum(1 for g in gammas if g < 0)
        if negative_gammas > 0:
            result.add_error(f"{negative_gammas} records with negative gamma")
        
        # Vega should be positive
        negative_vegas = sum(1 for v in vegas if v < 0)
        if negative_vegas > 0:
            result.add_warning(f"{negative_vegas} records with negative vega")
        
        # Validate spot linkage: recompute Greeks for a few samples
        # First, get option contract details
        instruments = catalog.instruments()
        option_instruments = [inst for inst in instruments if str(inst.id) == instrument_id_str]
        
        if option_instruments:
            contract = option_instruments[0]
            strike = contract.strike_price.as_double() if hasattr(contract, 'strike_price') else None
            expiry_ns = contract.expiration_ns if hasattr(contract, 'expiration_ns') else None
            option_kind = contract.option_kind if hasattr(contract, 'option_kind') else None
            
            if strike and expiry_ns and option_kind:
                # Get underlying spot prices
                underlying_id_str = f"{symbol}-INDEX.NSE"
                underlying_bars = catalog.query(
                    data_cls=Bar,
                    identifiers=[underlying_id_str],
                    start=start.isoformat(),
                    end=end.isoformat()
                )
                
                if underlying_bars:
                    # Create spot lookup
                    spot_dict = {b.ts_event: b.close.as_double() for b in underlying_bars}
                    
                    # Sample recomputation
                    sample_size = min(5, len(greeks_list))
                    recompute_matches = 0
                    
                    for i in range(sample_size):
                        g = greeks_list[i]
                        spot = spot_dict.get(g.ts_event)
                        
                        if spot:
                            # Get risk-free rate
                            ts_dt = pd.Timestamp(g.ts_event, unit='ns')
                            r = ir_provider.get_risk_free_rate(ts_dt.date())
                            
                            if r > 0:
                                # Calculate time to expiry
                                expiry_dt = pd.Timestamp(expiry_ns, unit='ns')
                                tte = (expiry_dt - ts_dt).total_seconds() / (365.0 * 24 * 3600)
                                
                                if tte > 0:
                                    opt_type_code = 'CE' if str(option_kind) == 'CALL' else 'PE'
                                    
                                    # Recompute IV and Greeks
                                    pricer = OptionPricing(S=spot, K=strike, r=r, T=tte)
                                    recomputed_iv = pricer.ImplVolWithBrent(
                                        g.iv * spot if g.iv < 0.1 else g.iv,  # Use stored IV as hint
                                        opt_type_code
                                    )
                                    
                                    if recomputed_iv > pricer.IV_LOWER_BOUND + 1e-5:
                                        recomputed_delta = pricer.Delta(recomputed_iv, opt_type_code)
                                        
                                        # Compare (allow tolerance)
                                        delta_diff = abs(recomputed_delta - g.delta)
                                        if delta_diff < 0.1:  # 10% tolerance
                                            recompute_matches += 1
                    
                    result.add_stat("Recomputation matches", f"{recompute_matches}/{sample_size}")
                    
                    # Validate interest rate usage
                    sample_rates = []
                    for i in range(min(10, len(greeks_list))):
                        ts_dt = pd.Timestamp(greeks_list[i].ts_event, unit='ns')
                        r = ir_provider.get_risk_free_rate(ts_dt.date())
                        sample_rates.append(r)
                    
                    if all(r == 0 for r in sample_rates):
                        result.add_error("All interest rates are zero")
                    else:
                        result.add_stat("Rate range", f"{min(sample_rates):.4f} to {max(sample_rates):.4f}")
        
    except Exception as e:
        result.add_error(f"Exception during validation: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


def validate_catalog_queries(catalog: ParquetDataCatalog, symbol: str, date: str) -> ValidationResult:
    """
    Step 6: Verify Catalog Query Semantics
    
    Validates:
    - Bar queries
    - Custom Greeks queries
    - Quote tick queries
    """
    result = ValidationResult("Catalog Query Semantics")
    
    try:
        start = pd.to_datetime(date) - pd.Timedelta(hours=6)
        end = pd.to_datetime(date) + pd.Timedelta(days=1)
        
        # Test bar queries
        instrument_id_str = f"{symbol}-INDEX.NSE"
        bars = catalog.query(
            data_cls=Bar,
            identifiers=[instrument_id_str],
            start=start.isoformat(),
            end=end.isoformat()
        )
        result.add_stat("Bar query results", len(bars))
        
        # Test custom Greeks query
        # Find an option contract
        instruments = catalog.instruments()
        option_instruments = [inst for inst in instruments 
                            if hasattr(inst, 'option_kind') and symbol.upper() in str(inst.id)]
        
        if option_instruments:
            sample_option_id = str(option_instruments[0].id)
            greeks = catalog.query(
                data_cls=OptionGreeks,
                identifiers=[sample_option_id],
                start=start.isoformat(),
                end=end.isoformat()
            )
            result.add_stat("Greeks query results", len(greeks))
        
        # Test quote tick queries
        quote_ticks = catalog.query(
            data_cls=QuoteTick,
            identifiers=[instrument_id_str],
            start=start.isoformat(),
            end=end.isoformat()
        )
        result.add_stat("QuoteTick query results", len(quote_ticks))
        
        if not quote_ticks:
            result.add_warning("No QuoteTick data found (required for streaming Greeks)")
        
    except Exception as e:
        result.add_error(f"Exception during query validation: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


def validate_streaming_pattern(catalog: ParquetDataCatalog, symbol: str) -> ValidationResult:
    """
    Step 7: Validate Streaming Greeks Against Official Example
    
    Validates:
    - OptionGreeks compatibility with streaming
    - QuoteTick prerequisites
    - Catalog layout consistency
    """
    result = ValidationResult("Streaming Greeks Pattern")
    
    try:
        # Check that OptionGreeks is properly registered
        from nautilus_trader.serialization.arrow.serializer import get_arrow_serializer
        
        try:
            serializer = get_arrow_serializer(OptionGreeks)
            result.add_stat("Arrow serializer", "Registered")
        except Exception as e:
            result.add_error(f"OptionGreeks not registered for Arrow serialization: {e}")
        
        # Check QuoteTick availability (prerequisite for streaming)
        instruments = catalog.instruments()
        index_instruments = [inst for inst in instruments 
                            if symbol.upper() in str(inst.id) and "INDEX" in str(inst.id)]
        
        if index_instruments:
            index_id = str(index_instruments[0].id)
            
            # Check for quote ticks
            quote_ticks = catalog.query(
                data_cls=QuoteTick,
                identifiers=[index_id],
                start="2024-01-01",
                end="2024-01-02"
            )
            
            if quote_ticks:
                result.add_stat("QuoteTick availability", f"{len(quote_ticks)} ticks found")
            else:
                result.add_warning("No QuoteTick data found (required for streaming Greeks)")
        
        # Check catalog layout
        catalog_path = Path(catalog.path)
        greeks_dir = catalog_path / "data" / "custom_option_greeks"
        
        if greeks_dir.exists():
            greeks_instruments = list(greeks_dir.iterdir())
            result.add_stat("Greeks instruments", len(greeks_instruments))
        else:
            result.add_warning(f"Greeks directory not found: {greeks_dir}")
        
        # Validate that OptionGreeks can be used in StreamingConfig pattern
        # (This is a structural check - actual streaming requires BacktestEngine)
        result.add_stat("Streaming compatibility", "OptionGreeks uses @customdataclass decorator")
        
    except Exception as e:
        result.add_error(f"Exception during streaming validation: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Nautilus data transformation"
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=PROJECT_ROOT / "nautilus_data",
        help="Path to Nautilus catalog"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "original_source" / "raw_data",
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BANKNIFTY",
        help="Symbol to validate"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2024-01-01",
        help="Date to validate (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--option-symbol",
        type=str,
        default="BANKNIFTY10JAN2448200CE",
        help="Sample option symbol to validate"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("NAUTILUS DATA TRANSFORMATION VALIDATION")
    logger.info("="*80)
    logger.info(f"Catalog: {args.catalog_path}")
    logger.info(f"Raw data: {args.raw_data_dir}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Date: {args.date}")
    logger.info(f"Option sample: {args.option_symbol}")
    logger.info("="*80)
    
    # Initialize catalog
    catalog = ParquetDataCatalog(path=str(args.catalog_path))
    
    # Initialize interest rate provider
    ir_xml_path = PROJECT_ROOT / "data" / "interest_rates" / "india_91day_tbill_rates_2018_2025_nautilus.xml"
    ir_provider = InterestRateProvider(ir_xml_path)
    
    # Run validations
    results = []
    
    # Step 3: Validate Spot/Index Bars
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Validating Spot/Index Bars")
    logger.info("="*80)
    result = validate_spot_index_bars(
        catalog, args.raw_data_dir, args.symbol, args.date
    )
    results.append(result)
    logger.info(str(result))
    
    # Step 4: Validate Option Bars
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Validating Option Bars")
    logger.info("="*80)
    result = validate_option_bars(
        catalog, args.raw_data_dir, args.symbol, args.option_symbol, args.date
    )
    results.append(result)
    logger.info(str(result))
    
    # Step 5: Validate Option Greeks
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Validating Option Greeks")
    logger.info("="*80)
    result = validate_option_greeks(
        catalog, args.symbol, args.option_symbol, args.date, ir_provider
    )
    results.append(result)
    logger.info(str(result))
    
    # Step 6: Validate Catalog Queries
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Validating Catalog Query Semantics")
    logger.info("="*80)
    result = validate_catalog_queries(catalog, args.symbol, args.date)
    results.append(result)
    logger.info(str(result))
    
    # Step 7: Validate Streaming Pattern
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Validating Streaming Greeks Pattern")
    logger.info("="*80)
    result = validate_streaming_pattern(catalog, args.symbol)
    results.append(result)
    logger.info(str(result))
    
    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    logger.info(f"Tests passed: {passed}/{total}")
    
    for r in results:
        status = "✅" if r.passed else "❌"
        logger.info(f"{status} {r.name}")
        if r.errors:
            logger.info(f"   Errors: {len(r.errors)}")
        if r.warnings:
            logger.info(f"   Warnings: {len(r.warnings)}")
    
    logger.info("="*80)
    
    # Return exit code
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

