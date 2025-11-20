#!/usr/bin/env python3
"""
Validate multiple strikes at multiple times against StockMock values.

Reads data from Nautilus parquet catalog and validates:
- Index spot price
- Option prices (Call/Put LTP)
- Option deltas
- Implied volatility
- Option Greeks (especially Gamma)
"""

import sys
from pathlib import Path
from datetime import datetime
import pytz
import pandas as pd
from typing import Optional, Dict, List
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "nautilus_trader"))

try:
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
except ImportError:
    try:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog
    except ImportError as e:
        print(f"ERROR: Cannot import ParquetDataCatalog: {e}")
        print("Please ensure nautilus_trader is properly installed")
        sys.exit(1)

# StockMock reference values
STOCKMOCK_DATA = {
    "2024-01-01T09:30:00": {
        "spot": 48183.4,
        "strikes": {
            48200: {
                "call_ltp": 274.1,
                "call_delta": 0.52,
                "put_ltp": 253.6,
                "put_delta": -0.48,
                "iv": 17.5,
            }
        }
    },
    "2024-01-01T12:00:00": {
        "spot": 48325.9,
        "strikes": {
            47200: {"call_ltp": 1218, "call_delta": 0.95, "put_ltp": 13.6, "put_delta": -0.05, "iv": 19.5},
            47300: {"call_ltp": 1110, "call_delta": 0.94, "put_ltp": 16.6, "put_delta": -0.06, "iv": 18.9},
            47400: {"call_ltp": 1019.6, "call_delta": 0.93, "put_ltp": 20.9, "put_delta": -0.07, "iv": 18.5},
            47500: {"call_ltp": 926.1, "call_delta": 0.91, "put_ltp": 27.4, "put_delta": -0.09, "iv": 18.2},
            47600: {"call_ltp": 832, "call_delta": 0.89, "put_ltp": 36, "put_delta": -0.11, "iv": 18.0},
            47700: {"call_ltp": 741.1, "call_delta": 0.86, "put_ltp": 47.3, "put_delta": -0.14, "iv": 17.8},
            47800: {"call_ltp": 662, "call_delta": 0.83, "put_ltp": 61.3, "put_delta": -0.17, "iv": 17.6},
            47900: {"call_ltp": 580, "call_delta": 0.78, "put_ltp": 80.5, "put_delta": -0.22, "iv": 17.5},
            48000: {"call_ltp": 506, "call_delta": 0.74, "put_ltp": 104.5, "put_delta": -0.26, "iv": 17.5},
            48100: {"call_ltp": 436, "call_delta": 0.68, "put_ltp": 133.3, "put_delta": -0.32, "iv": 17.4},
            48200: {"call_ltp": 370.8, "call_delta": 0.63, "put_ltp": 166.9, "put_delta": -0.37, "iv": 17.3},
            48300: {"call_ltp": 310.1, "call_delta": 0.57, "put_ltp": 206.8, "put_delta": -0.43, "iv": 17.3},
            48400: {"call_ltp": 256.3, "call_delta": 0.5, "put_ltp": 252.7, "put_delta": -0.5, "iv": 17.2},
            48500: {"call_ltp": 207.7, "call_delta": 0.44, "put_ltp": 304.9, "put_delta": -0.56, "iv": 17.1},
            48600: {"call_ltp": 167.1, "call_delta": 0.38, "put_ltp": 364.1, "put_delta": -0.62, "iv": 17.1},
            48700: {"call_ltp": 132.5, "call_delta": 0.32, "put_ltp": 429.6, "put_delta": -0.68, "iv": 17.1},
            48800: {"call_ltp": 103.2, "call_delta": 0.27, "put_ltp": 501.6, "put_delta": -0.73, "iv": 17.1},
            48900: {"call_ltp": 80, "call_delta": 0.22, "put_ltp": 577.1, "put_delta": -0.78, "iv": 17.1},
            49000: {"call_ltp": 61.4, "call_delta": 0.18, "put_ltp": 660.4, "put_delta": -0.82, "iv": 17.2},
            49100: {"call_ltp": 46.8, "call_delta": 0.14, "put_ltp": 749.4, "put_delta": -0.86, "iv": 17.3},
            49200: {"call_ltp": 35.8, "call_delta": 0.11, "put_ltp": 836.3, "put_delta": -0.89, "iv": 17.5},
            49300: {"call_ltp": 27.6, "call_delta": 0.09, "put_ltp": 929, "put_delta": -0.91, "iv": 17.8},
            49400: {"call_ltp": 21.3, "call_delta": 0.07, "put_ltp": 1015.2, "put_delta": -0.93, "iv": 18.1},
        }
    },
    "2024-01-01T15:00:00": {
        "spot": 48404.8,
        "strikes": {
            48200: {"call_ltp": 490.1, "call_delta": 0.73, "put_ltp": 105.8, "put_delta": -0.27, "iv": 17.6},
            48400: {"call_ltp": 348.7, "call_delta": 0.62, "put_ltp": 163.1, "put_delta": -0.38, "iv": 17.0},
        }
    }
}

# Catalog paths
CATALOG_PATH = project_root / "nautilus_data"

# Instrument base
UNDERLYING = "BANKNIFTY"
EXPIRY = "03JAN24"  # Format: DDMMMYY

# Time conversion
IST = pytz.timezone("Asia/Kolkata")


def timestamp_to_nanos(dt_str: str) -> int:
    """Convert ISO timestamp string to Unix nanoseconds."""
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return int(dt.timestamp() * 1e9)


def nanos_to_datetime(nanos: int) -> datetime:
    """Convert Unix nanoseconds to datetime."""
    return datetime.fromtimestamp(nanos / 1e9, tz=IST)


def get_instrument_ids(strike: int) -> tuple:
    """Generate instrument IDs for call and put."""
    call_id = f"{UNDERLYING}{EXPIRY}{strike}CE.NSE"
    put_id = f"{UNDERLYING}{EXPIRY}{strike}PE.NSE"
    return call_id, put_id


def find_closest_bar(bars: list, target_nanos: int, window_seconds: int = 60, expected_range: tuple = None) -> Optional[dict]:
    """Find the bar closest to target timestamp within window.
    
    Args:
        bars: List of bar objects
        target_nanos: Target timestamp in nanoseconds
        window_seconds: Time window in seconds
        expected_range: Optional tuple (min, max) for expected price range to filter outliers
    """
    if not bars:
        return None
    
    window_nanos = window_seconds * 1_000_000_000
    candidates = []
    
    for bar in bars:
        ts = bar.ts_event
        diff = abs(ts - target_nanos)
        if diff < window_nanos:
            close_price = float(bar.close)
            # Filter out obviously wrong values (e.g., prices that are 100x too large)
            if expected_range:
                min_price, max_price = expected_range
                if close_price < min_price or close_price > max_price:
                    continue
            candidates.append((diff, bar))
    
    if not candidates:
        return None
    
    # Sort by time difference and pick closest
    candidates.sort(key=lambda x: x[0])
    closest = candidates[0][1]
    
    return {
        "ts_event": closest.ts_event,
        "ts_event_dt": nanos_to_datetime(closest.ts_event),
        "open": float(closest.open),
        "high": float(closest.high),
        "low": float(closest.low),
        "close": float(closest.close),
        "volume": int(closest.volume),
    }


def read_custom_greeks(catalog: ParquetDataCatalog, instrument_id: str, target_nanos: int) -> Optional[dict]:
    """Read custom option greeks data using catalog."""
    try:
        # Try to read custom data - need to import OptionGreeks class
        OptionGreeks = None
        import_paths = [
            "nautilus_trader.data.custom.option_greeks",
            "data.custom.option_greeks",
            "marvelquant.data.custom.option_greeks",
        ]
        
        for import_path in import_paths:
            try:
                module = __import__(import_path, fromlist=["OptionGreeks"])
                OptionGreeks = getattr(module, "OptionGreeks")
                break
            except (ImportError, AttributeError):
                continue
        
        if OptionGreeks is None:
            return None
        
        custom_data = catalog.custom_data(
            cls=OptionGreeks,
            instrument_ids=[instrument_id],
            start=target_nanos - 60_000_000_000,  # 1 minute before
            end=target_nanos + 60_000_000_000,  # 1 minute after
        )
        
        if not custom_data:
            return None
        
        # Find closest to target
        closest = None
        min_diff = float("inf")
        for greek_wrapper in custom_data:
            diff = abs(greek_wrapper.ts_event - target_nanos)
            if diff < min_diff:
                min_diff = diff
                closest = greek_wrapper
        
        if closest:
            greek_data = closest.data
            return {
                "ts_event": closest.ts_event,
                "ts_event_dt": nanos_to_datetime(closest.ts_event),
                "delta": float(greek_data.delta),
                "gamma": float(greek_data.gamma),
                "theta": float(greek_data.theta),
                "vega": float(greek_data.vega),
                "iv": float(greek_data.iv),
                "rho": float(greek_data.rho) if hasattr(greek_data, "rho") else None,
            }
    except Exception as e:
        print(f"      Error reading greeks: {e}")
    
    return None


def validate_strike_at_time(
    catalog: ParquetDataCatalog,
    strike: int,
    timestamp_str: str,
    stockmock_ref: dict,
) -> dict:
    """Validate a single strike at a specific time."""
    target_nanos = timestamp_to_nanos(timestamp_str)
    call_id, put_id = get_instrument_ids(strike)
    
    result = {
        "strike": strike,
        "timestamp": timestamp_str,
        "spot": None,
        "call_ltp": None,
        "put_ltp": None,
        "call_delta": None,
        "put_delta": None,
        "call_gamma": None,
        "put_gamma": None,
        "call_iv": None,
        "put_iv": None,
        "errors": [],
    }
    
    # Get spot price
    try:
        bar_type_str = "BANKNIFTY-INDEX.NSE-1-MINUTE-LAST-EXTERNAL"
        index_bars = catalog.bars(
            bar_types=[bar_type_str],
            start=target_nanos - 60_000_000_000,
            end=target_nanos + 60_000_000_000,
        )
        # Expected BANKNIFTY spot range: 40000-60000
        index_bar = find_closest_bar(index_bars, target_nanos, expected_range=(40000, 60000))
        if index_bar:
            result["spot"] = index_bar["close"]
    except Exception as e:
        result["errors"].append(f"Spot: {e}")
    
    # Get call bar
    try:
        call_bar_type_str = f"{call_id}-1-MINUTE-LAST-EXTERNAL"
        call_bars = catalog.bars(
            bar_types=[call_bar_type_str],
            start=target_nanos - 60_000_000_000,
            end=target_nanos + 60_000_000_000,
        )
        call_bar = find_closest_bar(call_bars, target_nanos)
        if call_bar:
            result["call_ltp"] = call_bar["close"]
    except Exception as e:
        result["errors"].append(f"Call bar: {e}")
    
    # Get put bar
    try:
        put_bar_type_str = f"{put_id}-1-MINUTE-LAST-EXTERNAL"
        put_bars = catalog.bars(
            bar_types=[put_bar_type_str],
            start=target_nanos - 60_000_000_000,
            end=target_nanos + 60_000_000_000,
        )
        put_bar = find_closest_bar(put_bars, target_nanos)
        if put_bar:
            result["put_ltp"] = put_bar["close"]
    except Exception as e:
        result["errors"].append(f"Put bar: {e}")
    
    # Get call greeks
    call_greeks = read_custom_greeks(catalog, call_id, target_nanos)
    if call_greeks:
        result["call_delta"] = call_greeks["delta"]
        result["call_gamma"] = call_greeks["gamma"]
        result["call_iv"] = call_greeks["iv"]
    
    # Get put greeks
    put_greeks = read_custom_greeks(catalog, put_id, target_nanos)
    if put_greeks:
        result["put_delta"] = put_greeks["delta"]
        result["put_gamma"] = put_greeks["gamma"]
        result["put_iv"] = put_greeks["iv"]
    
    return result


def print_validation_results(results: List[dict], stockmock_ref: dict):
    """Print validation results in a formatted table."""
    print(f"\n{'Strike':<8} {'Metric':<15} {'Our Value':<15} {'StockMock':<15} {'Diff':<12} {'Status':<8}")
    print("-" * 80)
    
    for result in results:
        strike = result["strike"]
        ref = stockmock_ref["strikes"].get(strike, {})
        
        # Spot
        if result["spot"] is not None:
            ref_spot = stockmock_ref.get("spot", 0)
            diff = abs(result["spot"] - ref_spot)
            status = "✓" if diff < 1.0 else "✗"
            print(f"{strike:<8} {'Spot':<15} {result['spot']:<15.2f} {ref_spot:<15.2f} {diff:<12.2f} {status:<8}")
        
        # Call LTP
        if result["call_ltp"] is not None:
            ref_call_ltp = ref.get("call_ltp", 0)
            diff = abs(result["call_ltp"] - ref_call_ltp)
            status = "✓" if diff < 5.0 else "✗"
            print(f"{strike:<8} {'Call LTP':<15} {result['call_ltp']:<15.2f} {ref_call_ltp:<15.2f} {diff:<12.2f} {status:<8}")
        
        # Put LTP
        if result["put_ltp"] is not None:
            ref_put_ltp = ref.get("put_ltp", 0)
            diff = abs(result["put_ltp"] - ref_put_ltp)
            status = "✓" if diff < 5.0 else "✗"
            print(f"{strike:<8} {'Put LTP':<15} {result['put_ltp']:<15.2f} {ref_put_ltp:<15.2f} {diff:<12.2f} {status:<8}")
        
        # Call Delta
        if result["call_delta"] is not None:
            ref_call_delta = ref.get("call_delta", 0)
            diff = abs(result["call_delta"] - ref_call_delta)
            status = "✓" if diff < 0.05 else "✗"
            print(f"{strike:<8} {'Call Delta':<15} {result['call_delta']:<15.4f} {ref_call_delta:<15.4f} {diff:<12.4f} {status:<8}")
        
        # Put Delta
        if result["put_delta"] is not None:
            ref_put_delta = ref.get("put_delta", 0)
            diff = abs(result["put_delta"] - ref_put_delta)
            status = "✓" if diff < 0.05 else "✗"
            print(f"{strike:<8} {'Put Delta':<15} {result['put_delta']:<15.4f} {ref_put_delta:<15.4f} {diff:<12.4f} {status:<8}")
        
        # Gamma
        if result["call_gamma"] is not None and result["put_gamma"] is not None:
            gamma_diff = abs(result["call_gamma"] - result["put_gamma"])
            status = "✓" if gamma_diff < 0.0001 else "✗"
            print(f"{strike:<8} {'Gamma Diff':<15} {gamma_diff:<15.6f} {'0.0000':<15} {gamma_diff:<12.6f} {status:<8}")
        
        # IV
        if result["call_iv"] is not None:
            ref_iv = ref.get("iv", 0)
            call_iv_pct = result["call_iv"] * 100
            ref_iv_pct = ref_iv
            diff = abs(call_iv_pct - ref_iv_pct)
            status = "✓" if diff < 1.0 else "✗"
            print(f"{strike:<8} {'Call IV %':<15} {call_iv_pct:<15.2f} {ref_iv_pct:<15.2f} {diff:<12.2f} {status:<8}")


def main():
    """Main validation function."""
    print("=" * 80)
    print("VALIDATION: Multiple Strikes at Multiple Times")
    print("=" * 80)
    
    # Initialize catalog
    try:
        catalog = ParquetDataCatalog(str(CATALOG_PATH))
        print("✓ Using Nautilus ParquetDataCatalog")
    except Exception as e:
        print(f"ERROR: Could not initialize catalog: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    all_results = {}
    
    # Validate each timestamp
    for timestamp_str, stockmock_ref in STOCKMOCK_DATA.items():
        print("\n" + "=" * 80)
        print(f"VALIDATING: {timestamp_str}")
        print("=" * 80)
        
        target_dt = nanos_to_datetime(timestamp_to_nanos(timestamp_str))
        print(f"Target Time: {target_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"StockMock Spot: {stockmock_ref.get('spot', 'N/A')}")
        print(f"Strikes to validate: {len(stockmock_ref.get('strikes', {}))}")
        
        timestamp_results = []
        strikes = sorted(stockmock_ref.get("strikes", {}).keys())
        
        for strike in strikes:
            print(f"\n  Validating strike {strike}...")
            result = validate_strike_at_time(catalog, strike, timestamp_str, stockmock_ref)
            timestamp_results.append(result)
            
            if result["errors"]:
                print(f"    ⚠️  Errors: {', '.join(result['errors'])}")
            else:
                print(f"    ✓ Data retrieved")
        
        all_results[timestamp_str] = timestamp_results
        
        # Print validation table for this timestamp
        print_validation_results(timestamp_results, stockmock_ref)
    
    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    total_validations = 0
    passed_validations = 0
    
    for timestamp_str, results in all_results.items():
        print(f"\n{timestamp_str}:")
        for result in results:
            strike = result["strike"]
            ref = STOCKMOCK_DATA[timestamp_str]["strikes"].get(strike, {})
            
            # Count validations
            if result["spot"] is not None:
                total_validations += 1
                if abs(result["spot"] - STOCKMOCK_DATA[timestamp_str].get("spot", 0)) < 1.0:
                    passed_validations += 1
            
            if result["call_ltp"] is not None:
                total_validations += 1
                if abs(result["call_ltp"] - ref.get("call_ltp", 0)) < 5.0:
                    passed_validations += 1
            
            if result["put_ltp"] is not None:
                total_validations += 1
                if abs(result["put_ltp"] - ref.get("put_ltp", 0)) < 5.0:
                    passed_validations += 1
            
            if result["call_delta"] is not None:
                total_validations += 1
                if abs(result["call_delta"] - ref.get("call_delta", 0)) < 0.05:
                    passed_validations += 1
            
            if result["put_delta"] is not None:
                total_validations += 1
                if abs(result["put_delta"] - ref.get("put_delta", 0)) < 0.05:
                    passed_validations += 1
    
    print(f"\nTotal Validations: {total_validations}")
    print(f"Passed: {passed_validations}")
    print(f"Failed: {total_validations - passed_validations}")
    print(f"Pass Rate: {passed_validations/total_validations*100:.1f}%" if total_validations > 0 else "N/A")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

