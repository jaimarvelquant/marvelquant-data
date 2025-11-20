"""
Custom Data Types for MarvelQuant Data Management.

Provides custom data types that extend Nautilus core Data class:
- OptionOI: Option open interest data
- FutureOI: Future open interest data
- OptionGreeks: Option Greeks and implied volatility data

These types are stored separately from Bar data because Nautilus Bar class
does NOT have an open_interest field.

Usage:
    >>> from marvelquant_data.data_types import OptionOI, FutureOI, OptionGreeks
    >>> from nautilus_trader.model.identifiers import InstrumentId
    >>>
    >>> # Option OI
    >>> option_oi = OptionOI(
    ...     instrument_id=InstrumentId.from_str("BANKNIFTY28OCT2548000CE.NSE"),
    ...     oi=1_500_000,
    ...     coi=50_000,
    ...     ts_event=1704177900000000000,
    ...     ts_init=1704177900000000000
    ... )
    >>>
    >>> # Future OI
    >>> future_oi = FutureOI(
    ...     instrument_id=InstrumentId.from_str("BANKNIFTY-I.NSE"),
    ...     oi=2_500_000,
    ...     coi=100_000,
    ...     ts_event=1704177900000000000,
    ...     ts_init=1704177900000000000
    ... )
    >>>
    >>> # Option Greeks
    >>> greeks = OptionGreeks(
    ...     instrument_id=InstrumentId.from_str("NIFTY24JAN18000CE.NSE"),
    ...     iv=0.15,
    ...     delta=0.5,
    ...     gamma=0.001,
    ...     theta=-0.05,
    ...     vega=0.2,
    ...     rho=0.01,
    ...     ts_event=1704067200000000000,
    ...     ts_init=1704067200000000000
    ... )
"""

from .option_oi import OptionOI
from .future_oi import FutureOI
from .option_greeks import OptionGreeks

__all__ = [
    "OptionOI",
    "FutureOI",
    "OptionGreeks",
]
