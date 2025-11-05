# MarvelQuant Data Management

A standalone Python library for managing and transforming NSE (National Stock Exchange of India) market data for use with the [NautilusTrader](https://nautilustrader.io/) backtesting framework.

## Features

- **Custom Data Types**: OptionOI and FutureOI classes for tracking open interest
- **NSE-Specific Utilities**:
  - Lot size mappings for all NSE instruments
  - Holiday calendar (2018-2024) for accurate DTE calculations
  - Expiry date calculators (monthly/weekly)
  - Expiry bucket classification (CW/NW/CM/NM)
- **Timestamp Conversion**: IST to UTC nanosecond conversion for Nautilus compatibility
- **Contract Generators**: Automated OptionContract and FuturesContract metadata generation
- **Sample Data**: Includes 2 days of NIFTY and BANKNIFTY option chain data for January 2024

## Installation

### From Source

```bash
git clone https://github.com/jaimarvelquant/marvelquant-data.git
cd marvelquant-data
pip install -e .
```

### Using pip (once published)

```bash
pip install marvelquant-data
```

## Requirements

- Python >= 3.10
- nautilus_trader >= 1.200.0
- pandas >= 2.0.0
- pyarrow >= 14.0.0
- pytz >= 2023.3

## Quick Start

### 1. Working with Custom Data Types

```python
from marvelquant_data.data_types import OptionOI, FutureOI
from nautilus_trader.model.identifiers import InstrumentId

# Create Option OI data
option_oi = OptionOI(
    instrument_id=InstrumentId.from_str("NIFTY01FEB2422000CE.NSE"),
    oi=150_000,
    coi=2_500,
    ts_event=1704177900000000000,
    ts_init=1704177900000000000
)

print(option_oi)  # OptionOI[NIFTY01FEB2422000CE.NSE]: OI=150,000, COI=+2,500
```

### 2. Using NSE Utilities

```python
from marvelquant_data.utils import NSEHolidayCalendar, get_nse_monthly_expiry
from datetime import date

# Check if a date is a trading day
calendar = NSEHolidayCalendar()
is_trading = calendar.is_trading_day(date(2024, 1, 26))  # False (Republic Day)

# Calculate trading days between dates
trading_days = calendar.trading_days_between(
    date(2024, 1, 15),
    date(2024, 1, 25)
)  # 7 trading days

# Get monthly expiry
expiry = get_nse_monthly_expiry(2024, 1, calendar)
print(expiry)  # 2024-01-25 (last Thursday of January)
```

### 3. Creating Contract Metadata

```python
from marvelquant_data.utils import create_options_contract
from datetime import date

# Create Nautilus OptionContract
contract = create_options_contract(
    symbol="NIFTY01FEB2422000CE",
    strike=22000.0,
    expiry=date(2024, 2, 1),
    option_kind="CALL",
    underlying="NIFTY"
)

print(contract.instrument_id)  # NIFTY01FEB2422000CE.NSE
print(contract.strike_price)   # 22000.00 INR
print(contract.lot_size)       # 25 (NIFTY lot size)
```

### 4. Timestamp Conversion

```python
from marvelquant_data.utils import yyyymmdd_seconds_to_utc_ns

# Convert NSE timestamp (IST) to UTC nanoseconds
utc_ns = yyyymmdd_seconds_to_utc_ns(
    date_int=20240102,  # 2024-01-02
    seconds_int=34500   # 09:35:00 IST
)

print(utc_ns)  # 1704177900000000000 (2024-01-02 04:05:00 UTC)
```

## Data Architecture

This library follows the NautilusTrader data architecture pattern:

### Tier 1: Bar Data (OHLCV)
- Stored in Apache Parquet binary format
- Location: `data/transformed/bar/{BAR_TYPE}/{TIMESTAMP_RANGE}.parquet`
- Uses Nautilus native `Bar` class with binary price encoding

### Tier 1.5: QuoteTick Data
- Required for Greeks calculation
- Location: `data/transformed/quote_tick/{INSTRUMENT_ID}/{TIMESTAMP_RANGE}.parquet`
- Generated from bars using `bars_to_quote_ticks()` function

### Tier 2: Contract Metadata
- OptionContract and FuturesContract metadata
- Loaded once into memory per instrument

### Custom Data: Open Interest
- Separate custom data types (OptionOI, FutureOI)
- Stored separately from Bar data
- Uses `@customdataclass` decorator for serialization

For more details, see [docs/data-architecture.md](docs/data-architecture.md)

## Sample Data

The repository includes sample data for testing and development:

```
data/raw_source/option_chain/
├── nifty/
│   ├── 2024-01-02_nifty_processed_corrected.parquet
│   └── 2024-01-03_nifty_processed_corrected.parquet
└── banknifty/
    ├── 2024-01-02_banknifty_processed_corrected.parquet
    └── 2024-01-03_banknifty_processed_corrected.parquet
```

## NSE Instrument Reference

### Lot Sizes (as of 2024)

| Instrument | Lot Size | Asset Class |
|-----------|----------|-------------|
| NIFTY | 25 | Equity |
| BANKNIFTY | 15 | Equity |
| FINNIFTY | 25 | Equity |
| MIDCPNIFTY | 50 | Equity |
| SENSEX | 10 | Equity |
| CRUDEOIL | 100 | Commodity |
| NATURALGAS | 1250 | Commodity |

### Expiry Rules

- **Monthly Options/Futures**: Last Thursday of the month
- **Weekly Options**: Every Thursday
- **Holiday Adjustment**: If Thursday is a holiday, expiry moves to the previous trading day

## Project Structure

```
marvelquant-data/
├── marvelquant_data/          # Main package
│   ├── data_types/           # Custom data types (OptionOI, FutureOI)
│   ├── utils/                # NSE utilities and helpers
│   ├── transformers/         # Data transformation modules
│   └── validators/           # Data validation tools
├── data/                     # Data directory
│   ├── raw_source/          # Original source data
│   └── transformed/         # Transformed Nautilus catalog
├── docs/                    # Documentation
├── examples/                # Usage examples
├── tests/                   # Unit tests
├── pyproject.toml          # Package configuration
└── README.md               # This file
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/jaimarvelquant/marvelquant-data.git
cd marvelquant-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code with black
black marvelquant_data/

# Sort imports with isort
isort marvelquant_data/

# Type checking with mypy
mypy marvelquant_data/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NautilusTrader](https://nautilustrader.io/) - The backtesting framework this library is built for
- NSE (National Stock Exchange of India) - Data source

## Support

For issues, questions, or contributions, please visit:
- **Issues**: https://github.com/jaimarvelquant/marvelquant-data/issues
- **Discussions**: https://github.com/jaimarvelquant/marvelquant-data/discussions

## Roadmap

- [ ] Add more transformation examples
- [ ] Support for BSE (Bombay Stock Exchange) data
- [ ] Add streaming data adapters
- [ ] Improve documentation with tutorials
- [ ] Add more sample data for different underlyings
- [ ] Performance optimization for large datasets

---

**Made with ❤️ for the Indian trading community**
