"""
Build a hurricane + stock-market dataset from a hurricane CSV.

Expected hurricane CSV columns:
    hurricane_name,date,severity,deaths,location,injuries,damage_billions,duration_days

IMPORTANT:
- `date` may contain an actual event date (e.g. 2017-08-25) or a storm period
  (e.g. 2017-08-25 to 2017-09-02).  For a period, the first date is used as
  the event date.
- The script uses the first available stock-market trading day ON OR AFTER the
  hurricane date as the event trading day.
- All pre-event input features use only trading days strictly BEFORE the event
  trading day, preventing look-ahead/data leakage.
- Target variables use the next 5 trading days after the event trading day.

Outputs:
    complete_dataset.csv  -> one row per hurricane with hurricane features,
                             pre-hurricane market features, and four target outputs.
    market_raw.csv        -> raw downloaded market data for transparency/debugging.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit(
        "Missing dependency: yfinance. Install with: pip install yfinance pandas numpy"
    )


# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = Path("hurricanes_since_2000.csv")
OUTPUT_FILE = Path("complete_dataset.csv")
RAW_MARKET_FILE = Path("market_raw.csv")

TICKERS = {
    "SPY": "SPY",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Financials": "XLF",
    "Industrials": "XLI",
    "VIX": "^VIX",
}

SECTORS = ["Energy", "Utilities", "Financials", "Industrials"]
PRICE_TICKERS = [TICKERS["SPY"]] + [TICKERS[s] for s in SECTORS]
ALL_TICKERS = PRICE_TICKERS + [TICKERS["VIX"]]

# Number of trading days needed around each event.
LOOKBACK_RETURN_DAYS = 5
VOLATILITY_WINDOW = 20
DRAWDOWN_WINDOW = 20
BETA_WINDOW = 60
TARGET_FORWARD_DAYS = 5
EXTRA_BUFFER_BEFORE = 10
EXTRA_BUFFER_AFTER = 10

REQUIRED_COLUMNS = [
    "hurricane_name",
    "date",
    "severity",
    "deaths",
    "location",
    "injuries",
    "damage_billions",
    "duration_days",
]


# -----------------------------
# Helpers
# -----------------------------
def fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)


def load_hurricanes(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Could not find {path.resolve()}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fail(
            "The hurricane CSV is missing these required columns: "
            + ", ".join(missing)
        )

    # The supplied source records a storm's lifetime as
    # "YYYY-MM-DD to YYYY-MM-DD".  Market features need one date, so use the
    # beginning of that period.  Single ISO dates remain supported.
    date_text = df["date"].astype("string").str.strip()
    period = date_text.str.fullmatch(
        r"(?P<start>\d{4}-\d{2}-\d{2})\s+to\s+\d{4}-\d{2}-\d{2}",
        na=False,
    )
    event_date_text = date_text.where(~period, date_text.str.extract(
        r"^(?P<start>\d{4}-\d{2}-\d{2})\s+to\s+\d{4}-\d{2}-\d{2}$",
        expand=False,
    ))
    parsed = pd.to_datetime(event_date_text, format="%Y-%m-%d", errors="coerce")
    year_only = date_text.str.fullmatch(r"\d{4}", na=False)

    if year_only.any():
        examples = sorted(df.loc[year_only, "date"].astype(str).unique())[:5]
        fail(
            "The `date` column contains year-only values such as "
            f"{examples}. Stock data cannot be aligned to a hurricane from year alone. "
            "Replace `date` with an event date such as 2017-08-25, or a date range such as "
            "2017-08-25 to 2017-09-02."
        )

    if parsed.isna().any():
        bad_rows = df.index[parsed.isna()].tolist()[:10]
        fail(
            "Some hurricane dates could not be parsed. Check rows: "
            + ", ".join(map(str, bad_rows))
        )

    df = df.copy()
    df["date"] = parsed.dt.tz_localize(None)
    return df


def normalize_yfinance_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a flat dataframe with Date + ticker fields."""
    if raw.empty:
        fail("yfinance returned no market data.")

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance can return columns in either (Price, Ticker) or (Ticker, Price) form.
        level0 = list(raw.columns.get_level_values(0))
        level1 = list(raw.columns.get_level_values(1))
        known_prices = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if any(x in known_prices for x in level0):
            price_level, ticker_level = 0, 1
        else:
            ticker_level, price_level = 0, 1

        pieces = {}
        for ticker in sorted(set(raw.columns.get_level_values(ticker_level))):
            for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                key = (field, ticker) if price_level == 0 else (ticker, field)
                if key in raw.columns:
                    out_name = f"{ticker}_{field.replace(' ', '_')}"
                    pieces[out_name] = raw[key]
        df = pd.DataFrame(pieces, index=raw.index)
    else:
        df = raw.copy()

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def download_market_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Use a single call for consistent timestamps and alignment.
    raw = yf.download(
        tickers=ALL_TICKERS,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    df = normalize_yfinance_columns(raw)

    # Confirm the expected Close columns exist.
    for ticker in ALL_TICKERS:
        if ticker == "^VIX":
            col = "^VIX_Close"
        else:
            col = f"{ticker}_Close"
        if col not in df.columns:
            fail(f"Missing downloaded column: {col}")

    return df


def trading_day_on_or_after(index: pd.DatetimeIndex, date: pd.Timestamp):
    pos = index.searchsorted(date)
    if pos >= len(index):
        return None
    return index[pos]


def prior_window(series: pd.Series, event_pos: int, n: int) -> pd.Series:
    start = event_pos - n
    end = event_pos
    if start < 0:
        return pd.Series(dtype=float)
    return series.iloc[start:end].dropna()


def forward_window(series: pd.Series, event_pos: int, n: int) -> pd.Series:
    start = event_pos + 1
    end = event_pos + 1 + n
    if end > len(series):
        return pd.Series(dtype=float)
    return series.iloc[start:end].dropna()


def percent_return(prices: pd.Series) -> float:
    if len(prices) < 2:
        return np.nan
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def previous_n_return(series: pd.Series, event_pos: int, n: int) -> float:
    # Uses the last close before the event vs. the close n trading days earlier.
    if event_pos - n < 0:
        return np.nan
    p0 = series.iloc[event_pos - n]
    p1 = series.iloc[event_pos - 1]
    if pd.isna(p0) or pd.isna(p1):
        return np.nan
    return float(p1 / p0 - 1.0)


def future_n_return(series: pd.Series, event_pos: int, n: int) -> float:
    # Uses event close as the start and the close n trading days later as the end.
    if event_pos + n >= len(series):
        return np.nan
    p0 = series.iloc[event_pos]
    p1 = series.iloc[event_pos + n]
    if pd.isna(p0) or pd.isna(p1):
        return np.nan
    return float(p1 / p0 - 1.0)


def volatility_annualized(series: pd.Series, event_pos: int, window: int) -> float:
    if event_pos < window:
        return np.nan
    window_prices = series.iloc[event_pos - window:event_pos]
    returns = window_prices.pct_change().dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(252))


def volume_change(series: pd.Series, event_pos: int, window: int = 20) -> float:
    # Compares the most recent pre-event trading-day volume with its prior 20-day average.
    if event_pos < window + 1:
        return np.nan
    recent = series.iloc[event_pos - 1]
    baseline = series.iloc[event_pos - 1 - window:event_pos - 1].mean()
    if pd.isna(recent) or pd.isna(baseline) or baseline == 0:
        return np.nan
    return float(recent / baseline - 1.0)


def drawdown(series: pd.Series, event_pos: int, window: int = 20) -> float:
    if event_pos < window:
        return np.nan
    window_prices = series.iloc[event_pos - window:event_pos]
    peak = window_prices.max()
    last = series.iloc[event_pos - 1]
    if pd.isna(peak) or pd.isna(last) or peak == 0:
        return np.nan
    return float(last / peak - 1.0)


def beta_to_spy(
    sector_prices: pd.Series,
    spy_prices: pd.Series,
    event_pos: int,
    window: int = 60,
) -> float:
    if event_pos < window + 1:
        return np.nan
    sector_window = sector_prices.iloc[event_pos - window:event_pos]
    spy_window = spy_prices.iloc[event_pos - window:event_pos]
    aligned = pd.concat(
        [sector_window.pct_change(), spy_window.pct_change()], axis=1
    ).dropna()
    aligned.columns = ["sector", "spy"]
    if len(aligned) < 20:
        return np.nan
    variance = aligned["spy"].var(ddof=1)
    if pd.isna(variance) or variance == 0:
        return np.nan
    covariance = aligned["sector"].cov(aligned["spy"])
    return float(covariance / variance)


def build_row(hurricane: pd.Series, market: pd.DataFrame) -> Dict:
    event_date = pd.Timestamp(hurricane["date"])
    event_trading_date = trading_day_on_or_after(market.index, event_date)
    if event_trading_date is None:
        return {}

    event_pos = market.index.get_loc(event_trading_date)
    spy = market["SPY_Close"]

    row = hurricane.to_dict()
    row["event_trading_date"] = event_trading_date.date().isoformat()
    row["event_date_shift_days"] = (event_trading_date - event_date).days

    # Market input features known BEFORE the event.
    row["spy_prev_5d_return"] = previous_n_return(spy, event_pos, LOOKBACK_RETURN_DAYS)
    row["vix"] = (
        float(market["^VIX_Close"].iloc[event_pos - 1])
        if event_pos >= 1 and pd.notna(market["^VIX_Close"].iloc[event_pos - 1])
        else np.nan
    )

    for sector in SECTORS:
        ticker = TICKERS[sector]
        prices = market[f"{ticker}_Close"]
        volume = market[f"{ticker}_Volume"]

        prefix = sector.lower()
        row[f"{prefix}_prev_5d_return"] = previous_n_return(
            prices, event_pos, LOOKBACK_RETURN_DAYS
        )
        row[f"{prefix}_20d_volatility"] = volatility_annualized(
            prices, event_pos, VOLATILITY_WINDOW
        )
        row[f"{prefix}_volume_change"] = volume_change(volume, event_pos)
        row[f"{prefix}_drawdown"] = drawdown(prices, event_pos, DRAWDOWN_WINDOW)
        row[f"{prefix}_beta"] = beta_to_spy(prices, spy, event_pos, BETA_WINDOW)

        # 5-trading-day post-event return.
        sector_future = future_n_return(prices, event_pos, TARGET_FORWARD_DAYS)
        spy_future = future_n_return(spy, event_pos, TARGET_FORWARD_DAYS)
        row[f"{prefix}_5d_abnormal_return"] = (
            sector_future - spy_future
            if pd.notna(sector_future) and pd.notna(spy_future)
            else np.nan
        )

    return row


def main() -> None:
    hurricanes = load_hurricanes(INPUT_FILE)

    min_date = hurricanes["date"].min()
    max_date = hurricanes["date"].max()

    # We need enough pre-event history for beta/volatility plus the future target window.
    download_start = min_date - pd.Timedelta(days=180)
    download_end = max_date + pd.Timedelta(days=30)

    print(f"Hurricanes: {len(hurricanes)}")
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print("Downloading daily market data...")
    market = download_market_data(download_start, download_end)
    market.to_csv(RAW_MARKET_FILE, index_label="Date")
    print(f"Saved raw market data to {RAW_MARKET_FILE}")

    rows: List[Dict] = []
    skipped = []

    for _, hurricane in hurricanes.iterrows():
        try:
            row = build_row(hurricane, market)
            if row:
                rows.append(row)
            else:
                skipped.append((hurricane["hurricane_name"], "no trading date after event"))
        except Exception as exc:  # keep one bad hurricane from stopping the entire run
            skipped.append((hurricane["hurricane_name"], str(exc)))

    if not rows:
        fail("No hurricane rows could be processed.")

    complete = pd.DataFrame(rows)

    # Put hurricane columns first, then market inputs, then targets.
    hurricane_cols = REQUIRED_COLUMNS + ["event_trading_date", "event_date_shift_days"]
    market_cols = ["spy_prev_5d_return", "vix"]
    for sector in SECTORS:
        p = sector.lower()
        market_cols.extend(
            [
                f"{p}_prev_5d_return",
                f"{p}_20d_volatility",
                f"{p}_volume_change",
                f"{p}_drawdown",
                f"{p}_beta",
            ]
        )
    target_cols = [f"{s.lower()}_5d_abnormal_return" for s in SECTORS]

    ordered = [c for c in hurricane_cols + market_cols + target_cols if c in complete.columns]
    extras = [c for c in complete.columns if c not in ordered]
    complete = complete[ordered + extras]

    complete.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved complete dataset to {OUTPUT_FILE}")
    print(f"Rows created: {len(complete)}")
    print(f"Rows skipped: {len(skipped)}")

    if skipped:
        print("\nSkipped rows:")
        for name, reason in skipped[:25]:
            print(f"  - {name}: {reason}")
        if len(skipped) > 25:
            print(f"  ... and {len(skipped) - 25} more")

    print("\nOutput columns:")
    for col in complete.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
