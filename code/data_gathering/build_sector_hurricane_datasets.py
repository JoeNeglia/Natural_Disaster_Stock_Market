"""Build one EM-DAT/NOAA matched hurricane market dataset per sector.

Usage from the repository root:
    python code/modeling/build_sector_hurricane_datasets.py

Input: data/disaster_data/emdat_noaa_matched.csv and data/stock_data/market_raw.csv
Outputs: four CSVs in data/sector_datasets/ (one per sector)

The event date is the first market trading day on or after ``start_date``.
Returns and other stock metrics use daily close/volume data from market_raw.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHED = ROOT / "data" / "disaster_data" / "emdat_noaa_matched.csv"
DEFAULT_MARKET = ROOT / "data" / "stock_data" / "market_raw.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "sector_datasets"

SECTORS = {
    "energy": "XLE",
    "utilities": "XLU",
    "industrials": "XLI",
    "financials": "XLF",
}


def _market_features(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    market = market.copy()
    market["Date"] = pd.to_datetime(market["Date"], errors="coerce")
    market = market.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    close = pd.DataFrame(index=market.index)
    volume = pd.DataFrame(index=market.index)
    for ticker in ["SPY", "XLE", "XLU", "XLI", "XLF"]:
        close[ticker] = pd.to_numeric(market[f"{ticker}_Close"], errors="coerce")
        volume[ticker] = pd.to_numeric(market[f"{ticker}_Volume"], errors="coerce")

    returns = close.pct_change(fill_method=None)
    features = pd.DataFrame(index=market.index)
    features["spy_prev_5d_return"] = close["SPY"].pct_change(5, fill_method=None)
    features["vix"] = pd.to_numeric(market["^VIX_Close"], errors="coerce")
    sector_returns: dict[str, pd.Series] = {}
    for sector, ticker in SECTORS.items():
        r = returns[ticker]
        sector_returns[sector] = r
        features[f"{sector}_prev_5d_return"] = close[ticker].pct_change(5, fill_method=None)
        features[f"{sector}_20d_volatility"] = r.rolling(20, min_periods=20).std() * np.sqrt(252)
        features[f"{sector}_volume_change"] = volume[ticker].pct_change(5, fill_method=None)
        features[f"{sector}_drawdown"] = close[ticker] / close[ticker].rolling(20, min_periods=1).max() - 1
        covariance = r.rolling(60, min_periods=20).cov(returns["SPY"])
        variance = returns["SPY"].rolling(60, min_periods=20).var()
        features[f"{sector}_beta"] = covariance / variance

    features["^VIX_Close"] = pd.to_numeric(market["^VIX_Close"], errors="coerce")
    return features, sector_returns


def build_sector_datasets(
    matched_path: Path = DEFAULT_MATCHED,
    market_path: Path = DEFAULT_MARKET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    matched = pd.read_csv(matched_path, low_memory=False)
    market = pd.read_csv(market_path, low_memory=False)
    features, sector_returns = _market_features(market)

    if "start_date" not in matched.columns:
        raise ValueError(f"Matched input has no start_date column: {matched_path}")
    matched["_event_date"] = pd.to_datetime(matched["start_date"], errors="coerce")
    # Map every disaster date to the first available market date at or after it.
    trading_dates = features.index
    matched["event_trading_date"] = matched["_event_date"].map(
        lambda date: trading_dates[trading_dates.searchsorted(date)]
        if pd.notna(date) and trading_dates.searchsorted(date) < len(trading_dates)
        else pd.NaT
    )
    matched["event_date_shift_days"] = (
        matched["event_trading_date"] - matched["_event_date"]
    ).dt.days
    matched = matched.drop(columns="_event_date")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for sector in SECTORS:
        cols = ["spy_prev_5d_return", "vix"] + [
            f"{sector}_{metric}"
            for metric in ["prev_5d_return", "20d_volatility", "volume_change", "drawdown", "beta"]
        ]
        sector_frame = matched.copy()
        for col in cols:
            sector_frame[col] = sector_frame["event_trading_date"].map(features[col])

        # Five-session sector return after the event, adjusted by its pre-event beta
        # against SPY. The conceptual dataset uses this as the sector abnormal return.
        abnormal_col = f"{sector}_5d_abnormal_return"
        market_close = _market_closes(market)
        future_sector = market_close[SECTORS[sector]].shift(-5) / market_close[SECTORS[sector]] - 1
        future_spy = market_close["SPY"].shift(-5) / market_close["SPY"] - 1
        beta = features[f"{sector}_beta"]
        abnormal = future_sector - beta * future_spy
        sector_frame[abnormal_col] = sector_frame["event_trading_date"].map(abnormal)

        # Stable, readable order: matched disaster data, event market context, sector metrics.
        stock_cols = ["spy_prev_5d_return", "vix"] + [
            f"{sector}_{metric}"
            for metric in ["prev_5d_return", "20d_volatility", "volume_change", "drawdown", "beta"]
        ] + [abnormal_col]
        sector_frame = sector_frame[[c for c in sector_frame.columns if c not in stock_cols] + stock_cols]
        path = output_dir / f"{sector}_hurricane_dataset.csv"
        sector_frame.to_csv(path, index=False)
        outputs[sector] = path
        print(f"Saved {len(sector_frame):,} rows to {path}")
    return outputs


def _market_closes(market: pd.DataFrame) -> pd.DataFrame:
    data = market.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    return pd.DataFrame(
        {ticker: pd.to_numeric(data[f"{ticker}_Close"], errors="coerce")
         for ticker in ["SPY", *SECTORS.values()]},
        index=data.index,
    )


if __name__ == "__main__":
    build_sector_datasets()
