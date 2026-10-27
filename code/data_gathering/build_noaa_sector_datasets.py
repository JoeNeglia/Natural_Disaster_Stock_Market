"""Build one NOAA landfall hurricane market dataset per sector.

Usage from the repository root:
    python code/data_gathering/build_noaa_sector_datasets.py

Input: data/disaster_data/worldwide_hurricanes_noaa_landfall.csv and data/stock_data/market_raw.csv
Outputs: four CSVs in data/sector_datasets/ named <sector>_noaa_landfall_dataset.csv

Each row is one hurricane. The event date is the first market trading day on or
after ``start_date``. Sector metrics are taken on that date, and the target is
the sector's five-session beta-adjusted abnormal return after it.
"""

from pathlib import Path

import pandas as pd

from build_sector_hurricane_datasets import SECTORS, _market_closes, _market_features


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HURRICANES = ROOT / "data" / "disaster_data" / "worldwide_hurricanes_noaa_landfall.csv"
DEFAULT_MARKET = ROOT / "data" / "stock_data" / "market_raw.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "sector_datasets"

SECTOR_METRICS = ["prev_5d_return", "20d_volatility", "volume_change", "drawdown", "beta"]


def build_noaa_sector_datasets(
    hurricanes_path: Path = DEFAULT_HURRICANES,
    market_path: Path = DEFAULT_MARKET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    hurricanes = pd.read_csv(hurricanes_path)
    market = pd.read_csv(market_path, low_memory=False)
    features, _ = _market_features(market)
    market_close = _market_closes(market)

    # Map every hurricane start date to the first available market date at or after it.
    start = pd.to_datetime(hurricanes["start_date"], errors="coerce")
    trading_dates = features.index
    positions = trading_dates.searchsorted(start)
    hurricanes["event_trading_date"] = [
        trading_dates[pos] if pd.notna(date) and pos < len(trading_dates) else pd.NaT
        for date, pos in zip(start, positions)
    ]
    hurricanes["event_date_shift_days"] = (hurricanes["event_trading_date"] - start).dt.days

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for sector, ticker in SECTORS.items():
        frame = hurricanes.copy()
        market_cols = ["spy_prev_5d_return", "vix"] + [f"{sector}_{m}" for m in SECTOR_METRICS]
        for col in market_cols:
            frame[col] = frame["event_trading_date"].map(features[col])

        # Five-session sector return after the event, adjusted by its pre-event beta against SPY.
        future_sector = market_close[ticker].shift(-5) / market_close[ticker] - 1
        future_spy = market_close["SPY"].shift(-5) / market_close["SPY"] - 1
        abnormal = future_sector - features[f"{sector}_beta"] * future_spy
        frame[f"{sector}_5d_abnormal_return"] = frame["event_trading_date"].map(abnormal)

        path = output_dir / f"{sector}_noaa_landfall_dataset.csv"
        frame.to_csv(path, index=False)
        outputs[sector] = path
        print(f"Saved {len(frame):,} rows to {path}")
    return outputs


if __name__ == "__main__":
    build_noaa_sector_datasets()
