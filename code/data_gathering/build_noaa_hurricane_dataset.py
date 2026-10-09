"""Build a storm-level NOAA IBTrACS dataset.

Default output: data/disaster_data/worldwide_hurricanes_noaa_complete.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd


NOAA_IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r00/access/csv/ibtracs.ALL.list.v04r00.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "disaster_data" / "worldwide_hurricanes_noaa_complete.csv"
REQUIRED_COLUMNS = {
    "SID", "SEASON", "NAME", "BASIN", "USA_WIND", "USA_PRES",
    "DIST2LAND", "LAT", "LON", "ISO_TIME",
}

def wind_to_category(knots):
    """Convert max sustained wind speed (knots) to Saffir-Simpson Category."""
    if pd.isna(knots) or knots < 34:
        return "Tropical Depression"
    elif knots < 64:
        return "Tropical Storm"
    elif knots < 83:
        return "Category 1"
    elif knots < 96:
        return "Category 2"
    elif knots < 113:
        return "Category 3"
    elif knots < 137:
        return "Category 4"
    else:
        return "Category 5"

def build_worldwide_hurricane_dataset(
    start_year: int = 2000,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    source_url: str = NOAA_IBTRACS_URL,
) -> Path:
    """Build a storm-level IBTrACS CSV and return its output path."""
    if start_year < 0:
        raise ValueError("start_year must be a non-negative integer")

    output_path = Path(output_csv)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print(f"1. Fetching official NOAA IBTrACS global dataset (since {start_year})...")
    try:
        # The second row documents units rather than supplying a storm record.
        df = pd.read_csv(source_url, skiprows=[1], low_memory=False)
    except (OSError, URLError) as error:
        raise RuntimeError(
            "Could not download the NOAA IBTrACS source. Check your internet "
            f"connection and source URL: {source_url}"
        ) from error

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "The NOAA source is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Clean up column names and types
    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    df = df[df['SEASON'] >= start_year].copy()

    # Numeric conversions for key metrics
    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')      # Wind in knots
    df['USA_PRES'] = pd.to_numeric(df['USA_PRES'], errors='coerce')      # Min Pressure in mb
    df['DIST2LAND'] = pd.to_numeric(df['DIST2LAND'], errors='coerce')    # Distance to land (km)
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])

    df['NAME'] = df['NAME'].str.upper().str.strip()

    print("2. Processing track points, landfalls, pressure metrics, and ocean basins...")

    records = []
    grouped = df.groupby('SID')

    for sid, group in grouped:
        storm_name = group['NAME'].iloc[0]
        season = int(group['SEASON'].iloc[0])
        basins = group['BASIN'].dropna().astype(str).str.strip()
        basins = basins[basins.ne("")]
        basin = basins.mode().iat[0] if not basins.empty else "Unknown"

        # Max Wind & Min Pressure
        max_wind_kts = group['USA_WIND'].max()
        if pd.isna(max_wind_kts) or max_wind_kts < 34:
            continue  # Filter out weak unorganized depressions

        min_pressure_mb = group['USA_PRES'].min()
        peak_category = wind_to_category(max_wind_kts)

        # Timestamps and Duration
        start_time = group['ISO_TIME'].min()
        end_time = group['ISO_TIME'].max()
        duration_days = round((end_time - start_time).total_seconds() / 86400.0, 1)

        # Landfall Detection (DIST2LAND == 0 indicates landfall point)
        landfall_points = group[group['DIST2LAND'] == 0]
        made_landfall = len(landfall_points) > 0

        if made_landfall:
            first_landfall = landfall_points.iloc[0]
            landfall_time = first_landfall['ISO_TIME'].strftime('%Y-%m-%d %H:%M')
            landfall_lat = first_landfall['LAT']
            landfall_lon = first_landfall['LON']
            landfall_wind_kts = first_landfall['USA_WIND']
            landfall_category = wind_to_category(landfall_wind_kts)
            landfall_status = "Landfall"
        else:
            landfall_time = "N/A"
            landfall_lat = np.nan
            landfall_lon = np.nan
            landfall_wind_kts = np.nan
            landfall_category = "At Sea / Open Water"
            landfall_status = "At Sea / Open Water"

        records.append({
            'storm_id': sid,
            'hurricane_name': f"{storm_name.title()} ({season})",
            'season': season,
            'basin': basin,
            'peak_category': peak_category,
            'peak_wind_kts': max_wind_kts,
            'peak_wind_mph': round(max_wind_kts * 1.15078, 1) if not pd.isna(max_wind_kts) else np.nan,
            'min_pressure_mb': min_pressure_mb,
            'start_date': start_time.strftime('%Y-%m-%d'),
            'end_date': end_time.strftime('%Y-%m-%d'),
            'duration_days': duration_days,
            'landfall_status': landfall_status,
            'landfall_category': landfall_category,
            'landfall_time': landfall_time,
            'landfall_lat': landfall_lat,
            'landfall_lon': landfall_lon,
            'landfall_wind_kts': landfall_wind_kts
        })

    result_df = pd.DataFrame(records)
    if result_df.empty:
        raise ValueError(f"No storms with winds of at least 34 knots found since {start_year}.")
    result_df = result_df.sort_values(by=['season', 'hurricane_name']).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"3. Writing {len(result_df)} verified records to '{output_path}'...")
    result_df.to_csv(output_path, index=False)
    print("Dataset generation complete!")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a worldwide NOAA IBTrACS storm dataset.")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()
    build_worldwide_hurricane_dataset(start_year=args.start_year, output_csv=args.output)
