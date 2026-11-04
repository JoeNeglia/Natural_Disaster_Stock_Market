"""Clean the sector landfall datasets for a regression model.

Keeps only storms that made landfall in the USA at Category 1 or higher and
reduces the columns to the hurricane identifiers (id, name, category, start/end
date) plus the stock features.

US landfalls come from the official NHC landfall points (USA_RECORD == 'L') in
the NOAA IBTrACS North Atlantic and East Pacific track files, since the sector
datasets only record each storm's first landfall (often outside the USA).
The category is the strongest US landfall of the storm.

Usage from the repository root:
    python code/data_cleaning/clean_sector_regression_datasets.py

Input: data/sector_datasets/<sector>_noaa_landfall_dataset.csv
       (built by code/data_gathering/build_noaa_sector_datasets.py)
       NOAA IBTrACS NA and EP track files (downloaded)
Output: data/regression_datasets/<sector>_cat1plus_regression_dataset.csv
"""

import pandas as pd
from pathlib import Path

# Define file paths
data_dir = Path(__file__).resolve().parents[2] / 'data'
input_dir = data_dir / 'sector_datasets'
output_dir = data_dir / 'regression_datasets'

SECTORS = ['energy', 'financials', 'industrials', 'utilities']

IBTRACS_URL = ('https://www.ncei.noaa.gov/data/'
               'international-best-track-archive-for-climate-stewardship-ibtracs/'
               'v04r00/access/csv/ibtracs.{basin}.list.v04r00.csv')
# Only NHC-tracked basins can make US landfall (Hawaii storms are in EP)
IBTRACS_BASINS = ['NA', 'EP']
IBTRACS_COLUMNS = ['SID', 'LAT', 'LON', 'USA_WIND', 'USA_RECORD']

MIN_CATEGORY = 1

HURRICANE_COLUMNS = ['storm_id', 'hurricane_name', 'category', 'start_date', 'end_date']
MARKET_COLUMNS = ['spy_prev_5d_return', 'vix']
SECTOR_FEATURES = ['prev_5d_return', '20d_volatility', 'volume_change', 'drawdown', 'beta',
                   '5d_abnormal_return']


def wind_to_category(knots):
    """Saffir-Simpson category from wind in knots; below hurricane strength -> 0."""
    for category, threshold in [(5, 137), (4, 113), (3, 96), (2, 83), (1, 64)]:
        if knots >= threshold:
            return category
    return 0


def in_usa(lat, lon):
    """True for landfall points on the US mainland Gulf/Atlantic coast or Hawaii."""
    mainland = (lat.between(24.4, 45.5) & lon.between(-97.5, -66.9)
                & ~((lat < 28) & (lon > -79.9))         # Bahamas
                & ~((lon < -97.1) & (lat < 25.95)))     # Mexico, south of the Rio Grande
    hawaii = lat.between(18.5, 22.5) & lon.between(-161, -154.5)
    return mainland | hawaii


def us_landfall_categories():
    """Return the strongest US landfall category of each storm, indexed by storm_id."""
    print("Downloading NOAA IBTrACS track data...")
    tracks = pd.concat([
        pd.read_csv(IBTRACS_URL.format(basin=basin), skiprows=[1], usecols=IBTRACS_COLUMNS,
                    low_memory=False)
        for basin in IBTRACS_BASINS
    ])
    landfalls = tracks[tracks['USA_RECORD'].astype(str).str.strip() == 'L'].copy()
    for col in ['LAT', 'LON', 'USA_WIND']:
        landfalls[col] = pd.to_numeric(landfalls[col], errors='coerce')

    us_landfalls = landfalls[in_usa(landfalls['LAT'], landfalls['LON'])]
    us_wind = us_landfalls.groupby('SID')['USA_WIND'].max()
    return us_wind.map(wind_to_category).rename('category')


def clean_sector(sector, us_category):
    input_csv = input_dir / f'{sector}_noaa_landfall_dataset.csv'
    output_csv = output_dir / f'{sector}_cat{MIN_CATEGORY}plus_regression_dataset.csv'

    df = pd.read_csv(input_csv)

    df['category'] = df['storm_id'].map(us_category)
    cleaned_df = df[df['category'] >= MIN_CATEGORY].copy()
    cleaned_df['category'] = cleaned_df['category'].astype(int)

    stock_columns = MARKET_COLUMNS + [f'{sector}_{name}' for name in SECTOR_FEATURES]
    cleaned_df = cleaned_df[HURRICANE_COLUMNS + stock_columns]

    # Rows without stock history can't be used for regression
    before = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=stock_columns)

    cleaned_df.to_csv(output_csv, index=False)

    print(f"\n{sector}:")
    print(f"  Original rows: {len(df)}")
    print(f"  US landfall Category {MIN_CATEGORY}+ rows: {before}")
    print(f"  Dropped for missing stock data: {before - len(cleaned_df)}")
    print(f"  Rows kept: {len(cleaned_df)}")
    print(f"  Saved to: {output_csv}")


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    us_category = us_landfall_categories()

    for sector in SECTORS:
        clean_sector(sector, us_category)


if __name__ == '__main__':
    main()
