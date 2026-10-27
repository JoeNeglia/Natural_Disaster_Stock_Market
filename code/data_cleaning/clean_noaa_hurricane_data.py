"""Clean the NOAA hurricane dataset down to storms that made landfall.

Input: data/disaster_data/worldwide_hurricanes_noaa_complete.csv
Output: data/disaster_data/worldwide_hurricanes_noaa_landfall.csv
"""

import pandas as pd
from pathlib import Path

# Define file paths
data_dir = Path(__file__).resolve().parents[2] / 'data' / 'disaster_data'
input_path = data_dir / 'worldwide_hurricanes_noaa_complete.csv'
output_path = data_dir / 'worldwide_hurricanes_noaa_landfall.csv'

DROP_COLUMNS = ['landfall_time', 'min_pressure_mb', 'basin', 'season', 'peak_wind_mph']

REGIONS = ['north_atlantic', 'east_pacific', 'west_pacific', 'north_indian', 'south_indian',
           'south_pacific', 'south_atlantic']


def landfall_region(lat, lon):
    """Assign an ocean basin to a landfall point using NOAA basin boundaries."""
    if lat < 0:
        if -90 < lon < -30:
            return 'south_atlantic'
        if 20 <= lon < 135:
            return 'south_indian'
        return 'south_pacific'

    if 30 <= lon < 100:
        return 'north_indian'
    if lon >= 100 or lon < -105:
        return 'west_pacific' if lon >= 100 else 'east_pacific'

    # Mexico and Central America: split Pacific-coast from Gulf/Caribbean-coast landfalls.
    if lon <= -100:
        return 'east_pacific' if lat < 21 else 'north_atlantic'
    if lon <= -97:
        return 'east_pacific' if lat < 18 else 'north_atlantic'
    if lon <= -90:
        return 'east_pacific' if lat < 17.5 else 'north_atlantic'
    if lon <= -83:
        return 'east_pacific' if lat < 15 and lon < -84.5 else 'north_atlantic'
    return 'north_atlantic'


def add_region_flags(df):
    """Replace landfall_lat/landfall_lon with one 0/1 column per landfall region."""
    df = df.copy()
    region = [landfall_region(lat, lon) for lat, lon in zip(df['landfall_lat'], df['landfall_lon'])]
    for name in REGIONS:
        df[f'region_{name}'] = [int(r == name) for r in region]

    print("Landfall regions:")
    print(pd.Series(region).value_counts().to_string())
    return df.drop(columns=['landfall_lat', 'landfall_lon'])


def impute_landfall_wind(df):
    """Estimate missing landfall_wind_kts from peak_wind_kts.

    For storms with a known landfall wind, compute the ratio landfall / peak.
    A missing value is filled with peak_wind_kts times the median ratio of
    storms sharing the same peak_category and landfall_category (falling back
    to peak_category alone, then the overall median). The estimate is then
    clipped to the wind range observed for its landfall_category so it stays
    consistent with that label.
    """
    df = df.copy()
    known = df.dropna(subset=['landfall_wind_kts'])
    ratio = known['landfall_wind_kts'] / known['peak_wind_kts']

    pair_ratio = ratio.groupby([known['peak_category'], known['landfall_category']]).median()
    peak_ratio = ratio.groupby(known['peak_category']).median()
    category_range = known.groupby('landfall_category')['landfall_wind_kts'].agg(['min', 'max'])

    missing = df['landfall_wind_kts'].isna()
    for idx, row in df[missing].iterrows():
        key = (row['peak_category'], row['landfall_category'])
        r = pair_ratio.get(key, peak_ratio.get(row['peak_category'], ratio.median()))
        estimate = row['peak_wind_kts'] * r

        if row['landfall_category'] in category_range.index:
            low, high = category_range.loc[row['landfall_category']]
            estimate = min(max(estimate, low), high)

        df.at[idx, 'landfall_wind_kts'] = round(estimate)

    print(f"Imputed landfall_wind_kts for {missing.sum()} rows")
    return df


def clean_noaa_hurricanes(input_csv, output_csv):
    print(f"Reading dataset from:\n{input_csv}")

    df = pd.read_csv(input_csv)

    # Keep only storms that made landfall (handles casing/spacing differences)
    at_sea = df['landfall_status'].astype(str).str.strip().str.lower() == 'at sea / open water'
    cleaned_df = df[~at_sea].copy()

    # Every remaining row made landfall, so the status column is no longer needed
    cleaned_df = cleaned_df.drop(columns=DROP_COLUMNS + ['landfall_status'])

    cleaned_df = impute_landfall_wind(cleaned_df)
    cleaned_df = add_region_flags(cleaned_df)

    cleaned_df.to_csv(output_csv, index=False)

    print(f"\nCleaned successfully!")
    print(f"Original rows: {len(df)}")
    print(f"Landfall rows kept: {len(cleaned_df)}")
    print(f"Columns: {', '.join(cleaned_df.columns)}")
    print(f"Saved to:\n{output_csv}")


if __name__ == '__main__':
    clean_noaa_hurricanes(input_path, output_path)
