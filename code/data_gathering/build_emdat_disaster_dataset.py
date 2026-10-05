"""Build an IBTrACS hurricane dataset with optional NOAA/EM-DAT impact data.

Default output: data/disaster_data/worldwide_hurricanes_complete.csv
"""

import pandas as pd
import numpy as np
import urllib.request
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'disaster_data' / 'worldwide_hurricanes_complete.csv'

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

def fetch_noaa_billion_dollar_disasters():
    """Fetch official US damage ($ Billions) and casualty figures from NOAA NCEI API."""
    print("Fetching NOAA NCEI Billion-Dollar Disasters database...")
    url = "https://www.ncei.noaa.gov/access/billions/services/v2/disasters.json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            records = []
            for item in data.get('disasters', []):
                dtype = item.get('disasterType', '')
                if 'Tropical' in dtype or 'Hurricane' in dtype:
                    records.append({
                        'clean_name': str(item.get('name', '')).upper().strip(),
                        'year': int(item.get('year', 0)),
                        'damage_billions_usd': float(item.get('cost', 0.0)), # Inflation-adjusted
                        'us_deaths': int(item.get('deaths', 0)) if item.get('deaths') is not None else np.nan
                    })
            return pd.DataFrame(records)
    except Exception as e:
        print(f"Notice: Could not fetch NOAA Billion-Dollar Disasters API dynamically ({e}).")
        return pd.DataFrame()

def build_complete_dataset(start_year=2000, emdat_csv_path=None, output_csv=DEFAULT_OUTPUT):
    # Step 1: Download global track and landfall data from NOAA IBTrACS
    print(f"1. Fetching official NOAA IBTrACS global dataset (since {start_year})...")
    ibtracs_url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv"

    df = pd.read_csv(ibtracs_url, skiprows=[1], low_memory=False)

    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    df = df[df['SEASON'] >= start_year].copy()

    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')      # knots
    df['USA_PRES'] = pd.to_numeric(df['USA_PRES'], errors='coerce')      # mb
    df['DIST2LAND'] = pd.to_numeric(df['DIST2LAND'], errors='coerce')    # km
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df['NAME'] = df['NAME'].str.upper().str.strip()

    print("2. Extracting meteorological, landfall, and duration metrics...")
    records = []
    grouped = df.groupby(['SID'])

    for sid, group in grouped:
        storm_name = group['NAME'].iloc[0]
        season = int(group['SEASON'].iloc[0])
        basin = group['BASIN'].iloc[0]

        max_wind_kts = group['USA_WIND'].max()
        if pd.isna(max_wind_kts) or max_wind_kts < 34:
            continue  # Exclude weak tropical depressions

        min_pressure_mb = group['USA_PRES'].min()
        peak_category = wind_to_category(max_wind_kts)

        start_time = group['ISO_TIME'].min()
        end_time = group['ISO_TIME'].max()
        duration_days = round((end_time - start_time).total_seconds() / 86400.0, 1)

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
            'raw_name': storm_name,
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
            'landfall_lat': landfall_lat,
            'landfall_lon': landfall_lon,
            'deaths': np.nan,
            'injuries': np.nan,
            'damage_billions': np.nan
        })

    result_df = pd.DataFrame(records)

    # Step 2: Merge US impact data from NOAA Billion-Dollar Disasters API
    bdoc_df = fetch_noaa_billion_dollar_disasters()
    if not bdoc_df.empty:
        print("3. Merging US landfalls with NOAA economic damage & deaths...")
        result_df = pd.merge(
            result_df,
            bdoc_df,
            left_on=['season', 'raw_name'],
            right_on=['year', 'clean_name'],
            how='left'
        )
        result_df['damage_billions'] = result_df['damage_billions_usd'].combine_first(result_df['damage_billions'])
        result_df['deaths'] = result_df['us_deaths'].combine_first(result_df['deaths'])
        result_df = result_df.drop(columns=['clean_name', 'year', 'damage_billions_usd', 'us_deaths'])

    # Step 3: Optional Merge with EM-DAT Global Export CSV (if provided)
    if emdat_csv_path:
        print(f"4. Merging global non-US casualty & damage data from '{emdat_csv_path}'...")
        try:
            emdat_df = pd.read_csv(emdat_csv_path)
            # Standard EM-DAT field names: 'Disaster Name', 'Year', 'Total Deaths', 'No. Injured', 'Total Damage ('000 US$)'
            emdat_df['clean_name'] = emdat_df['Disaster Name'].astype(str).str.upper().str.strip()
            emdat_df['year'] = pd.to_numeric(emdat_df['Year'], errors='coerce')
            emdat_df['emdat_deaths'] = pd.to_numeric(emdat_df['Total Deaths'], errors='coerce')
            emdat_df['emdat_injuries'] = pd.to_numeric(emdat_df['No. Injured'], errors='coerce')
            emdat_df['emdat_damage_billions'] = pd.to_numeric(emdat_df["Total Damage ('000 US$)"], errors='coerce') / 1e6

            result_df = pd.merge(
                result_df,
                emdat_df[['year', 'clean_name', 'emdat_deaths', 'emdat_injuries', 'emdat_damage_billions']],
                left_on=['season', 'raw_name'],
                right_on=['year', 'clean_name'],
                how='left'
            )
            result_df['deaths'] = result_df['emdat_deaths'].combine_first(result_df['deaths'])
            result_df['injuries'] = result_df['emdat_injuries'].combine_first(result_df['injuries'])
            result_df['damage_billions'] = result_df['emdat_damage_billions'].combine_first(result_df['damage_billions'])
            result_df = result_df.drop(columns=['year', 'clean_name', 'emdat_deaths', 'emdat_injuries', 'emdat_damage_billions'])
        except Exception as e:
            print(f"Could not merge EM-DAT file: {e}")

    result_df = result_df.drop(columns=['raw_name'])
    result_df = result_df.sort_values(by=['season', 'hurricane_name']).reset_index(drop=True)

    print(f"5. Complete! Writing dataset with {len(result_df)} worldwide storms to '{output_csv}'...")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    build_complete_dataset(start_year=2000)
