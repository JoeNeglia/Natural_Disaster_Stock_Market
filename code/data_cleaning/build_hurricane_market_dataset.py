"""Match EM-DAT tropical cyclones with NOAA hurricane records.

Input: data/disaster_data/emdat_tropical_cyclones.csv and
       data/disaster_data/worldwide_hurricanes_noaa_complete.csv
Output: data/disaster_data/emdat_noaa_matched.csv
"""

import pandas as pd
import numpy as np
import re
import os

def merge_reorder_and_clean_complete(emdat_path, noaa_path, output_path):
    print(f"1. Loading base EM-DAT dataset from:\n   {emdat_path}")
    emdat_df = pd.read_csv(emdat_path, low_memory=False)
    original_emdat_count = len(emdat_df)

    print(f"2. Loading NOAA dataset from:\n   {noaa_path}")
    noaa_df = pd.read_csv(noaa_path, low_memory=False)

    # Helper: Extract clean storm name
    def extract_emdat_name(event_name):
        if pd.isna(event_name):
            return ""
        event_str = str(event_name)
        match = re.search(r"'([^']+)'", event_str)
        if match:
            return match.group(1).upper().strip()
        cleaned = re.sub(
            r'(?i)\b(tropical cyclone|tropical depression|typhoon|hurricane|cyclone|tropical storm)\b',
            '',
            event_str
        )
        return cleaned.replace("'", "").strip().upper()

    # Helper: Parse dates
    def parse_emdat_date(row, prefix):
        year = row.get(f'{prefix} Year') or row.get('Year') or row.get('Start Year')
        month = row.get(f'{prefix} Month', 1)
        day = row.get(f'{prefix} Day', 1)

        if pd.isna(year):
            return pd.NaT
        try:
            return pd.to_datetime(f"{int(year)}-{int(month if pd.notna(month) and month != 0 else 1):02d}-{int(day if pd.notna(day) and day != 0 else 1):02d}")
        except Exception:
            return pd.NaT

    print("3. Preprocessing match keys and dates...")
    event_col = 'Event Name' if 'Event Name' in emdat_df.columns else [c for c in emdat_df.columns if 'Name' in c][0]
    emdat_df['_match_name'] = emdat_df[event_col].apply(extract_emdat_name)

    if 'start_date' in emdat_df.columns:
        emdat_df['_emdat_start'] = pd.to_datetime(emdat_df['start_date'], errors='coerce')
        emdat_df['_emdat_end'] = pd.to_datetime(emdat_df['end_date'], errors='coerce').fillna(emdat_df['_emdat_start'])
    elif 'Start Date' in emdat_df.columns:
        emdat_df['_emdat_start'] = pd.to_datetime(emdat_df['Start Date'], errors='coerce')
        emdat_df['_emdat_end'] = pd.to_datetime(emdat_df.get('End Date', emdat_df['Start Date']), errors='coerce').fillna(emdat_df['_emdat_start'])
    else:
        emdat_df['_emdat_start'] = emdat_df.apply(lambda r: parse_emdat_date(r, 'Start'), axis=1)
        emdat_df['_emdat_end'] = emdat_df.apply(lambda r: parse_emdat_date(r, 'End'), axis=1).fillna(emdat_df['_emdat_start'])

    emdat_df['_match_year'] = emdat_df['_emdat_start'].dt.year.fillna(
        pd.to_numeric(emdat_df.get('Start Year', emdat_df.get('Year')), errors='coerce')
    )

    # Parse NOAA storm keys
    noaa_df['_match_name'] = noaa_df['hurricane_name'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.upper().str.strip()
    noaa_df['_match_year'] = pd.to_numeric(noaa_df['season'], errors='coerce')
    noaa_df['_noaa_start'] = pd.to_datetime(noaa_df['start_date'], errors='coerce')
    noaa_df['_noaa_end'] = pd.to_datetime(noaa_df['end_date'], errors='coerce')

    noaa_cols_to_add = [c for c in noaa_df.columns if not c.startswith('_')]

    print("4. Performing 1-to-1 matching from NOAA onto EM-DAT rows...")
    buffer = pd.Timedelta(days=4)
    matched_noaa_rows = []

    for idx, em_row in emdat_df.iterrows():
        e_name = em_row['_match_name']
        e_year = em_row['_match_year']
        e_start = em_row['_emdat_start']
        e_end = em_row['_emdat_end']

        candidates = noaa_df[(noaa_df['_match_name'] == e_name) & (noaa_df['_match_year'] == e_year)]

        selected_noaa = None
        if not candidates.empty:
            if len(candidates) == 1 or pd.isna(e_start):
                selected_noaa = candidates.iloc[0]
            else:
                for _, n_row in candidates.iterrows():
                    n_s, n_e = n_row['_noaa_start'], n_row['_noaa_end']
                    if pd.notna(n_s) and (n_s - buffer <= e_end) and (n_e + buffer >= e_start):
                        selected_noaa = n_row
                        break
                if selected_noaa is None:
                    selected_noaa = candidates.iloc[0]

        if selected_noaa is not None:
            noaa_dict = selected_noaa[noaa_cols_to_add].to_dict()
            prefixed_dict = {f"noaa_{k}" if k in emdat_df.columns else k: v for k, v in noaa_dict.items()}
            prefixed_dict['noaa_matched'] = True
        else:
            prefixed_dict = {'noaa_matched': False}

        matched_noaa_rows.append(prefixed_dict)

    noaa_appended_df = pd.DataFrame(matched_noaa_rows)
    combined_df = pd.concat([emdat_df.reset_index(drop=True), noaa_appended_df], axis=1)

    # Filter: Keep ONLY rows where a NOAA match occurred
    final_df = combined_df[combined_df['noaa_matched'] == True].copy()

    # Ensure start_date exists for build_hurricane_market_dataset.py
    if 'start_date' not in final_df.columns or final_df['start_date'].isna().all():
        if 'Start Date' in final_df.columns:
            final_df['start_date'] = final_df['Start Date']
        else:
            final_df['start_date'] = final_df['_emdat_start'].dt.strftime('%Y-%m-%d')

    # ---------------------------------------------------------
    # 5. Drop Unwanted Columns
    # ---------------------------------------------------------
    columns_to_remove = [
        'Entry Date', 'Last Update', 'Start Year', 'Start Month', 'Start Day',
        'End Year', 'End Month', 'End Day', 'Latitude', 'Longitude',
        'River Basin', 'Magnitude Scale', 'Origin', 'Associated Types',
        'Location', 'Disaster Subgroup', 'Disaster Type', 'Disaster Subtype',
        'Event Name', 'Country', 'Subregion', 'country', 'subregion'
    ]

    internal_cols = ['_match_name', '_match_year', '_emdat_start', '_emdat_end', 'noaa_matched']
    all_drop_cols = columns_to_remove + internal_cols

    final_df = final_df.drop(columns=[c for c in all_drop_cols if c in final_df.columns])

    # ---------------------------------------------------------
    # 6. Fill Missing (NaN) Values
    # ---------------------------------------------------------
    zero_fill_columns = [
        "AID Contribution ('000 US$)",
        "Total Deaths",
        "No. Injured",
        "No. Affected",
        "Total Affected",
        "Insured Damage ('000 US$)",
        "Insured Damage, Adjusted ('000 US$)",
        "Total Damage ('000 US$)",
        "Total Damage, Adjusted ('000 US$)",
        "landfall_time",
        "landfall_lat",
        "landfall_lon",
        "landfall_wind_kts"
    ]

    for col in zero_fill_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    # Option C: Estimate missing min_pressure_mb based on Wind Speed
    if 'min_pressure_mb' in final_df.columns:
        def estimate_pressure(row):
            pressure = row['min_pressure_mb']
            if pd.notna(pressure) and float(pressure) > 0:
                return float(pressure)

            wind_speed = 0.0
            for w_col in ['landfall_wind_kts', 'max_wind_kts', 'max_wind', 'wind_kts']:
                if w_col in row and pd.notna(row[w_col]) and float(row[w_col]) > 0:
                    wind_speed = float(row[w_col])
                    break

            if wind_speed >= 137:
                return 915.0
            elif wind_speed >= 113:
                return 935.0
            elif wind_speed >= 96:
                return 950.0
            elif wind_speed >= 83:
                return 968.0
            elif wind_speed >= 64:
                return 980.0
            elif wind_speed >= 34:
                return 995.0
            elif wind_speed > 0:
                return 1008.0
            else:
                return 985.0

        final_df['min_pressure_mb'] = final_df.apply(estimate_pressure, axis=1)

    # ---------------------------------------------------------
    # 7. Move Required Identifier Columns to Front (Left Side)
    # ---------------------------------------------------------
    id_cols = ['storm_id', 'hurricane_name', 'season', 'start_date']

    first_cols = [c for c in id_cols if c in final_df.columns]
    remaining_cols = [c for c in final_df.columns if c not in first_cols]

    final_df = final_df[first_cols + remaining_cols]

    # ---------------------------------------------------------
    # 8. Output Final CSV
    # ---------------------------------------------------------
    print("8. Saving final processed dataset...")
    # Save to requested output path
    final_df.to_csv(output_path, index=False)

    print("\nProcess Complete!")
    print(f"Base EM-DAT rows:       {original_emdat_count}")
    print(f"Matched rows saved:     {len(final_df)}")
    print(f"Leading columns:        {first_cols}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, 'data', 'disaster_data')
    emdat_file = os.path.join(data_dir, 'emdat_tropical_cyclones.csv')
    noaa_file = os.path.join(data_dir, 'worldwide_hurricanes_noaa_complete.csv')
    output_file = os.path.join(data_dir, 'emdat_noaa_matched.csv')

    merge_reorder_and_clean_complete(emdat_file, noaa_file, output_file)
