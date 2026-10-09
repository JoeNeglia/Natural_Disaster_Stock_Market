"""Normalize EM-DAT date components into start_date and end_date columns.

Input: data/disaster_data/emdat_tropical_cyclones.csv
Output: data/disaster_data/emdat_tropical_cyclones_formatted_dates.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define file paths
data_dir = Path(__file__).resolve().parents[2] / 'data' / 'disaster_data'
input_csv = data_dir / 'emdat_tropical_cyclones.csv'
output_csv = data_dir / 'emdat_tropical_cyclones_formatted_dates.csv'

def format_emdat_dates(input_path, output_path):
    print("1. Loading dataset...")
    df = pd.read_csv(input_path, low_memory=False)

    # ------------------------------------------------------------------
    # Helper function to construct ISO YYYY-MM-DD date strings
    # Handles missing/NaN months or days by defaulting to 01
    # ------------------------------------------------------------------
    def create_iso_date(row, year_col, month_col, day_col):
        year = row.get(year_col)
        month = row.get(month_col)
        day = row.get(day_col)

        if pd.isna(year):
            return np.nan

        # Fill missing month or day with 1
        m = int(month) if pd.notna(month) and month != 0 else 1
        d = int(day) if pd.notna(day) and day != 0 else 1
        y = int(year)

        return f"{y:04d}-{m:02d}-{d:02d}"

    print("2. Formatting start_date and end_date columns...")
    # Map Start Date (EM-DAT typically provides Start Year, Start Month, Start Day)
    start_year_col = 'Start Year' if 'Start Year' in df.columns else 'Year'
    df['start_date'] = df.apply(
        lambda r: create_iso_date(r, start_year_col, 'Start Month', 'Start Day'),
        axis=1
    )

    # Map End Date (EM-DAT provides End Year, End Month, End Day)
    end_year_col = 'End Year' if 'End Year' in df.columns else start_year_col
    df['end_date'] = df.apply(
        lambda r: create_iso_date(r, end_year_col, 'End Month', 'End Day'),
        axis=1
    )

    # If end_date is missing or invalid, fallback to start_date
    df['end_date'] = df['end_date'].fillna(df['start_date'])

    # ------------------------------------------------------------------
    # Clean up original date component columns
    # ------------------------------------------------------------------
    cols_to_drop = [
        'Start Year', 'Start Month', 'Start Day',
        'End Year', 'End Month', 'End Day', 'Year'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Save formatted dataset
    df.to_csv(output_path, index=False)
    print(f"3. Done! Output saved to:\n{output_path}")

if __name__ == "__main__":
    format_emdat_dates(input_csv, output_csv)
