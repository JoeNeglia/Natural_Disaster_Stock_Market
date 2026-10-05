"""Filter the full EM-DAT export to tropical cyclone records.

Input: data/disaster_data/worldwide_natural_disaster_damage.csv
Output: data/disaster_data/emdat_tropical_cyclones.csv
"""

import pandas as pd
from pathlib import Path

# Define file paths
data_dir = Path(__file__).resolve().parents[2] / 'data' / 'disaster_data'
input_path = data_dir / 'worldwide_natural_disaster_damage.csv'
output_path = data_dir / 'emdat_tropical_cyclones.csv'

def filter_emdat_tropical_cyclones(input_csv, output_csv):
    print(f"Reading dataset from:\n{input_csv}")

    # Read the full EM-DAT CSV
    df = pd.read_csv(input_csv, low_memory=False)

    # Locate the Disaster Subtype column (case-insensitive search)
    subtype_col = None
    for col in df.columns:
        if col.strip().lower() == 'disaster subtype':
            subtype_col = col
            break

    if not subtype_col:
        raise KeyError("Could not find a 'Disaster Subtype' column in the CSV file.")

    # Filter for rows containing 'tropical cyclone' (handles casing/spacing differences)
    filtered_df = df[
        df[subtype_col].astype(str).str.strip().str.lower() == 'tropical cyclone'
    ].copy()

    # Save to new CSV
    filtered_df.to_csv(output_csv, index=False)

    print(f"\nFiltered successfully!")
    print(f"Original rows: {len(df)}")
    print(f"Tropical Cyclone rows: {len(filtered_df)}")
    print(f"Saved new dataset to:\n{output_csv}")

if __name__ == "__main__":
    filter_emdat_tropical_cyclones(input_path, output_path)
