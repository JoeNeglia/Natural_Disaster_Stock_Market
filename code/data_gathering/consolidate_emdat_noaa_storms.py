"""Collapse the EM-DAT/NOAA matched dataset to one row per storm.

Usage from the repository root:
    python code/data_gathering/consolidate_emdat_noaa_storms.py

Input: data/disaster_data/emdat_noaa_matched.csv (one row per storm x country)
Output: data/disaster_data/emdat_noaa_storms.csv (one row per storm_id)

EM-DAT records a separate disaster entry for every country a storm affects, so
the matched file repeats each storm once per country. Storm attributes (NOAA
track data, CPI) are identical across those rows and are kept as-is. Impact
columns (deaths, affected, damage, aid) are summed across countries. Country
information is kept as a list of ISO codes, a country count, and the most
common region.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "disaster_data" / "emdat_noaa_matched.csv"
DEFAULT_OUTPUT = ROOT / "data" / "disaster_data" / "emdat_noaa_storms.csv"

# Per-country impacts: totals across all affected countries.
SUM_COLUMNS = [
    "AID Contribution ('000 US$)",
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "Total Affected",
    "Insured Damage ('000 US$)",
    "Insured Damage, Adjusted ('000 US$)",
    "Total Damage ('000 US$)",
    "Total Damage, Adjusted ('000 US$)",
]

# Storm-level attributes: identical on every row of a storm.
FIRST_COLUMNS = [
    "hurricane_name",
    "season",
    "Classification Key",
    "CPI",
    "peak_category",
    "peak_wind_kts",
    "peak_wind_mph",
    "min_pressure_mb",
    "start_date",
    "end_date",
    "duration_days",
    "landfall_status",
    "landfall_category",
    "landfall_lat",
    "landfall_lon",
    "landfall_wind_kts",
]


def _join_unique(values: pd.Series) -> str:
    return ";".join(dict.fromkeys(values.dropna().astype(str)))


def consolidate(matched: pd.DataFrame) -> pd.DataFrame:
    grouped = matched.groupby("storm_id", sort=False)

    varying = [c for c in FIRST_COLUMNS if (grouped[c].nunique(dropna=False) > 1).any()]
    if varying:
        raise ValueError(f"Expected storm-level columns to be constant per storm: {varying}")

    storms = grouped[FIRST_COLUMNS].first()
    storms[SUM_COLUMNS] = grouped[SUM_COLUMNS].sum(min_count=1)
    # "2017-0381-USA" -> "2017-0381": the EM-DAT event id without the country suffix.
    storms["DisNo."] = grouped["DisNo."].agg(lambda s: _join_unique(s.str.rsplit("-", n=1).str[0]))
    storms["ISO"] = grouped["ISO"].agg(_join_unique)
    storms["n_countries"] = grouped["ISO"].nunique()
    storms["Region"] = grouped["Region"].agg(lambda s: s.mode().iloc[0])
    storms["n_regions"] = grouped["Region"].nunique()

    storms = storms.reset_index().sort_values(["start_date", "storm_id"])
    return storms[[c for c in matched.columns if c in storms.columns] + ["n_countries", "n_regions"]]


def main() -> None:
    matched = pd.read_csv(DEFAULT_INPUT)
    storms = consolidate(matched)
    storms.to_csv(DEFAULT_OUTPUT, index=False)
    print(f"{len(matched)} storm-country rows -> {len(storms)} storms")
    print(f"Saved to {DEFAULT_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
