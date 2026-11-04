"""Build a storm-level dataset of hurricanes that made US landfall, from NOAA HURDAT2.

Usage from the repository root:
    python code/data_gathering/build_hurdat_us_landfall_dataset.py [--start-year 2000]

Default output: data/disaster_data/us_landfall_hurricanes_hurdat.csv

HURDAT2 is the NHC's official best-track record for the Atlantic and the
Northeast/Central Pacific. Every landfall is an explicit track point (record
identifier "L"), so a storm's US landfalls are its "L" points inside the US.
Only storms with at least one US landfall at hurricane strength (64+ kt) are kept,
and the landfall columns describe the strongest US landfall only - landfalls in
other countries are ignored.

Storm-level features match worldwide_hurricanes_noaa_complete.csv (peak wind,
minimum pressure, dates, duration) plus details of the strongest US landfall.
Each storm is also mapped to its IBTrACS storm_id so it joins to the sector datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd


HURDAT_BASE_URL = "https://www.nhc.noaa.gov/data/hurdat/"
# Latest releases as of October 2026; update when the NHC posts a new season.
HURDAT_FILES = {
    "atlantic": "hurdat2-1851-2025-092326.txt",
    "pacific": "hurdat2-nepac-1949-2025-092926.txt",
}
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.{basin}.list.v04r01.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "disaster_data" / "us_landfall_hurricanes_hurdat.csv"

# HURDAT2 ids start with the basin code. Full names avoid "NA" being read back as missing.
BASIN_NAMES = {"AL": "North Atlantic", "EP": "East Pacific", "CP": "Central Pacific"}
HURRICANE_KTS = 64
# Leading fields of each HURDAT2 track line; the wind radii that follow aren't used.
TRACK_COLUMNS = ["date", "time", "record", "status", "lat", "lon", "wind_kts", "pressure_mb"]


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


def in_usa(lat, lon):
    """True for points on the US mainland Gulf/Atlantic coast or Hawaii."""
    mainland = (lat.between(24.4, 45.5) & lon.between(-97.5, -66.9)
                & ~((lat < 28) & (lon > -79.9))         # Bahamas
                & ~((lon < -97.1) & (lat < 25.95)))     # Mexico, south of the Rio Grande
    hawaii = lat.between(18.5, 22.5) & lon.between(-161, -154.5)
    return mainland | hawaii


def read_hurdat(filename: str) -> pd.DataFrame:
    """Parse a HURDAT2 file into one row per track point, tagged with its storm."""
    url = HURDAT_BASE_URL + filename
    try:
        with urlopen(url) as response:
            lines = response.read().decode("utf-8").splitlines()
    except (OSError, URLError) as error:
        raise RuntimeError(f"Could not download HURDAT2 from {url}") from error

    rows, storm_id, name = [], None, None
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 8:
            # Header line: storm id, name, number of track points.
            storm_id, name = fields[0], fields[1]
            continue
        rows.append([storm_id, name] + fields[:len(TRACK_COLUMNS)])

    track = pd.DataFrame(rows, columns=["hurdat_id", "name"] + TRACK_COLUMNS)
    track["time"] = pd.to_datetime(track["date"] + track["time"].str.zfill(4), format="%Y%m%d%H%M")
    track["lat"] = track["lat"].str[:-1].astype(float) * np.where(track["lat"].str.endswith("S"), -1, 1)
    track["lon"] = track["lon"].str[:-1].astype(float) * np.where(track["lon"].str.endswith("W"), -1, 1)
    numeric = ["wind_kts", "pressure_mb"]
    # -999 (and -99 for older winds) marks a missing value in HURDAT2.
    track[numeric] = track[numeric].apply(pd.to_numeric, errors="coerce").where(lambda v: v > -99)
    return track.drop(columns="date")


def ibtracs_ids() -> pd.Series:
    """Map HURDAT2/ATCF ids (e.g. AL092008) to IBTrACS storm ids."""
    frames = []
    for basin in ["NA", "EP"]:
        url = IBTRACS_URL.format(basin=basin)
        try:
            frames.append(pd.read_csv(url, skiprows=[1], usecols=["SID", "USA_ATCF_ID"],
                                      keep_default_na=False))
        except (OSError, URLError) as error:
            raise RuntimeError(f"Could not download IBTrACS from {url}") from error
    ids = pd.concat(frames)
    ids["USA_ATCF_ID"] = ids["USA_ATCF_ID"].str.strip().str.upper()
    ids = ids[ids["USA_ATCF_ID"] != ""].drop_duplicates("USA_ATCF_ID")
    return ids.set_index("USA_ATCF_ID")["SID"]


def storm_record(hurdat_id: str, storm: pd.DataFrame) -> dict | None:
    landfalls = storm[(storm["record"] == "L") & in_usa(storm["lat"], storm["lon"])]
    if landfalls.empty or landfalls["wind_kts"].max() < HURRICANE_KTS:
        return None

    strongest = landfalls.loc[landfalls["wind_kts"].idxmax()]
    season = int(hurdat_id[-4:])
    peak_wind = storm["wind_kts"].max()
    start, end = storm["time"].min(), storm["time"].max()
    return {
        "hurdat_id": hurdat_id,
        "hurricane_name": f"{storm['name'].iloc[0].title()} ({season})",
        "season": season,
        "basin": BASIN_NAMES.get(hurdat_id[:2], hurdat_id[:2]),
        "peak_category": wind_to_category(peak_wind),
        "peak_wind_kts": peak_wind,
        "peak_wind_mph": round(peak_wind * 1.15078, 1),
        "min_pressure_mb": storm["pressure_mb"].min(),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "duration_days": round((end - start).total_seconds() / 86400.0, 1),
        "us_landfall_count": len(landfalls),
        "first_us_landfall_date": landfalls["time"].min().strftime("%Y-%m-%d"),
        # Strongest US landfall.
        "landfall_category": wind_to_category(strongest["wind_kts"]),
        "landfall_time": strongest["time"].strftime("%Y-%m-%d %H:%M"),
        "landfall_lat": strongest["lat"],
        "landfall_lon": strongest["lon"],
        "landfall_wind_kts": strongest["wind_kts"],
        "landfall_wind_mph": round(strongest["wind_kts"] * 1.15078, 1),
        "landfall_pressure_mb": strongest["pressure_mb"],
    }


def build_us_landfall_dataset(
    start_year: int = 2000,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
) -> Path:
    """Build the US-landfall hurricane CSV and return its output path."""
    output_path = Path(output_csv)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print("1. Fetching NOAA HURDAT2 Atlantic and Northeast Pacific best tracks...")
    track = pd.concat([read_hurdat(f) for f in HURDAT_FILES.values()], ignore_index=True)
    track = track[track["hurdat_id"].str[-4:].astype(int) >= start_year]

    print(f"2. Finding US hurricane landfalls since {start_year}...")
    records = [r for hid, storm in track.groupby("hurdat_id", sort=False)
               if (r := storm_record(hid, storm)) is not None]
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError(f"No US hurricane landfalls found since {start_year}.")

    print("3. Matching storms to IBTrACS ids...")
    result.insert(0, "storm_id", result["hurdat_id"].map(ibtracs_ids()))
    unmatched = result["storm_id"].isna().sum()
    if unmatched:
        print(f"   {unmatched} storm(s) had no IBTrACS match; storm_id left blank.")

    result = result.sort_values(["season", "landfall_time"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"4. Writing {len(result)} US landfalling hurricanes to '{output_path}'...")
    result.to_csv(output_path, index=False)
    print("Dataset generation complete!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a US-landfall hurricane dataset from HURDAT2.")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()
    build_us_landfall_dataset(start_year=args.start_year, output_csv=args.output)
