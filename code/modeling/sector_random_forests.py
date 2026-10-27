"""Random forest models predicting each sector's 5-day abnormal return direction.

Usage from the repository root:
    python code/modeling/sector_random_forests.py

Input: data/sector_datasets/<sector>_noaa_landfall_dataset.csv
       (built by code/data_gathering/build_noaa_sector_datasets.py)
Output: reports/random_forest_results.csv

Uses the same data, features and chronological split as sector_decision_trees.py.
For each sector two models are trained:
    baseline  - sector market data on the hurricane's event date only
    hurricane - the same sector market data plus the NOAA hurricane features
"""

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from sector_decision_trees import main


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "reports" / "random_forest_results.csv"

# Every sector starts from these settings.
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

# Per-sector adjustments, merged over DEFAULT_PARAMS. Tune each sector here,
# e.g. "energy": {"max_depth": 6}. Applies to both its baseline and hurricane model.
SECTOR_PARAMS = {
    "energy": {},
    "utilities": {},
    "industrials": {},
    "financials": {},
}


if __name__ == "__main__":
    main(RandomForestClassifier, DEFAULT_PARAMS, SECTOR_PARAMS, RESULTS_PATH)
