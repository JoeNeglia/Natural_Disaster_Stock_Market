"""SVM models predicting each sector's 5-day abnormal return direction.

Usage from the repository root:
    python code/modeling/sector_svms.py

Input: data/sector_datasets/<sector>_noaa_landfall_dataset.csv
       (built by code/data_gathering/build_noaa_sector_datasets.py)
Output: reports/svm_results.csv

Uses the same data, features and chronological split as sector_decision_trees.py.
Features are standardized before the SVM; the scaler is fit on the training
split only. For each sector two models are trained:
    baseline  - sector market data on the hurricane's event date only
    hurricane - the same sector market data plus the NOAA hurricane features
"""

from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sector_decision_trees import main


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "reports" / "svm_results.csv"

# Every sector starts from these settings.
DEFAULT_PARAMS = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}

# Per-sector adjustments, merged over DEFAULT_PARAMS. Tune each sector here,
# e.g. "energy": {"C": 10.0}. Applies to both its baseline and hurricane model.
SECTOR_PARAMS = {
    "energy": {},
    "utilities": {},
    "industrials": {},
    "financials": {},
}


def standardized_svm(**params):
    return make_pipeline(StandardScaler(), SVC(**params))


if __name__ == "__main__":
    main(standardized_svm, DEFAULT_PARAMS, SECTOR_PARAMS, RESULTS_PATH)
