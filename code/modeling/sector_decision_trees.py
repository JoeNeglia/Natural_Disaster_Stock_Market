"""Decision tree models predicting each sector's 5-day abnormal return direction.

Usage from the repository root:
    python code/modeling/sector_decision_trees.py

Input: data/sector_datasets/<sector>_noaa_landfall_dataset.csv
       (built by code/data_gathering/build_noaa_sector_datasets.py)
Output: reports/decision_tree_results.csv

For each sector two models are trained on the same chronological split:
    baseline  - sector market data on the hurricane's event date only
    hurricane - the same sector market data plus the NOAA hurricane features
The target is 1 when the sector's 5-day abnormal return is positive, else 0.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "sector_datasets"
RESULTS_PATH = ROOT / "reports" / "decision_tree_results.csv"

SECTORS = ["energy", "utilities", "industrials", "financials"]
SECTOR_METRICS = ["prev_5d_return", "20d_volatility", "volume_change", "drawdown", "beta"]
# Columns in the sector datasets that are not model features. Every other
# non-market column comes from the cleaned NOAA file and is a hurricane feature.
NON_FEATURE_COLUMNS = [
    "storm_id", "hurricane_name", "start_date", "end_date",
    "event_trading_date", "event_date_shift_days",
]
CATEGORY_ORDER = {
    "Tropical Depression": 0, "Tropical Storm": 1, "Category 1": 2, "Category 2": 3,
    "Category 3": 4, "Category 4": 5, "Category 5": 6,
}
TEST_FRACTION = 0.2

# Every sector starts from these settings.
DEFAULT_PARAMS = {"max_depth": 4, "min_samples_leaf": 20, "random_state": 42}

# Per-sector adjustments, merged over DEFAULT_PARAMS. Tune each sector here,
# e.g. "energy": {"max_depth": 6}. Applies to both its baseline and hurricane model.
SECTOR_PARAMS = {
    "energy": {},
    "utilities": {},
    "industrials": {},
    "financials": {},
}


def baseline_features(sector):
    return ["spy_prev_5d_return", "vix"] + [f"{sector}_{m}" for m in SECTOR_METRICS]


def hurricane_features(df, sector):
    excluded = set(NON_FEATURE_COLUMNS + baseline_features(sector) + [f"{sector}_5d_abnormal_return", "target"])
    return [col for col in df.columns if col not in excluded]


def load_sector(sector):
    df = pd.read_csv(DATA_DIR / f"{sector}_noaa_landfall_dataset.csv")
    target_col = f"{sector}_5d_abnormal_return"

    for col in ["peak_category", "landfall_category"]:
        df[col] = df[col].map(CATEGORY_ORDER)

    # Drop storms without enough market history for every feature or the target.
    df = df.dropna(subset=baseline_features(sector) + hurricane_features(df, sector) + [target_col])
    df["target"] = (df[target_col] > 0).astype(int)
    return df.sort_values(["event_trading_date", "storm_id"]).reset_index(drop=True)


def chronological_split(df):
    cutoff = int(len(df) * (1 - TEST_FRACTION))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def evaluate(model, features, train, test):
    model.fit(train[features], train["target"])
    pred = model.predict(test[features])
    return {
        "accuracy": accuracy_score(test["target"], pred),
        "balanced_accuracy": balanced_accuracy_score(test["target"], pred),
        "f1": f1_score(test["target"], pred),
    }


def run_sector(sector, model_class=DecisionTreeClassifier, default_params=None, sector_params=None):
    df = load_sector(sector)
    train, test = chronological_split(df)
    default_params = DEFAULT_PARAMS if default_params is None else default_params
    sector_params = SECTOR_PARAMS if sector_params is None else sector_params
    params = {**default_params, **sector_params[sector]}

    # Accuracy from always predicting the training set's most common direction.
    majority = train["target"].mode()[0]
    majority_accuracy = (test["target"] == majority).mean()

    feature_sets = {
        "baseline": baseline_features(sector),
        "hurricane": baseline_features(sector) + hurricane_features(df, sector),
    }
    rows = []
    for name, features in feature_sets.items():
        model = model_class(**params)
        metrics = evaluate(model, features, train, test)
        rows.append({
            "sector": sector,
            "model": name,
            **metrics,
            "majority_class_accuracy": majority_accuracy,
            "n_features": len(features),
            "n_train": len(train),
            "n_test": len(test),
            "train_end": train["event_trading_date"].iloc[-1],
            "test_start": test["event_trading_date"].iloc[0],
            "params": params,
        })

        line = f"  {name:<9} acc={metrics['accuracy']:.3f}"
        # Models without built-in importances (e.g. SVM) only report accuracy.
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=features)
            top = importances[importances > 0].sort_values(ascending=False).head(5)
            line += f"  top features: {', '.join(f'{k} ({v:.2f})' for k, v in top.items()) or 'none'}"
        print(line)
    return rows


def main(model_class=DecisionTreeClassifier, default_params=None, sector_params=None,
         results_path=RESULTS_PATH):
    rows = []
    for sector in SECTORS:
        print(f"\n{sector}")
        rows.extend(run_sector(sector, model_class, default_params, sector_params))

    results = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    summary = results.pivot(index="sector", columns="model", values="accuracy")
    summary["majority_class"] = results.groupby("sector")["majority_class_accuracy"].first()
    summary["hurricane_gain"] = summary["hurricane"] - summary["baseline"]
    print("\nTest accuracy")
    print(summary.round(3).to_string())
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
