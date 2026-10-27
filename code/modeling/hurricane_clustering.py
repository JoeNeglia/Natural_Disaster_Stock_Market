"""Cluster landfalling hurricanes by physical traits, then compare sector returns by cluster.

Usage from the repository root:
    python code/modeling/hurricane_clustering.py

Input: data/sector_datasets/<sector>_noaa_landfall_dataset.csv
       (built by code/data_gathering/build_noaa_sector_datasets.py)
Outputs in reports/:
    cluster_k_selection.csv     silhouette score for each candidate k
    cluster_profiles.csv        average storm traits and region mix per cluster
    cluster_sector_returns.csv  sector abnormal returns per cluster, with tests
    hurricane_clusters.csv      cluster assignment for every storm

Storms are clustered with k-means on standardized storm features only. Sector
returns are never used for clustering; they are compared across clusters
afterwards to see whether storm type relates to the market reaction.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "sector_datasets"
REPORTS_DIR = ROOT / "reports"

SECTORS = ["energy", "utilities", "industrials", "financials"]
CATEGORY_ORDER = {
    "Tropical Depression": 0, "Tropical Storm": 1, "Category 1": 2, "Category 2": 3,
    "Category 3": 4, "Category 4": 5, "Category 5": 6,
}
CLUSTER_FEATURES = [
    "peak_category", "peak_wind_kts", "duration_days", "landfall_category", "landfall_wind_kts",
]
CANDIDATE_K = range(2, 9)
K = None  # Set to an integer to override the silhouette choice.
RANDOM_STATE = 42


def load_storms():
    """One row per storm: hurricane features plus every sector's 5-day abnormal return."""
    storms = None
    for sector in SECTORS:
        df = pd.read_csv(DATA_DIR / f"{sector}_noaa_landfall_dataset.csv")
        target = f"{sector}_5d_abnormal_return"
        if storms is None:
            storms = df[[c for c in df.columns if not c.startswith(tuple(SECTORS)) and c not in
                         ("spy_prev_5d_return", "vix")]].copy()
        storms[target] = df[target].values

    for col in ["peak_category", "landfall_category"]:
        storms[col] = storms[col].map(CATEGORY_ORDER)
    # Share of peak strength the storm still had at landfall.
    storms["landfall_to_peak_ratio"] = storms["landfall_wind_kts"] / storms["peak_wind_kts"]
    return storms


def choose_k(X):
    scores = []
    for k in CANDIDATE_K:
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
        scores.append({"k": k, "silhouette": silhouette_score(X, labels)})
    scores = pd.DataFrame(scores)
    scores.to_csv(REPORTS_DIR / "cluster_k_selection.csv", index=False)
    print("Silhouette by k (higher = better separated):")
    print(scores.round(3).to_string(index=False))
    return int(scores.loc[scores["silhouette"].idxmax(), "k"]) if K is None else K


def profile_clusters(storms):
    region_cols = [c for c in storms.columns if c.startswith("region_")]
    profile = storms.groupby("cluster")[CLUSTER_FEATURES + ["landfall_to_peak_ratio"]].mean()
    profile.insert(0, "n_storms", storms.groupby("cluster").size())
    # Region columns are 0/1, so their mean is the share of the cluster in each region.
    regions = storms.groupby("cluster")[region_cols].mean()
    regions.columns = [c.replace("region_", "share_") for c in region_cols]
    return pd.concat([profile, regions], axis=1)


def compare_sector_returns(storms):
    rows = []
    for sector in SECTORS:
        target = f"{sector}_5d_abnormal_return"
        data = storms.dropna(subset=[target])
        groups = [g[target].values for _, g in data.groupby("cluster")]
        _, kruskal_p = kruskal(*groups)
        direction = pd.crosstab(data["cluster"], data[target] > 0)
        _, chi2_p, _, _ = chi2_contingency(direction)

        for cluster, g in data.groupby("cluster"):
            rows.append({
                "sector": sector,
                "cluster": cluster,
                "n_storms": len(g),
                "mean_abnormal_return": g[target].mean(),
                "median_abnormal_return": g[target].median(),
                "pct_positive": (g[target] > 0).mean(),
                "kruskal_p_value": kruskal_p,
                "direction_chi2_p_value": chi2_p,
            })
    return pd.DataFrame(rows)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    storms = load_storms()
    features = CLUSTER_FEATURES + ["landfall_to_peak_ratio"]
    X = StandardScaler().fit_transform(storms[features])

    k = choose_k(X)
    storms["cluster"] = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
    print(f"\nUsing k = {k}")

    profiles = profile_clusters(storms)
    profiles.to_csv(REPORTS_DIR / "cluster_profiles.csv")
    print("\nCluster profiles (feature means, region shares)")
    print(profiles.round(2).T.to_string())

    returns = compare_sector_returns(storms)
    returns.to_csv(REPORTS_DIR / "cluster_sector_returns.csv", index=False)
    print("\nSector 5-day abnormal returns by cluster")
    for sector, g in returns.groupby("sector", sort=False):
        print(f"\n{sector}  (Kruskal-Wallis p = {g['kruskal_p_value'].iloc[0]:.3f}, "
              f"direction chi-square p = {g['direction_chi2_p_value'].iloc[0]:.3f})")
        print(g[["cluster", "n_storms", "mean_abnormal_return", "median_abnormal_return",
                 "pct_positive"]].round(4).to_string(index=False))

    storms[["storm_id", "hurricane_name", "start_date", "cluster"]].to_csv(
        REPORTS_DIR / "hurricane_clusters.csv", index=False)
    print(f"\nSaved outputs to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
