"""Linear regression of each sector's 5-day abnormal return on market conditions.

Usage from the repository root:
    python code/modeling/sector_regression.py

Input: data/regression_datasets/<sector>_cat1plus_regression_dataset.csv
       (built by code/data_cleaning/clean_sector_regression_datasets.py)
Output: reports/regression_results.csv       - fit and out-of-sample metrics per sector
        reports/regression_coefficients.csv  - coefficients with robust p-values

Each storm made US landfall at Category 1+. Features are measured on the
hurricane's event date only:
    spy_prev_5d_return - SPY return over the 5 sessions before the event
    log_vix            - natural log of the VIX
    <sector>_beta      - the sector's beta to SPY
The target is the sector's 5-day beta-adjusted abnormal return after the event.

With ~40 storms per sector a held-out test split would be a handful of rows, so
out-of-sample performance is measured with leave-one-out cross-validation and
compared against predicting the training mean (the no-skill baseline).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
# Imported directly: statsmodels.api fails to import with newer scipy versions.
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "regression_datasets"
RESULTS_PATH = ROOT / "reports" / "regression_results.csv"
COEFFICIENTS_PATH = ROOT / "reports" / "regression_coefficients.csv"

SECTORS = ["energy", "utilities", "industrials", "financials"]


def features(sector):
    return ["spy_prev_5d_return", "log_vix", f"{sector}_beta"]


def load_sector(sector):
    df = pd.read_csv(DATA_DIR / f"{sector}_cat1plus_regression_dataset.csv")
    df["log_vix"] = np.log(df["vix"])
    return df.dropna(subset=features(sector) + [f"{sector}_5d_abnormal_return"])


def run_sector(sector):
    df = load_sector(sector)
    X, y = df[features(sector)], df[f"{sector}_5d_abnormal_return"]

    # In-sample fit with heteroskedasticity-robust (HC3) standard errors.
    ols = OLS(y, add_constant(X)).fit(cov_type="HC3")

    # Out-of-sample: each storm is predicted by a model fit on the other storms.
    loo_pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    loo_mean = np.array([y.drop(i).mean() for i in y.index])
    sse_model = ((y - loo_pred) ** 2).sum()
    sse_mean = ((y - loo_mean) ** 2).sum()

    result = {
        "sector": sector,
        "n": len(df),
        "r2": ols.rsquared,
        "adj_r2": ols.rsquared_adj,
        "f_pvalue": ols.f_pvalue,
        "loo_r2": 1 - sse_model / sse_mean,
        "loo_rmse": np.sqrt(sse_model / len(y)),
        "mean_baseline_rmse": np.sqrt(sse_mean / len(y)),
        "loo_direction_accuracy": (np.sign(loo_pred) == np.sign(y)).mean(),
        "positive_rate": (y > 0).mean(),
    }
    coefficients = pd.DataFrame({
        "sector": sector,
        "term": ols.params.index,
        "coef": ols.params.values,
        "std_err": ols.bse.values,
        "p_value": ols.pvalues.values,
    })

    print(f"\n{sector} (n={len(df)})")
    print(f"  R2={ols.rsquared:.3f}  adj R2={ols.rsquared_adj:.3f}  F p={ols.f_pvalue:.3f}")
    print(f"  LOO R2={result['loo_r2']:.3f}  LOO RMSE={result['loo_rmse']:.4f} "
          f"(mean baseline {result['mean_baseline_rmse']:.4f})  "
          f"direction acc={result['loo_direction_accuracy']:.3f}")
    print(coefficients.drop(columns="sector").round(4).to_string(index=False))
    return result, coefficients


def main():
    results, coefficients = [], []
    for sector in SECTORS:
        result, coefs = run_sector(sector)
        results.append(result)
        coefficients.append(coefs)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    pd.concat(coefficients).to_csv(COEFFICIENTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")
    print(f"Saved coefficients to {COEFFICIENTS_PATH}")


if __name__ == "__main__":
    main()
