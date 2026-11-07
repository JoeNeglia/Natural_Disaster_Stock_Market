# Event study of top 10 hurricanes that made landfall on USA. Trying to establish baseline effects of hurricanes


"""
TOP 10 HURRICANE EVENT STUDY
============================

Research question:
How do different U.S. stock market sectors react to the
10 strongest hurricanes at U.S. landfall?

Hurricane ranking:
    1. Landfall category (5 is highest)
    2. Wind speed at landfall (highest first)
    3. Landfall date (earliest first)

Outputs:
    - Top 10 hurricane list
    - Daily abnormal returns
    - Event-level cumulative abnormal returns (CAR)
    - Statistical summary
    - CAAR line chart
    - Sector comparison with confidence intervals
    - Individual hurricane heatmap

Run this file from your repository in VS Code.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. SETTINGS
# ============================================================

# Automatically locate the repository containing this script.
PROJECT = Path(__file__).resolve().parent

HURRICANE_FILE = PROJECT / "data/disaster_data/us_landfall_hurricanes_hurdat.csv"

STOCK_FILE = PROJECT / "SP500Last25Years/sp500_stocks.csv"
COMPANY_FILE = PROJECT / "SP500Last25Years/sp500_companies.csv"
MARKET_FILE = PROJECT / "data/stock_data/market_raw.csv"

OUTPUT = PROJECT / "top10_event_study_results"
OUTPUT.mkdir(exist_ok=True)

SECTORS = [
    "Energy",
    "Utilities",
    "Financials",
    "Industrials",
]

# Days are measured in stock-market trading sessions.
ESTIMATION_START = -120
ESTIMATION_END = -21

EVENT_START = -5
EVENT_END = 10

# Minimum number of stocks required in a sector portfolio.
MIN_STOCKS = 10


# ============================================================
# 2. SELECT THE TOP 10 HURRICANES
# ============================================================


def select_hurricanes():
    """Select the strongest hurricanes by landfall category."""

    storms = pd.read_csv(HURRICANE_FILE)

    # Convert dates and wind speeds to usable data types.
    storms["landfall_date"] = pd.to_datetime(
        storms["first_us_landfall_date"], errors="coerce"
    )

    storms["landfall_wind_mph"] = pd.to_numeric(
        storms["landfall_wind_mph"], errors="coerce"
    )

    # Convert "Category 5" into the number 5.
    storms["category_number"] = pd.to_numeric(
        storms["landfall_category"].astype(str).str.extract(r"([1-5])", expand=False),
        errors="coerce",
    )

    # Remove incomplete records.
    storms = storms.dropna(
        subset=[
            "hurdat_id",
            "landfall_date",
            "category_number",
            "landfall_wind_mph",
        ]
    )

    # Keep one observation per hurricane.
    storms = storms.sort_values("landfall_date").drop_duplicates("hurdat_id")

    # Rank by category, wind speed, and landfall date.
    storms = storms.sort_values(
        by=[
            "category_number",
            "landfall_wind_mph",
            "landfall_date",
        ],
        ascending=[False, False, True],
    )

    top10 = storms.head(10).copy()

    columns = [
        "hurricane_name",
        "landfall_date",
        "landfall_category",
        "landfall_wind_mph",
    ]

    print("\nTOP 10 HURRICANES")
    print(top10[columns].to_string(index=False))

    top10.to_csv(OUTPUT / "top10_hurricanes.csv", index=False)

    return top10


# ============================================================
# 3. PREPARE STOCK MARKET DATA
# ============================================================


def prepare_stock_data():
    """
    Calculate daily stock returns and combine stocks into
    four equal-weight sector portfolios.

    Daily return = (Today's Close / Yesterday's Close) - 1

    Equal-weight means every available stock contributes
    equally to its sector's daily return.
    """

    companies = pd.read_csv(COMPANY_FILE)

    # Only include companies in our four sectors.
    companies = companies[companies["sector"].isin(SECTORS)][
        ["symbol", "sector"]
    ].drop_duplicates("symbol")

    symbol_to_sector = companies.set_index("symbol")["sector"]

    # The stock file is large, so load it in chunks.
    stock_chunks = []

    for chunk in pd.read_csv(
        STOCK_FILE, usecols=["date", "symbol", "close"], chunksize=350_000
    ):
        chunk = chunk[chunk["symbol"].isin(symbol_to_sector.index)].copy()

        stock_chunks.append(chunk)

    stocks = pd.concat(stock_chunks, ignore_index=True)

    stocks["date"] = pd.to_datetime(stocks["date"])
    stocks["close"] = pd.to_numeric(stocks["close"], errors="coerce")

    stocks = stocks.dropna(subset=["date", "close"])
    stocks = stocks.drop_duplicates(["symbol", "date"])
    stocks = stocks.sort_values(["symbol", "date"])

    # Calculate each company's daily percentage return.
    stocks["return"] = stocks.groupby("symbol")["close"].pct_change(fill_method=None)

    # Avoid treating a long gap in prices as a daily return.
    previous_date = stocks.groupby("symbol")["date"].shift()

    long_gap = (stocks["date"] - previous_date).dt.days > 5

    stocks.loc[long_gap, "return"] = np.nan

    # Assign each company to its sector.
    stocks["sector"] = stocks["symbol"].map(symbol_to_sector)

    stocks["return"] = stocks["return"].replace([np.inf, -np.inf], np.nan)

    # Average daily returns within each sector.
    sector_daily = (
        stocks.groupby(["date", "sector"])
        .agg(sector_return=("return", "mean"), stock_count=("return", "count"))
        .reset_index()
    )

    sector_returns = sector_daily.pivot(
        index="date", columns="sector", values="sector_return"
    )

    stock_counts = sector_daily.pivot(
        index="date", columns="sector", values="stock_count"
    )

    # Exclude sector-days with too few available stocks.
    sector_returns = sector_returns.where(stock_counts >= MIN_STOCKS)

    # Load SPY benchmark data.
    market = pd.read_csv(MARKET_FILE)

    market["Date"] = pd.to_datetime(market["Date"])
    market = market.sort_values("Date").drop_duplicates("Date")
    market = market.set_index("Date")

    market["SPY_return"] = pd.to_numeric(
        market["SPY_Close"], errors="coerce"
    ).pct_change(fill_method=None)

    # Align sectors with SPY trading sessions.
    data = sector_returns.reindex(market.index)
    data["SPY"] = market["SPY_return"]

    return data


# ============================================================
# 4. CALCULATE ABNORMAL RETURNS
# ============================================================


def calculate_abnormal_returns(storms, market_data):
    """
    For each hurricane and sector:

    1. Find the first trading session on/after landfall.
    2. Estimate normal returns using historical data.
    3. Predict expected returns around landfall.
    4. Calculate abnormal returns.

    Market model:
        Expected Return = alpha + beta * SPY Return

    Abnormal Return:
        Actual Return - Expected Return
    """

    results = []
    exclusions = []

    for storm in storms.itertuples(index=False):
        # Find the first market session on/after landfall.
        event_index = market_data.index.searchsorted(storm.landfall_date)

        # Make sure the dataset contains enough history
        # and enough trading days after the event.
        if event_index < abs(ESTIMATION_START) or event_index + EVENT_END >= len(
            market_data
        ):
            exclusions.append((storm.hurricane_name, "Date range"))
            continue

        trading_date = market_data.index[event_index]

        # Reject implausibly distant trading dates.
        if (trading_date - storm.landfall_date).days > 4:
            exclusions.append((storm.hurricane_name, "No nearby session"))
            continue

        print(f"\nAnalyzing {storm.hurricane_name}")

        for sector in SECTORS:
            # Historical data used to estimate alpha and beta.
            estimation = market_data.iloc[
                event_index + ESTIMATION_START : event_index + ESTIMATION_END + 1
            ][["SPY", sector]].dropna()

            # Data surrounding hurricane landfall.
            event_window = market_data.iloc[
                event_index + EVENT_START : event_index + EVENT_END + 1
            ][["SPY", sector]].copy()

            # Require at least 80 valid estimation observations.
            if (
                len(estimation) < 80
                or len(event_window) != 16
                or event_window.isna().any().any()
            ):
                exclusions.append((storm.hurricane_name, f"Missing {sector} data"))
                continue

            # Estimate beta and alpha using ordinary least squares.
            x = estimation["SPY"].to_numpy()
            y = estimation[sector].to_numpy()

            beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
            alpha = y.mean() - beta * x.mean()

            # Calculate expected and abnormal returns.
            expected = alpha + beta * event_window["SPY"]
            abnormal = event_window[sector] - expected

            # Save each trading day's result.
            for relative_day, (date, ar) in enumerate(
                abnormal.items(), start=EVENT_START
            ):
                results.append(
                    {
                        "hurricane": storm.hurricane_name,
                        "hurdat_id": storm.hurdat_id,
                        "category": storm.category_number,
                        "wind_mph": storm.landfall_wind_mph,
                        "landfall_date": storm.landfall_date,
                        "trading_date": date,
                        "sector": sector,
                        "relative_day": relative_day,
                        "abnormal_return": ar,
                    }
                )

    if not results:
        raise ValueError("No valid hurricane event windows found.")

    daily = pd.DataFrame(results)

    daily.to_csv(OUTPUT / "daily_abnormal_returns.csv", index=False)

    pd.DataFrame(exclusions, columns=["hurricane", "reason"]).to_csv(
        OUTPUT / "exclusions.csv", index=False
    )

    return daily


# ============================================================
# 5. CALCULATE CAR AND STATISTICS
# ============================================================


def calculate_statistics(daily):
    """
    CAR = sum of abnormal returns over an event window.

    CAAR = average CAR across hurricanes.

    We calculate CAR for each hurricane separately,
    then calculate sector averages and statistical tests.
    """

    # Main analysis: landfall day through 10 sessions later.
    post_event = daily[daily["relative_day"].between(0, 10)]

    # One CAR per hurricane and sector.
    cars = (
        post_event.groupby(["hurdat_id", "hurricane", "sector"], as_index=False)[
            "abnormal_return"
        ]
        .sum()
        .rename(columns={"abnormal_return": "CAR"})
    )

    cars.to_csv(OUTPUT / "hurricane_CAR.csv", index=False)

    summary = []

    for sector in SECTORS:
        values = cars.loc[cars["sector"] == sector, "CAR"]

        n = len(values)
        mean = values.mean()
        std = values.std(ddof=1)

        # Standard error and one-sample t-test.
        if n >= 2 and std > 0:
            se = std / np.sqrt(n)
            t_stat = mean / se
            p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)

            margin = stats.t.ppf(0.975, n - 1) * se
            ci_low = mean - margin
            ci_high = mean + margin
        else:
            t_stat = p_value = ci_low = ci_high = np.nan

        summary.append(
            {
                "sector": sector,
                "hurricanes": n,
                "mean_CAR_pct": mean * 100,
                "std_CAR_pct": std * 100,
                "t_statistic": t_stat,
                "p_value": p_value,
                "CI_low_pct": ci_low * 100,
                "CI_high_pct": ci_high * 100,
                "positive_pct": (values > 0).mean() * 100,
            }
        )

    summary = pd.DataFrame(summary)

    summary.to_csv(OUTPUT / "statistical_summary.csv", index=False)

    print("\nSTATISTICAL RESULTS")
    print(summary.round(4).to_string(index=False))

    return cars, summary


# ============================================================
# 6. PLOT CUMULATIVE AVERAGE ABNORMAL RETURNS
# ============================================================


def plot_caar(daily):
    """
    Plot the average accumulated abnormal return for each
    sector from five days before to ten days after landfall.
    """

    average_daily = (
        daily.groupby(["sector", "relative_day"])["abnormal_return"]
        .mean()
        .reset_index()
    )

    average_daily["CAAR"] = (
        average_daily.groupby("sector")["abnormal_return"].cumsum() * 100
    )

    plt.figure(figsize=(11, 6))

    for sector in SECTORS:
        sector_data = average_daily[average_daily["sector"] == sector]

        plt.plot(
            sector_data["relative_day"], sector_data["CAAR"], marker="o", label=sector
        )

    plt.axvline(0, color="gray", linestyle="--", label="Landfall")
    plt.axhline(0, color="black", linewidth=0.7)

    plt.title("Top 10 Hurricanes: Cumulative Abnormal Returns")
    plt.xlabel("Trading Days Relative to Landfall")
    plt.ylabel("CAAR (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT / "CAAR_by_sector.png", dpi=200)
    plt.show()


# ============================================================
# 7. PLOT SECTOR COMPARISON WITH CONFIDENCE INTERVALS
# ============================================================


def plot_sector_comparison(summary):
    """Compare mean CAR and 95% confidence intervals."""

    plt.figure(figsize=(10, 6))

    positions = np.arange(len(summary))

    # Calculate lower and upper error bar lengths.
    lower = summary["mean_CAR_pct"] - summary["CI_low_pct"]
    upper = summary["CI_high_pct"] - summary["mean_CAR_pct"]

    plt.errorbar(
        summary["mean_CAR_pct"],
        positions,
        xerr=np.array([lower, upper]),
        fmt="o",
        capsize=5,
    )

    plt.yticks(positions, summary["sector"])
    plt.axvline(0, color="gray", linestyle="--")

    plt.title("Mean Sector CAR with 95% Confidence Intervals")
    plt.xlabel("Mean CAR, Day 0 to +10 (%)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT / "sector_comparison.png", dpi=200)
    plt.show()


# ============================================================
# 8. PLOT INDIVIDUAL HURRICANE RESULTS
# ============================================================


def plot_hurricane_heatmap(cars, storms):
    """Show how each hurricane affected each sector."""

    heatmap = cars.pivot(index="hurdat_id", columns="sector", values="CAR")

    # Preserve the strongest-to-weakest hurricane ranking.
    storm_order = storms["hurdat_id"].tolist()
    heatmap = heatmap.reindex(index=storm_order, columns=SECTORS)

    names = (
        storms.set_index("hurdat_id").reindex(storm_order)["hurricane_name"].tolist()
    )

    plt.figure(figsize=(10, 7))

    plt.imshow(heatmap.to_numpy() * 100, cmap="RdBu", aspect="auto", vmin=-10, vmax=10)

    plt.colorbar(label="CAR (%)")
    plt.xticks(range(len(SECTORS)), SECTORS)
    plt.yticks(range(len(names)), names)

    plt.title("Individual Hurricane CAR: Day 0 to +10")
    plt.tight_layout()

    plt.savefig(OUTPUT / "hurricane_heatmap.png", dpi=200)
    plt.show()


# ============================================================
# 9. RUN THE COMPLETE STUDY
# ============================================================


def main():

    print("Starting Top 10 Hurricane Event Study...")

    # Step 1: Identify the ten strongest hurricanes.
    storms = select_hurricanes()

    # Step 2: Calculate stock and sector returns.
    market_data = prepare_stock_data()

    # Step 3: Estimate abnormal returns for every event.
    daily = calculate_abnormal_returns(storms, market_data)

    # Step 4: Calculate CAR and statistical results.
    cars, summary = calculate_statistics(daily)

    # Step 5: Generate figures.
    plot_caar(daily)
    plot_sector_comparison(summary)
    plot_hurricane_heatmap(cars, storms)

    print("\nAnalysis complete!")
    print(f"Results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
