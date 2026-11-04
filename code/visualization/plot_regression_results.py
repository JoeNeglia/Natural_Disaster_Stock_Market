"""Plot how each sector moves around hurricanes against what the regression predicts.

Usage from the repository root:
    python code/visualization/plot_regression_results.py

Inputs: data/regression_datasets/<sector>_cat1plus_regression_dataset.csv
        data/stock_data/market_raw.csv
Outputs in reports/figures/:
    regression_event_paths.png         cumulative abnormal return from 10 sessions before
                                       to 5 after each storm, with the model's average forecast
    regression_actual_vs_predicted.png each storm's actual 5-day abnormal return vs the
                                       leave-one-out regression prediction

The abnormal return on day k is the sector's return from the event date minus
the sector's event-date beta times SPY's return over the same span, so day +5
equals the regression target exactly and day 0 is zero by construction.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "modeling"))
from sector_regression import features, load_sector  # noqa: E402


MARKET_PATH = ROOT / "data" / "stock_data" / "market_raw.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

SECTOR_TICKERS = {"energy": "XLE", "financials": "XLF", "industrials": "XLI", "utilities": "XLU"}
PRE_DAYS, POST_DAYS = 10, 5

# Colors (validated light-mode palette: categorical slots 1-2 plus chart chrome).
ACTUAL = "#2a78d6"
MODEL = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
STORM_LINE = "#c3c2b7"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POST_SHADE = "#f0efec"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
})


def load_closes():
    market = pd.read_csv(MARKET_PATH, low_memory=False)
    market["Date"] = pd.to_datetime(market["Date"], errors="coerce")
    market = market.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    return pd.DataFrame({t: pd.to_numeric(market[f"{t}_Close"], errors="coerce")
                         for t in ["SPY", *SECTOR_TICKERS.values()]})


def sector_frame(sector, closes):
    """Storm rows with event position, LOO prediction and the daily abnormal-return path."""
    df = load_sector(sector).reset_index(drop=True)
    X, y = df[features(sector)], df[f"{sector}_5d_abnormal_return"]
    df["predicted"] = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())

    # Event date = first trading day on or after the storm's start date (as in the dataset build).
    positions = closes.index.searchsorted(pd.to_datetime(df["start_date"]))
    sector_close = closes[SECTOR_TICKERS[sector]].to_numpy()
    spy_close = closes["SPY"].to_numpy()
    offsets = np.arange(-PRE_DAYS, POST_DAYS + 1)
    paths = []
    for pos, beta in zip(positions, df[f"{sector}_beta"]):
        window = pos + offsets
        sector_ret = sector_close[window] / sector_close[pos] - 1
        spy_ret = spy_close[window] / spy_close[pos] - 1
        paths.append(sector_ret - beta * spy_ret)
    return df, offsets, np.array(paths)


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


def finish(fig, title, subtitle, handles, filename):
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.945, subtitle, ha="left", fontsize=9, color=INK_SECONDARY)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=2.5, w_pad=3)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_event_paths(data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True)
    for ax, (sector, (df, offsets, paths)) in zip(axes.flat, data.items()):
        ax.axvspan(0, POST_DAYS, color=POST_SHADE, zorder=0)
        for path in paths:
            ax.plot(offsets, path, color=STORM_LINE, linewidth=0.8, alpha=0.7, zorder=1)
        mean_path = paths.mean(axis=0)
        ax.plot(offsets, mean_path, color=ACTUAL, linewidth=2.2, zorder=3)
        # The regression only forecasts day +5, so its average forecast is drawn from 0 to there.
        mean_pred = df["predicted"].mean()
        ax.plot([0, POST_DAYS], [0, mean_pred], color=MODEL, linewidth=2, linestyle=(0, (4, 2)), zorder=4)
        ax.scatter([POST_DAYS], [mean_pred], s=50, color=MODEL, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.scatter([POST_DAYS], [mean_path[-1]], s=50, color=ACTUAL, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=2)
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)

        # Clip the y-range to the bulk of storm paths so a few extreme storms don't flatten the mean.
        low, high = np.percentile(paths, [2, 98])
        pad = 0.15 * (high - low)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(-PRE_DAYS, POST_DAYS + 3.4)
        ax.set_title(f"{sector.capitalize()} ({SECTOR_TICKERS[sector]}, n={len(df)})",
                     loc="left", fontsize=11, color=INK)
        # Endpoint labels in the right margin, nudged apart when the two values are close.
        gap = 0.07 * (high - low + 2 * pad)
        mid = (mean_path[-1] + mean_pred) / 2
        upper, lower = max(mean_path[-1], mean_pred), min(mean_path[-1], mean_pred)
        upper, lower = max(upper, mid + gap / 2), min(lower, mid - gap / 2)
        actual_y, model_y = (upper, lower) if mean_path[-1] >= mean_pred else (lower, upper)
        ax.text(POST_DAYS + 0.35, actual_y, f"actual {mean_path[-1]:+.2%}", fontsize=8,
                color=INK_SECONDARY, va="center")
        ax.text(POST_DAYS + 0.35, model_y, f"model {mean_pred:+.2%}", fontsize=8,
                color=INK_SECONDARY, va="center")
        style_axis(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    for ax in axes[1]:
        ax.set_xlabel("Trading days from hurricane event date")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative abnormal return")

    handles = [
        Line2D([], [], color=STORM_LINE, linewidth=1, label="Individual storm"),
        Line2D([], [], color=ACTUAL, linewidth=2.2, label="Average actual path"),
        Line2D([], [], color=MODEL, linewidth=2, linestyle=(0, (4, 2)), label="Average regression forecast"),
    ]
    finish(fig, "Sector abnormal returns around US Category 1+ hurricane landfalls",
           "Beta-adjusted vs. SPY, set to 0 on the event date. Shaded area = the 5 days the model predicts.",
           handles, "regression_event_paths.png")


def plot_actual_vs_predicted(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for ax, (sector, (df, _, _)) in zip(axes.flat, data.items()):
        order = df.assign(start=pd.to_datetime(df["start_date"])).sort_values("start").reset_index(drop=True)
        x = np.arange(len(order))
        actual = order[f"{sector}_5d_abnormal_return"]
        ax.vlines(x, 0, actual, color=ACTUAL, linewidth=1, alpha=0.35, zorder=1)
        ax.scatter(x, actual, s=26, color=ACTUAL, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.plot(x, order["predicted"], color=MODEL, linewidth=2, zorder=4)
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=2)

        # Label the three storms with the largest moves.
        for i in actual.abs().nlargest(3).index:
            name = order.at[i, "hurricane_name"]
            ax.annotate(name, (x[i], actual[i]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7.5, color=INK_SECONDARY, va="center")

        rmse = np.sqrt(((actual - order["predicted"]) ** 2).mean())
        hit = (np.sign(actual) == np.sign(order["predicted"])).mean()
        ax.set_title(f"{sector.capitalize()}   RMSE {rmse:.2%} · direction right {hit:.0%}",
                     loc="left", fontsize=11, color=INK)
        style_axis(ax)
        ticks = x[::8]
        ax.set_xticks(ticks, [str(order.at[i, "start"].year) for i in ticks])
        ax.set_xlim(-1, len(order) + 3)
    for ax in axes[1]:
        ax.set_xlabel("Storms in chronological order")
    for ax in axes[:, 0]:
        ax.set_ylabel("5-day abnormal return")

    handles = [
        Line2D([], [], marker="o", linestyle="", color=ACTUAL, markersize=6, label="Actual 5-day abnormal return"),
        Line2D([], [], color=MODEL, linewidth=2, label="Regression prediction (leave-one-out)"),
    ]
    finish(fig, "What the regression predicts vs. what each sector actually did",
           "Features: SPY prior 5-day return, log VIX, sector beta. Each storm is predicted by a model fit on the other storms.",
           handles, "regression_actual_vs_predicted.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    closes = load_closes()
    data = {sector: sector_frame(sector, closes) for sector in SECTOR_TICKERS}
    plot_event_paths(data)
    plot_actual_vs_predicted(data)


if __name__ == "__main__":
    main()
