"""Plot the decision tree, random forest and SVM classification results.

Usage from the repository root:
    python code/visualization/plot_classification_results.py

Inputs in reports/:
    decision_tree_results.csv, random_forest_results.csv, svm_results.csv
Outputs in reports/figures/:
    classification_accuracy.png           baseline vs hurricane accuracy, one panel per model
    classification_balanced_accuracy.png  same layout for balanced accuracy
    classification_hurricane_gain.png     hurricane minus baseline accuracy, model x sector
"""

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS = {
    "decision_tree_results.csv": "Decision tree",
    "random_forest_results.csv": "Random forest",
    "svm_results.csv": "SVM",
}
SECTOR_ORDER = ["energy", "financials", "industrials", "utilities"]

# Colors (validated light-mode palette: categorical slots 1-2 plus chart chrome).
BASELINE = "#2a78d6"
HURRICANE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("gain", ["#e34948", "#f0efec", "#2a78d6"])

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


def load_results():
    frames = []
    for filename, label in MODELS.items():
        path = REPORTS_DIR / filename
        if not path.exists():
            print(f"Skipping missing {path}")
            continue
        df = pd.read_csv(path)
        df["model_type"] = label
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No classification results found in reports/. Run the model scripts first.")
    return pd.concat(frames, ignore_index=True)


def plot_metric(results, metric, title, filename):
    """Dumbbell chart: baseline -> hurricane score per sector, one panel per model."""
    model_types = [m for m in MODELS.values() if m in set(results["model_type"])]
    fig, axes = plt.subplots(1, len(model_types), figsize=(4 * len(model_types), 3.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    values = results[metric]
    low = min(values.min(), results["majority_class_accuracy"].min(), 0.5) - 0.03
    high = max(values.max(), 0.5) + 0.03
    y = np.arange(len(SECTOR_ORDER))

    for ax, model_type in zip(axes, model_types):
        data = results[results["model_type"] == model_type].set_index(["sector", "model"])
        for i, sector in enumerate(SECTOR_ORDER):
            base = data.loc[(sector, "baseline"), metric]
            hurr = data.loc[(sector, "hurricane"), metric]
            ax.plot([base, hurr], [i, i], color=AXIS, linewidth=2, zorder=1)
            ax.scatter(base, i, s=70, color=BASELINE, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.scatter(hurr, i, s=70, color=HURRICANE, edgecolor=SURFACE, linewidth=2, zorder=3)
            if metric == "accuracy":
                majority = data.loc[(sector, "baseline"), "majority_class_accuracy"]
                ax.scatter(majority, i, marker="|", s=220, color=MUTED, linewidth=2, zorder=2)
            # Change from adding hurricane features, in a column at the panel's right edge.
            ax.text(1.0, i, f"{hurr - base:+.3f}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=9, color=INK_SECONDARY)

        ax.axvline(0.5, color=MUTED, linestyle=(0, (3, 3)), linewidth=1, zorder=0)
        ax.set_title(model_type, fontsize=11, color=INK, loc="left")
        ax.set_xlim(low, high)
        ax.set_ylim(len(SECTOR_ORDER) - 0.5, -0.6)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.text(1.0, -0.6, "change", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=8, color=MUTED)
        ax.set_yticks(y, [s.capitalize() for s in SECTOR_ORDER])
        ax.tick_params(axis="y", colors=INK_SECONDARY, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.set_xlabel(metric.replace("_", " ").capitalize())

    handles = [
        Line2D([], [], marker="o", linestyle="", color=BASELINE, markersize=8, label="Baseline (sector data)"),
        Line2D([], [], marker="o", linestyle="", color=HURRICANE, markersize=8, label="+ Hurricane features"),
    ]
    if metric == "accuracy":
        handles.append(Line2D([], [], marker="|", linestyle="", color=MUTED, markersize=12,
                              markeredgewidth=2, label="Always guess common direction"))
    handles.append(Line2D([], [], color=MUTED, linestyle=(0, (3, 3)), label="Chance (0.5)"))

    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=4)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_gain_heatmap(results, filename):
    """Hurricane minus baseline accuracy for every model and sector."""
    pivot = results.pivot_table(index="model_type", columns=["sector", "model"], values="accuracy")
    gain = pd.DataFrame({
        sector: pivot[(sector, "hurricane")] - pivot[(sector, "baseline")] for sector in SECTOR_ORDER
    })
    gain = gain.reindex([m for m in MODELS.values() if m in gain.index])

    limit = max(abs(gain.values).max(), 0.01)
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.75 * len(gain)))
    image = ax.imshow(gain.values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")

    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            ax.text(j, i, f"{gain.iat[i, j]:+.3f}", ha="center", va="center", fontsize=10, color=INK)

    ax.set_xticks(range(len(SECTOR_ORDER)), [s.capitalize() for s in SECTOR_ORDER])
    ax.set_yticks(range(len(gain)), gain.index)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.set_xticks(np.arange(-0.5, len(SECTOR_ORDER)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(gain)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    colorbar.set_label("Accuracy change", color=INK_SECONDARY, fontsize=9)

    ax.set_title("Accuracy change from adding hurricane features\n"
                 "blue = hurricane features helped, red = they hurt",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    plot_metric(results, "accuracy", "Test accuracy: baseline vs. with hurricane features",
                "classification_accuracy.png")
    plot_metric(results, "balanced_accuracy", "Test balanced accuracy: baseline vs. with hurricane features",
                "classification_balanced_accuracy.png")
    plot_gain_heatmap(results, "classification_hurricane_gain.png")


if __name__ == "__main__":
    main()
