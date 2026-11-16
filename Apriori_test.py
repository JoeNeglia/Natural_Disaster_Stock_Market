"""
HURRICANE ASSOCIATION RULE MINING - APRIORI
===========================================

Purpose:
    Discover relationships between hurricane characteristics,
    market conditions, and abnormal stock market returns.

Dataset:
    conceptual_complete_dataset.csv

Methods:
    1. Convert numerical data into categorical items.
    2. Find frequent itemsets using Apriori.
    3. Generate association rules.
    4. Calculate support, confidence, and lift.
    5. Visualize and export the results.

Each hurricane is treated as one transaction.

This is exploratory analysis, NOT a prediction model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT = Path(__file__).resolve().parent

DATA_FILE = PROJECT / "data" / "sector_datasets" / "conceptual_complete_dataset.csv"

OUTPUT = PROJECT / "reports" / "apriori_results"
OUTPUT.mkdir(exist_ok=True)

# Minimum fraction of hurricanes containing an itemset.
MIN_SUPPORT = 0.15

# Minimum confidence for generated rules.
MIN_CONFIDENCE = 0.60

# Rules with lift > 1 occur more often than expected
# under independence.
MIN_LIFT = 1.10

# Avoid rules with too many conditions.
MAX_ITEMSET_LENGTH = 3

SECTORS = ["energy", "utilities", "financials", "industrials"]


# ============================================================
# 2. LOAD DATA
# ============================================================


def load_data():
    """Read the hurricane dataset and inspect its structure."""

    df = pd.read_csv(DATA_FILE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("\nHURRICANE DATASET")
    print("-------------------------")
    print("Number of records:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nSeverity distribution:")
    print(df["severity"].value_counts(dropna=False))

    return df


# ============================================================
# 3. CREATE CATEGORICAL TRANSACTIONS
# ============================================================


def create_transactions(df):
    """
    Apriori requires categorical items.

    Example:
        Hurricane category = 4
        Damage = 10 billion
        Energy abnormal return = +0.025

    Becomes:
        Major_Hurricane
        High_Damage
        Energy_Positive

    Each row represents one hurricane transaction.
    """

    transactions = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # A. HURRICANE SEVERITY
    # --------------------------------------------------------

    # Extract the category number from strings such as
    # "Category 4".
    category = pd.to_numeric(
        df["severity"].astype(str).str.extract(r"Category\s*([1-5])", expand=False),
        errors="coerce",
    )

    transactions["Major_Hurricane"] = category >= 3

    transactions["Category_4_5"] = category >= 4

    transactions["Category_1_2"] = category.between(1, 2)

    # --------------------------------------------------------
    # B. HURRICANE DAMAGE
    # --------------------------------------------------------

    damage = pd.to_numeric(df["damage_billions"], errors="coerce")

    # Thresholds are illustrative and can be adjusted.
    transactions["High_Damage"] = damage >= 5
    transactions["Moderate_Damage"] = damage.between(1, 5, inclusive="left")
    transactions["Low_Damage"] = damage < 1

    # --------------------------------------------------------
    # C. HURRICANE DURATION
    # --------------------------------------------------------

    duration = pd.to_numeric(df["duration_days"], errors="coerce")

    transactions["Long_Duration"] = duration >= 10
    transactions["Short_Duration"] = duration < 10

    # --------------------------------------------------------
    # D. MARKET CONDITIONS
    # --------------------------------------------------------

    vix = pd.to_numeric(df["vix"], errors="coerce")

    # VIX measures expected stock market volatility.
    transactions["High_VIX"] = vix >= 25
    transactions["Low_VIX"] = vix < 15

    spy = pd.to_numeric(df["spy_prev_5d_return"], errors="coerce")

    transactions["Market_Previously_Up"] = spy > 0
    transactions["Market_Previously_Down"] = spy < 0

    # --------------------------------------------------------
    # E. SECTOR ABNORMAL RETURNS
    # --------------------------------------------------------

    # These columns are already calculated in the dataset.
    # Returns are decimals:
    #   0.02  = +2%
    #  -0.01  = -1%

    for sector in SECTORS:
        column = f"{sector}_5d_abnormal_return"

        returns = pd.to_numeric(df[column], errors="coerce")

        name = sector.capitalize()

        # Positive or negative abnormal return.
        transactions[f"{name}_Positive"] = returns > 0
        transactions[f"{name}_Negative"] = returns < 0

        # Larger abnormal return movements.
        transactions[f"{name}_Strong_Positive"] = returns >= 0.02
        transactions[f"{name}_Strong_Negative"] = returns <= -0.02

        # ----------------------------------------------------
        # F. SECTOR VOLATILITY
        # ----------------------------------------------------

        volatility = pd.to_numeric(df[f"{sector}_20d_volatility"], errors="coerce")

        # Relative definition of high volatility:
        # top 25% of observed values.
        threshold = volatility.quantile(0.75)

        transactions[f"{name}_High_Volatility"] = (
            volatility >= threshold
        ) & volatility.notna()

    # Missing values should NOT be interpreted as negative.
    # Drop records missing essential market outcomes.
    required = [f"{sector}_5d_abnormal_return" for sector in SECTORS]

    valid_rows = df[required].notna().all(axis=1)

    transactions = transactions.loc[valid_rows].copy()

    # Ensure Apriori receives Boolean values.
    transactions = transactions.fillna(False).astype(bool)

    print("\nTRANSACTION DATA")
    print("-------------------------")
    print("Transactions:", len(transactions))
    print("Possible items:", len(transactions.columns))

    transactions.to_csv(OUTPUT / "hurricane_transactions.csv", index=False)

    return transactions


# ============================================================
# 4. RUN APRIORI
# ============================================================


def run_apriori(transactions):
    """
    Find frequent combinations of categorical items.

    Example:
        {Major_Hurricane, Utilities_Negative}

    Support tells us what fraction of hurricanes contain
    the combination.
    """

    frequent = apriori(
        transactions,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_ITEMSET_LENGTH,
    )

    if frequent.empty:
        print("No frequent itemsets found.")
        return frequent

    frequent["length"] = frequent["itemsets"].apply(len)

    frequent = frequent.sort_values("support", ascending=False)

    # Convert itemsets into readable strings for CSV.
    export = frequent.copy()

    export["itemsets"] = export["itemsets"].apply(
        lambda items: ", ".join(sorted(items))
    )

    export.to_csv(OUTPUT / "frequent_itemsets.csv", index=False)

    print("\nFREQUENT ITEMSETS")
    print("-------------------------")
    print("Number found:", len(frequent))

    return frequent


# ============================================================
# 5. GENERATE ASSOCIATION RULES
# ============================================================


def generate_rules(frequent):
    """
    Generate rules of the form:

        A -> B

    Example:
        High_Damage -> Utilities_Negative

    Support:
        Fraction of all hurricanes containing A and B.

    Confidence:
        P(B | A)

    Lift:
        Confidence / P(B)

    Lift > 1 indicates positive association.
    """

    if frequent.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent, metric="confidence", min_threshold=MIN_CONFIDENCE
    )

    # Keep rules with meaningful lift.
    rules = rules[rules["lift"] >= MIN_LIFT].copy()

    # Keep only rules with one item on the right.
    rules = rules[rules["consequents"].apply(len) == 1].copy()

    # Sort by lift.
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)

    return rules


# ============================================================
# 6. FILTER RULES FOR MARKET REACTIONS
# ============================================================


def filter_market_rules(rules):
    """
    Focus on rules where:

        LEFT: Hurricane or market characteristics
        RIGHT: Stock market outcome

    Example:
        Major_Hurricane + High_VIX
                    ->
              Energy_Negative

    This avoids reporting trivial rules such as:
        Energy_Strong_Positive -> Energy_Positive
    """

    if rules.empty:
        return rules

    # Items representing stock market outcomes.
    outcomes = set()

    for sector in SECTORS:
        name = sector.capitalize()

        outcomes.update(
            [
                f"{name}_Positive",
                f"{name}_Negative",
                f"{name}_Strong_Positive",
                f"{name}_Strong_Negative",
            ]
        )

    # Everything else is a possible explanatory item.
    explanatory = {
        "Major_Hurricane",
        "Category_4_5",
        "Category_1_2",
        "High_Damage",
        "Moderate_Damage",
        "Low_Damage",
        "Long_Duration",
        "Short_Duration",
        "High_VIX",
        "Low_VIX",
        "Market_Previously_Up",
        "Market_Previously_Down",
    }

    for sector in SECTORS:
        explanatory.add(f"{sector.capitalize()}_High_Volatility")

    mask = rules.apply(
        lambda row: (
            set(row["antecedents"]).issubset(explanatory)
            and set(row["consequents"]).issubset(outcomes)
        ),
        axis=1,
    )

    filtered = rules.loc[mask].copy()

    return filtered


# ============================================================
# 7. EXPORT AND PRINT RULES
# ============================================================


def save_rules(rules, filename, total_events):
    """Convert rules into a readable table."""

    if rules.empty:
        print(f"\nNo rules found for {filename}")
        return pd.DataFrame()

    output = pd.DataFrame(
        {
            "antecedent": rules["antecedents"].apply(lambda x: " + ".join(sorted(x))),
            "consequent": rules["consequents"].apply(lambda x: " + ".join(sorted(x))),
            "support": rules["support"],
            "confidence": rules["confidence"],
            "lift": rules["lift"],
        }
    )

    # Actual number of events supporting each rule.
    output["event_count"] = (output["support"] * total_events).round().astype(int)

    output.to_csv(OUTPUT / filename, index=False)

    print(f"\nTOP RULES: {filename}")
    print("------------------------------------------")

    for _, row in output.head(15).iterrows():
        print(f"\n{row['antecedent']} -> {row['consequent']}")

        print(f"  Support:    {row['support']:.3f}")
        print(f"  Confidence: {row['confidence']:.3f}")
        print(f"  Lift:       {row['lift']:.3f}")
        print(f"  Events:     {row['event_count']}")

    return output


# ============================================================
# 8. VISUALIZE SUPPORT, CONFIDENCE, AND LIFT
# ============================================================


def plot_rules(rules):
    """
    Scatterplot of discovered rules.

    X-axis: Support
    Y-axis: Confidence
    Color: Lift

    Higher lift suggests a stronger positive association
    relative to the baseline frequency.
    """

    if rules.empty:
        return

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        s=90,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
    )

    plt.colorbar(scatter, label="Lift")

    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Hurricane Association Rules")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT / "support_confidence_lift.png", dpi=200)

    plt.show()


# ============================================================
# 9. VISUALIZE TOP ASSOCIATION RULES
# ============================================================


def plot_top_rules(rules):
    """Bar chart showing the strongest rules by lift."""

    if rules.empty:
        return

    top = rules.head(12).copy()

    labels = top["antecedent"] + " -> " + top["consequent"]

    plt.figure(figsize=(12, 7))

    plt.barh(labels.iloc[::-1], top["lift"].iloc[::-1])

    plt.axvline(1, color="gray", linestyle="--", label="No association (lift = 1)")

    plt.xlabel("Lift")
    plt.title("Top Hurricane-Market Association Rules")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT / "top_association_rules.png", dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# 10. MAIN PROGRAM
# ============================================================


def main():

    print("Starting Hurricane Apriori Analysis")

    # Load hurricane records.
    df = load_data()

    # Convert numerical data to categorical transactions.
    transactions = create_transactions(df)

    # Discover frequent item combinations.
    frequent = run_apriori(transactions)

    # Generate association rules.
    rules = generate_rules(frequent)

    # Save all discovered rules.
    save_rules(rules, "all_association_rules.csv", len(transactions))

    # Focus on hurricane/market characteristics
    # associated with stock market outcomes.
    market_rules = filter_market_rules(rules)

    readable_rules = save_rules(
        market_rules, "hurricane_market_rules.csv", len(transactions)
    )

    # Generate visualizations.
    plot_rules(market_rules)
    plot_top_rules(readable_rules)

    print("\nAnalysis complete!")
    print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
