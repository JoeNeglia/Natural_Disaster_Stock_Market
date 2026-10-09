
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lit


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Hurricane Market Intelligence",
    page_icon="🌀",
    layout="wide"
)

st.title("🌀 Hurricane Market Intelligence")
st.caption(
    "Explore hurricane events and their relationship "
    "with US financial market performance."
)


# ==================================================
# 2. INITIALIZE SPARK
# ==================================================

@st.cache_resource
def initialize_spark():
    return (
        SparkSession.builder
        .appName("HurricaneMarketAnalytics")
        .master("local[*]")
        .getOrCreate()
    )


spark = initialize_spark()


# ==================================================
# 3. FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

HURRICANE_PATH = BASE_DIR / "data" / "complete_dataset.csv"
MARKET_PATH = BASE_DIR / "data" / "market_raw.csv"


# ==================================================
# 4. LOAD DATA WITH SPARK
# ==================================================

@st.cache_resource
def load_data():
    hurricanes = spark.read.csv(
        str(HURRICANE_PATH),
        header=True,
        inferSchema=True
    )

    market = spark.read.csv(
        str(MARKET_PATH),
        header=True,
        inferSchema=True
    )

    return hurricanes, market


hurricanes_df, market_df = load_data()


# ==================================================
# 5. DATA PREPARATION
# ==================================================

# Dates assumed to be yyyy-MM-dd.
# Adjust to_date() if your CSV uses another format.

hurricanes_df = (
    hurricanes_df
    .withColumn(
        "event_trading_date",
        to_date(col("event_trading_date"))
    )
    .withColumn(
        "date",
        to_date(col("date"))
    )
    .withColumn(
        "duration_days",
        col("duration_days").cast("int")
    )
)

market_df = market_df.withColumn(
    "Date",
    to_date(col("Date"))
)

# Market columns for analysis
market_columns = [
    "SPY_Open", "SPY_Close",
    "XLE_Open", "XLE_Close",
    "XLF_Open", "XLF_Close",
    "XLI_Open", "XLI_Close",
    "XLU_Open", "XLU_Close",
    "^VIX_Close"
]

for column in market_columns:
    market_df = market_df.withColumn(
        column,
        col(column).cast("double")
    )


# ==================================================
# 6. SIDEBAR CONTROLS
# ==================================================

st.sidebar.title("Dashboard Controls")

# Use an event list instead of only names.
# The same hurricane name may occur in multiple years.

events_pd = (
    hurricanes_df
    .select(
        "hurricane_name",
        "date",
        "event_trading_date",
        "duration_days"
    )
    .where(
        col("hurricane_name").isNotNull()
        & col("event_trading_date").isNotNull()
        & (col("duration_days") > 0)
    )
    .orderBy("event_trading_date")
    .toPandas()
)

if events_pd.empty:
    st.error("No valid hurricane events found.")
    st.stop()

events_pd["event_label"] = (
    events_pd["hurricane_name"].astype(str)
    + " — "
    + events_pd["event_trading_date"].astype(str)
)

# Make repeated event labels distinguishable.
events_pd["event_label"] = [
    f"{label} (record {i + 1})"
    for i, label in enumerate(events_pd["event_label"])
]

selected_index = st.sidebar.selectbox(
    "Select Hurricane",
    range(len(events_pd)),
    format_func=lambda i: events_pd.iloc[i]["event_label"]
)

selected_event = events_pd.iloc[selected_index]

selected_hurricane = selected_event["hurricane_name"]
start_date = selected_event["event_trading_date"]
duration = int(selected_event["duration_days"])

# The first calendar day counts as day one.
end_date = start_date + timedelta(days=duration - 1)

st.sidebar.divider()

days_before = st.sidebar.slider(
    "Days Before Hurricane",
    min_value=0,
    max_value=30,
    value=5
)

days_after = st.sidebar.slider(
    "Days After Hurricane",
    min_value=0,
    max_value=30,
    value=10
)

analysis_start = start_date - timedelta(days=days_before)
analysis_end = end_date + timedelta(days=days_after)

st.sidebar.caption(
    "The analysis window includes calendar days. "
    "Only available trading dates appear in charts."
)


# ==================================================
# 7. SELECT HURRICANE RECORD
# ==================================================

# Select the matching event, including its original
# hurricane date and trading date.

selected_df = hurricanes_df.filter(
    (col("hurricane_name") == lit(selected_hurricane))
    & (col("event_trading_date") == lit(start_date))
    & (col("duration_days") == lit(duration))
)

if pd.notna(selected_event["date"]):
    selected_df = selected_df.filter(
        col("date") == lit(selected_event["date"])
    )

hurricane_row = selected_df.first()

if hurricane_row is None:
    st.error("Selected hurricane record not found.")
    st.stop()

hurricane = hurricane_row.asDict()


# ==================================================
# 8. FILTER MARKET DATA WITH SPARK
# ==================================================

filtered_market = (
    market_df
    .filter(
        (col("Date") >= lit(analysis_start))
        & (col("Date") <= lit(analysis_end))
    )
    .select("Date", *market_columns)
    .orderBy("Date")
)

market_pd = filtered_market.toPandas()

if market_pd.empty:
    st.warning("No market data found for this period.")
    st.stop()

market_pd["Date"] = pd.to_datetime(market_pd["Date"])
market_pd = market_pd.sort_values("Date")

# Use pandas timestamps for date comparisons.
event_start = pd.Timestamp(start_date)
event_end = pd.Timestamp(end_date)

during_market = market_pd[
    (market_pd["Date"] >= event_start)
    & (market_pd["Date"] <= event_end)
].copy()


# ==================================================
# 9. HELPER FUNCTIONS
# ==================================================

def format_number(value, decimals=2):
    # Handle missing values
    if value is None or pd.isna(value):
        return "N/A"

    # Handle non-numeric strings
    if isinstance(value, str):
        value = value.strip()

        if value.lower() in [
            "not reported",
            "unknown",
            "n/a",
            "none",
            "",
            "null"
        ]:
            return "Not Reported"

    # Convert valid numeric values
    try:
        return f"{float(value):,.{decimals}f}"

    except (ValueError, TypeError, OverflowError):
        return str(value)


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):+.2f}%"


def calculate_return(data, column):
    values = data[column].dropna()

    if len(values) < 2 or values.iloc[0] == 0:
        return None

    return (
        (values.iloc[-1] / values.iloc[0]) - 1
    ) * 100


def add_hurricane_window(fig):
    # Highlight the hurricane's calendar-day interval.
    fig.add_vrect(
        x0=event_start,
        x1=event_end + pd.Timedelta(days=1),
        fillcolor="orange",
        opacity=0.12,
        line_width=0,
        annotation_text="Hurricane Period",
        annotation_position="top left"
    )
    return fig


# ==================================================
# 10. HURRICANE OVERVIEW
# ==================================================

st.header(f"Hurricane {selected_hurricane}")

st.write(
    f"**Location:** {hurricane.get('location', 'Unknown')}"
)

st.write(
    f"**Event Trading Date:** {start_date}  |  "
    f"**End Date:** {end_date}"
)

overview1, overview2, overview3, overview4 = st.columns(4)

with overview1:
    st.metric(
        "Severity",
        str(hurricane.get("severity", "N/A"))
    )

with overview2:
    st.metric(
        "Estimated Damage",
        "$" + format_number(
            hurricane.get("damage_billions")
        ) + "B"
    )

with overview3:
    st.metric(
        "Duration",
        f"{duration} days"
    )

with overview4:
    st.metric(
        "Deaths",
        format_number(hurricane.get("deaths"), 0)
    )


# ==================================================
# 11. DASHBOARD TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Market Trends",
    "Sector Analysis",
    "Raw Data"
])


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with tab1:

    st.subheader("Market Performance During Hurricane")

    spy_return = calculate_return(
        during_market, "SPY_Close"
    )

    energy_return = calculate_return(
        during_market, "XLE_Close"
    )

    utilities_return = calculate_return(
        during_market, "XLU_Close"
    )

    financial_return = calculate_return(
        during_market, "XLF_Close"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "S&P 500 ETF",
            format_percent(spy_return)
        )

    with c2:
        st.metric(
            "Energy Sector",
            format_percent(energy_return)
        )

    with c3:
        st.metric(
            "Utilities Sector",
            format_percent(utilities_return)
        )

    with c4:
        st.metric(
            "Financial Sector",
            format_percent(financial_return)
        )

    st.divider()

    st.subheader("SPY Price Movement")

    fig = px.line(
        market_pd,
        x="Date",
        y="SPY_Close",
        markers=True,
        labels={
            "SPY_Close": "SPY Closing Price ($)",
            "Date": "Date"
        }
    )

    fig = add_hurricane_window(fig)

    fig.update_layout(
        hovermode="x unified",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The shaded region represents the hurricane "
        "period. Prices outside it provide context."
    )

    st.subheader("Hurricane Impact Statistics")

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Injuries",
            format_number(hurricane.get("injuries"), 0)
        )

    with b:
        st.metric(
            "Pre-event SPY 5D Return",
            format_percent(
                hurricane.get("spy_prev_5d_return")
            )
        )

    with c:
        st.metric(
            "Event VIX",
            format_number(hurricane.get("vix"))
        )


# ==================================================
# TAB 2: MARKET TRENDS
# ==================================================

with tab2:

    st.subheader("Market Price Analysis")

    # All available closing-price instruments.
    metric_options = {
        "S&P 500 (SPY)": "SPY_Close",
        "Energy (XLE)": "XLE_Close",
        "Financials (XLF)": "XLF_Close",
        "Industrials (XLI)": "XLI_Close",
        "Utilities (XLU)": "XLU_Close",
        "Volatility Index (VIX)": "^VIX_Close"
    }

    selected_metric = st.selectbox(
        "Select Market Metric",
        list(metric_options.keys())
    )

    selected_column = metric_options[selected_metric]

    fig = px.line(
        market_pd,
        x="Date",
        y=selected_column,
        markers=True,
        title=f"{selected_metric} Over Time"
    )

    fig = add_hurricane_window(fig)

    fig.update_layout(
        height=450,
        hovermode="x unified",
        xaxis_title="Trading Date",
        yaxis_title="Value"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Compare Sector Performance")

    # Normalize each series to 100 at its
    # first available observation.
    # This makes different ETFs comparable.

    comparison_columns = {
        "SPY": "SPY_Close",
        "Energy": "XLE_Close",
        "Financials": "XLF_Close",
        "Industrials": "XLI_Close",
        "Utilities": "XLU_Close"
    }

    comparison_df = market_pd[["Date"]].copy()

    for name, column in comparison_columns.items():
        prices = market_pd[column]
        valid = prices.dropna()

        if not valid.empty and valid.iloc[0] != 0:
            comparison_df[name] = (
                prices / valid.iloc[0]
            ) * 100

    fig = px.line(
        comparison_df,
        x="Date",
        y=[
            c for c in comparison_columns
            if c in comparison_df.columns
        ],
        title="Normalized Market Performance",
        labels={
            "value": "Indexed Price (Base = 100)",
            "variable": "Sector"
        }
    )

    fig = add_hurricane_window(fig)

    fig.update_layout(
        height=480,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each ETF starts at an index value of 100 "
        "on its first available trading day in the "
        "selected analysis window."
    )


# ==================================================
# TAB 3: SECTOR ANALYSIS
# ==================================================

with tab3:

    st.subheader("Sector Abnormal Returns")

    abnormal_returns = {
        "Energy": hurricane.get(
            "energy_5d_abnormal_return"
        ),
        "Utilities": hurricane.get(
            "utilities_5d_abnormal_return"
        ),
        "Financials": hurricane.get(
            "financials_5d_abnormal_return"
        ),
        "Industrials": hurricane.get(
            "industrials_5d_abnormal_return"
        )
    }

    abnormal_df = pd.DataFrame({
        "Sector": list(abnormal_returns.keys()),
        "Abnormal Return": list(abnormal_returns.values())
    })

    abnormal_df["Abnormal Return"] = pd.to_numeric(
        abnormal_df["Abnormal Return"],
        errors="coerce"
    )

    abnormal_df = abnormal_df.dropna()

    if not abnormal_df.empty:
        fig = px.bar(
            abnormal_df,
            x="Sector",
            y="Abnormal Return",
            color="Abnormal Return",
            color_continuous_scale="RdYlGn",
            title="5-Day Sector Abnormal Returns"
        )

        fig.update_layout(
            height=430,
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Sector Risk Indicators")

    sector_metrics = {
        "Energy": "energy",
        "Utilities": "utilities",
        "Financials": "financials",
        "Industrials": "industrials"
    }

    sector_rows = []

    for sector, prefix in sector_metrics.items():
        sector_rows.append({
            "Sector": sector,
            "Previous 5D Return": hurricane.get(
                f"{prefix}_prev_5d_return"
            ),
            "20D Volatility": hurricane.get(
                f"{prefix}_20d_volatility"
            ),
            "Volume Change": hurricane.get(
                f"{prefix}_volume_change"
            ),
            "Drawdown": hurricane.get(
                f"{prefix}_drawdown"
            ),
            "Beta": hurricane.get(
                f"{prefix}_beta"
            )
        })

    sector_df = pd.DataFrame(sector_rows)

    for column in sector_df.columns[1:]:
        sector_df[column] = pd.to_numeric(
            sector_df[column],
            errors="coerce"
        )

    st.dataframe(
        sector_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Risk indicators come directly from the "
        "hurricane dataset. Their units and calculation "
        "windows depend on how those features were built."
    )


# ==================================================
# TAB 4: RAW DATA
# ==================================================

with tab4:

    st.subheader("Hurricane Event Record")

    st.dataframe(
        selected_df.toPandas(),
        use_container_width=True
    )

    st.subheader("Filtered Market Data")

    st.dataframe(
        market_pd,
        use_container_width=True
    )

    csv = market_pd.to_csv(index=False)

    st.download_button(
        label="Download Filtered Market Data",
        data=csv,
        file_name="hurricane_market_data.csv",
        mime="text/csv"
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Hurricane Market Intelligence | "
    "PySpark + Streamlit + Plotly"
)
