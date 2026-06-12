# Streamlit dashboard for the dissertation forecast.
# Entry point for Streamlit Cloud: `streamlit run app.py`

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Supplier Payment Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
DATA_PATH = Path("data/forecast_panel.parquet")
MODEL_PATH = Path("models/xgboost_model.joblib")
FEATURES_PATH = Path("models/feature_cols.json")
METRICS_PATH = Path("models/validation_metrics.json")

# Forecast configuration
LATEST_DATA_YEAR = 2025
FORECAST_YEAR = 2026
SECTOR_SNAPSHOT_YEAR = 2024

# Sectors with too few firm-years to plot meaningfully on the sector dashboard.
# (Activities of households has only 10 firm-years across the whole panel.)
THIN_SECTORS = ["Activities of households as employers"]


@st.cache_data
def load_panel():
    return pd.read_parquet(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_cols():
    with open(FEATURES_PATH) as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


# Sidebar navigation
st.sidebar.title("Supplier Payment Forecast")
st.sidebar.caption("MSc Dissertation · UCL · 2026")
st.sidebar.divider()

view = st.sidebar.radio(
    "Navigate",
    ["Overview", "Firm explorer", "Sector dashboard", "Model validation"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "Industry collaboration: EY-Parthenon. "
    "Data: UK Prompt Payment Reporting (2017-2025) and "
    "Bureau van Dijk FAME."
)


# Main panel — route to the selected view
if view == "Overview":
    st.title("Supplier Payment Behaviour Forecast")
    st.caption(
        "Companion dashboard to the MSc dissertation "
        "*Does Capital Intensity Predict Supplier Payment Behaviour? "
        "Evidence from UK Prompt Payment Reporting and Firm Financials*"
    )

    st.markdown("""
    ### What this dashboard is

    This dashboard forecasts how long UK firms take to pay their suppliers,
    based on firm-level financial characteristics and recent payment history.
    It accompanies a dissertation examining whether capital intensity — the
    ratio of fixed assets to turnover — predicts supplier payment behaviour.

    The dissertation argues that capital-intensive firms pay suppliers more
    slowly because they have less flexible working capital and more bargaining
    power. This dashboard tests that claim by deploying a forecasting model
    and reporting its accuracy honestly.

    ### What it contains

    - **Firm explorer** — pick any of 1,390 UK firms and see its historical
      payment behaviour. For firms with 2025 data, a forecast for 2026 is
      provided using the XGBoost model.
    - **Sector dashboard** — aggregate patterns across 17 broad sectors,
      including the dissertation's main associational findings.
    - **Model validation** — honest reporting on forecast accuracy.
      An XGBoost model is the production forecast; lag-1 and panel
      autoregression baselines are reported for comparison.

    ### The data

    Two sources, both UK official:

    - **Prompt Payment Reporting (PPR)** — mandatory disclosures filed by
      large UK firms under the 2017 Reporting on Payment Practices and
      Performance Regulations. Reports include the average time to pay
      suppliers and the percentage of invoices paid within 30 days.
    - **Bureau van Dijk FAME** — firm-level financial statement data,
      providing capital intensity, leverage, working capital, and other
      controls.

    The analytical panel is 8,370 firm-years across 1,390 firms over
    2017-2025.

    ### Caveats

    - The dissertation's analytical claims are *associational*, not causal.
      The dashboard's forecasts are predictive, not interpretive.
    - The XGBoost model beats the lag-1 baseline by 3.4% RMSE — a real
      but modest improvement. Year-to-year payment behaviour at the
      individual firm level is genuinely hard to predict.
    - The 2026 forecast is only available for firms that filed PPR data
      in 2025. Firms with older data show historical observations only.
    - The forecast assumes the relationship between firm characteristics
      and payment behaviour observed during 2017-2023 continues into 2026.
    """)


elif view == "Firm explorer":
    st.title("Firm explorer")
    st.caption(f"Select a firm to see its payment history and (if eligible) a {FORECAST_YEAR} forecast.")

    panel = load_panel()
    model = load_model()
    feature_cols = load_feature_cols()

    firm_options = (
        panel[["crn_clean", "Company name", "sector_name"]]
        .drop_duplicates()
        .sort_values("Company name")
    )
    firm_options["label"] = firm_options["Company name"] + " (" + firm_options["crn_clean"] + ")"

    selected_label = st.selectbox(
        "Search for a firm",
        options=firm_options["label"].tolist(),
        index=0,
    )

    selected_crn = firm_options.loc[firm_options["label"] == selected_label, "crn_clean"].iloc[0]
    firm_data = panel[panel["crn_clean"] == selected_crn].sort_values("year")

    company_name = firm_data["Company name"].iloc[0]
    sector = firm_data["sector_name"].iloc[0]
    years_observed = len(firm_data)
    latest_year = int(firm_data["year"].max())
    latest_payment_time = firm_data[firm_data["year"] == latest_year]["Average time to pay"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sector", sector)
    col2.metric("Years observed", years_observed)
    col3.metric("Latest year", latest_year)
    col4.metric(f"Avg time to pay ({latest_year})", f"{latest_payment_time:.1f} days")

    st.divider()

    st.subheader(f"Payment history and {FORECAST_YEAR} forecast")

    has_latest_data = latest_year == LATEST_DATA_YEAR
    latest_row = firm_data[firm_data["year"] == latest_year].iloc[0]
    features_complete = not any(pd.isna(latest_row[col]) for col in feature_cols)
    can_forecast = has_latest_data and features_complete

    history_chart = firm_data[["year", "Average time to pay"]].copy()
    history_chart = history_chart.rename(columns={"Average time to pay": "Days"})
    history_chart["Type"] = "Actual"

    if can_forecast:
        forecast_features = latest_row[feature_cols].copy()
        forecast_features["avg_time_to_pay_lag_1"] = latest_row["Average time to pay"]

        forecast_value = float(model.predict(forecast_features.values.reshape(1, -1))[0])

        forecast_row = pd.DataFrame({
            "year": [FORECAST_YEAR],
            "Days": [forecast_value],
            "Type": ["Forecast"],
        })
        history_chart = pd.concat([history_chart, forecast_row], ignore_index=True)

        st.caption(
            f"The {FORECAST_YEAR} forecast uses XGBoost trained on 2017-2023 "
            f"and validated on 2024 (RMSE 5.64 days, MAE 3.79 days vs lag-1 "
            f"baseline 5.84 / 3.49). The forecast assumes the firm's 2025 "
            f"financial characteristics persist into 2026."
        )
    elif not has_latest_data:
        st.info(
            f"**No {FORECAST_YEAR} forecast available.** This firm's most recent "
            f"PPR filing is {latest_year}. Forecasts require 2025 data because "
            f"the model uses the most recent year as input. Historical payment "
            f"behaviour is shown below."
        )
    else:
        st.info(
            f"**No {FORECAST_YEAR} forecast available.** This firm has 2025 PPR "
            f"data but is missing some financial features required by the model. "
            f"Historical payment behaviour is shown below."
        )

    chart = (
        alt.Chart(history_chart)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("Days:Q", title="Average time to pay (days)"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Actual", "Forecast"],
                    range=["#3a5a7c", "#c45a3b"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["year:O", "Days:Q", "Type:N"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    st.divider()

    st.subheader(f"How {company_name} compares to its sector")

    sector_peers = (
        panel[(panel["sector_name"] == sector) & (panel["year"] == latest_year)]
        .copy()
        .sort_values("Average time to pay")
    )
    sector_peers["is_focal"] = sector_peers["crn_clean"] == selected_crn

    n_peers = len(sector_peers)
    median_peer_value = sector_peers["Average time to pay"].median()

    rank_in_sector = (sector_peers["Average time to pay"].rank(method="min").loc[
        sector_peers["crn_clean"] == selected_crn
    ].iloc[0])

    st.write(
        f"In {latest_year}, this firm paid suppliers in {latest_payment_time:.1f} days, "
        f"ranking {int(rank_in_sector)} of {n_peers} firms in the {sector} sector "
        f"(median peer: {median_peer_value:.1f} days)."
    )

    sector_chart = (
        alt.Chart(sector_peers)
        .mark_bar()
        .encode(
            x=alt.X("Average time to pay:Q", title="Average time to pay (days)"),
            y=alt.Y("Company name:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.is_focal,
                alt.value("#c45a3b"),
                alt.value("#3a5a7c"),
            ),
            tooltip=["Company name:N", "Average time to pay:Q"],
        )
        .properties(height=max(300, 18 * n_peers))
    )
    st.altair_chart(sector_chart, use_container_width=True)

    st.divider()

    st.subheader("Firm financials")

    display_features = {
        "Capital intensity": "capital_intensity",
        "Log total assets": "log_total_assets",
        "Profit margin": "profit_margin",
        "Leverage": "leverage",
        "Net working capital": "net_working_capital",
        "Debtors to turnover": "debtors_to_turnover",
    }

    fin_table = pd.DataFrame({
        "Year": firm_data["year"].astype(int).values,
    })
    for label, col in display_features.items():
        fin_table[label] = firm_data[col].values

    fin_table = fin_table.set_index("Year")
    st.dataframe(fin_table.style.format("{:.3f}"), use_container_width=True)


elif view == "Sector dashboard":
    st.title("Sector dashboard")
    st.caption(
        f"Aggregate patterns across the 16 broad ONS sectors with sufficient "
        f"data. Snapshot year is {SECTOR_SNAPSHOT_YEAR} (the most complete "
        f"recent year; 2025 data is partial). "
        f"\"Activities of households as employers\" is excluded — only 10 "
        f"firm-years in the entire panel, too thin for sector-level analysis."
    )

    panel = load_panel()

    # Filter out thin sectors and 2017 (partial first reporting year)
    panel_clean = panel[~panel["sector_name"].isin(THIN_SECTORS)].copy()
    snapshot = panel_clean[panel_clean["year"] == SECTOR_SNAPSHOT_YEAR].copy()
    n_firms_snapshot = len(snapshot)

    st.markdown(f"""
    The snapshot below covers **{n_firms_snapshot:,} firm-years** across
    **16 sectors**. The dissertation finds an associational link between
    capital intensity and payment time — capital-intensive firms tend to pay
    suppliers more slowly. This view shows three angles on that pattern at
    the sector level.
    """)

    st.divider()

    # Chart 1: Distribution of payment time by sector (box plot)
    st.subheader("Distribution of payment time by sector")

    st.markdown(f"""
    Each box shows the spread of average payment time for firms in that sector
    in {SECTOR_SNAPSHOT_YEAR}. The line inside each box is the median;
    the box covers the 25th–75th percentile (the middle half of firms); the
    whiskers extend to the typical range.
    """)

    sector_order = (
        snapshot.groupby("sector_name")["Average time to pay"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    box_chart = (
        alt.Chart(snapshot)
        .mark_boxplot(extent="min-max", color="#3a5a7c")
        .encode(
            x=alt.X(
                "Average time to pay:Q",
                title="Average time to pay (days)",
                scale=alt.Scale(domain=[0, snapshot["Average time to pay"].quantile(0.99)]),
            ),
            y=alt.Y(
                "sector_name:N",
                sort=sector_order,
                title=None,
            ),
        )
        .properties(height=450)
    )
    st.altair_chart(box_chart, use_container_width=True)

    st.divider()

    # Chart 2: Sector mean payment time over time
    st.subheader("Sector mean payment time over time")

    st.markdown("""
    How have sectors moved together or apart since mandatory Prompt Payment
    Reporting began? The chart below shows the mean payment time per sector
    per year, 2018-2025. (2017 is excluded — PPR only became mandatory
    partway through that year, so only 11 firms in the panel filed, and the
    resulting sector means are too noisy to compare against later years.)
    Hover over a line to see the sector. Payment times are relatively sticky
    across the COVID period — there's no broad spike in 2020-2021.
    """)

    sector_yearly = (
        panel_clean[panel_clean["year"] >= 2018]
        .groupby(["year", "sector_name"])["Average time to pay"]
        .mean()
        .reset_index()
    )

    line_chart = (
        alt.Chart(sector_yearly)
        .mark_line(point=False, strokeWidth=1.5)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(
                "Average time to pay:Q",
                title="Mean average time to pay (days)",
            ),
            color=alt.Color(
                "sector_name:N",
                title="Sector",
                legend=alt.Legend(orient="bottom", columns=3),
            ),
            tooltip=["year:O", "sector_name:N", "Average time to pay:Q"],
        )
        .properties(height=400)
    )
    st.altair_chart(line_chart, use_container_width=True)

    st.divider()

    # Chart 3: Capital intensity vs payment time at sector level
    st.subheader("Capital intensity vs payment time (sector means)")

    st.markdown(f"""
    The dissertation's main associational finding: at the sector level,
    capital intensity is associated with longer payment times. Each dot below
    is one sector's mean capital intensity and mean payment time in
    {SECTOR_SNAPSHOT_YEAR}. Hover over any dot to see which sector. The
    pattern is consistent with the dissertation's "Reason 3" channel — sector
    and product structure shape working-capital cycles and supplier-payment
    norms.
    """)

    sector_means = (
        snapshot.groupby("sector_name")
        .agg(
            mean_capital_intensity=("capital_intensity", "mean"),
            mean_payment_time=("Average time to pay", "mean"),
            n_firms=("crn_clean", "nunique"),
        )
        .reset_index()
        .sort_values("mean_capital_intensity")
    )

    scatter_chart = (
        alt.Chart(sector_means)
        .mark_circle(size=300, color="#3a5a7c", opacity=0.75, stroke="#1a1a2e", strokeWidth=1)
        .encode(
            x=alt.X(
                "mean_capital_intensity:Q",
                title="Mean capital intensity (sector)",
            ),
            y=alt.Y(
                "mean_payment_time:Q",
                title="Mean average time to pay (days, sector)",
            ),
            tooltip=[
                alt.Tooltip("sector_name:N", title="Sector"),
                alt.Tooltip("mean_capital_intensity:Q", title="Capital intensity", format=".3f"),
                alt.Tooltip("mean_payment_time:Q", title="Payment time (days)", format=".1f"),
                alt.Tooltip("n_firms:Q", title="Number of firms"),
            ],
        )
        .properties(height=400)
    )

    st.altair_chart(scatter_chart, use_container_width=True)

    st.caption(
        "Note: this is an aggregate sector-level association, not a "
        "firm-level finding. The dissertation's regression analysis (in the "
        "methodology chapter) examines the firm-within-sector relationship, "
        "which is what the associational claim rests on."
    )

    # Reference table: shows the same data with sector names visible
    st.markdown("**Sector means in detail**")
    sector_table = sector_means.rename(columns={
        "sector_name": "Sector",
        "mean_capital_intensity": "Capital intensity",
        "mean_payment_time": "Payment time (days)",
        "n_firms": "Number of firms",
    }).sort_values("Capital intensity", ascending=False)

    st.dataframe(
        sector_table.set_index("Sector").style.format({
            "Capital intensity": "{:.3f}",
            "Payment time (days)": "{:.1f}",
            "Number of firms": "{:,}",
        }),
        use_container_width=True,
    )


elif view == "Model validation":
    st.title("Model validation")
    st.write("Placeholder for the model validation view.")