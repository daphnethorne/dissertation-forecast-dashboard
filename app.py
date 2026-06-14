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
ANALYTICAL_PATH = Path("data/analytical_panel.parquet")
REGRESSOR_PATH = Path("models/xgboost_model.joblib")
CLASSIFIER_PATH = Path("models/xgboost_classifier.joblib")
FEATURES_PATH = Path("models/feature_cols.json")
METRICS_PATH = Path("models/validation_metrics.json")

# Forecast configuration
LATEST_DATA_YEAR = 2025
FORECAST_YEARS = [2026, 2027, 2028]
SECTOR_SNAPSHOT_YEAR = 2024
LATE_THRESHOLD = 50.0

THIN_SECTORS = ["Activities of households as employers"]


@st.cache_data
def load_panel():
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def load_analytical():
    return pd.read_parquet(ANALYTICAL_PATH)

@st.cache_resource
def load_regressor():
    return joblib.load(REGRESSOR_PATH)

@st.cache_resource
def load_classifier():
    return joblib.load(CLASSIFIER_PATH)

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


if view == "Overview":
    st.title("Supplier Payment Behaviour Forecast")
    st.caption(
        "Companion dashboard to the MSc dissertation "
        "*Does Capital Intensity Predict Supplier Payment Behaviour? "
        "Evidence from UK Prompt Payment Reporting and Firm Financials*"
    )

    st.markdown(f"""
    ### What this dashboard is

    This dashboard forecasts how long UK firms take to pay their suppliers
    and the likelihood they will pay late, based on firm-level financial
    characteristics and recent payment history. It accompanies a dissertation
    examining whether capital intensity — the ratio of fixed assets to
    turnover — predicts supplier payment behaviour.

    ### What it contains

    - **Firm explorer** — pick any of 1,390 UK firms and see its historical
      payment behaviour. For firms with 2025 data, two forecasts are produced
      for {FORECAST_YEARS[0]}, {FORECAST_YEARS[1]}, and {FORECAST_YEARS[2]}:
      the expected payment time in days, and the probability of being late
      (paying more than 50% of invoices outside agreed terms).
    - **Sector dashboard** — aggregate patterns across 17 broad sectors.
    - **Model validation** — honest reporting on forecast accuracy.

    ### The forecasts

    Two models are deployed:

    - **Days regressor (XGBoost).** Predicts a firm's average time to pay
      suppliers in days. Validation RMSE 5.64 days, MAE 3.79.
    - **Late-payment classifier (XGBoost).** Predicts probability that the
      firm will pay more than 50% of invoices outside its agreed terms.
      Validation ROC AUC 0.78.

    ### The data

    Two sources, both UK official:

    - **Prompt Payment Reporting (PPR)** — mandatory disclosures filed by
      large UK firms under the 2017 Reporting on Payment Practices and
      Performance Regulations.
    - **Bureau van Dijk FAME** — firm-level financial statement data.

    The analytical panel is 8,370 firm-years across 1,390 firms over
    2017-2025.

    ### Caveats

    - The dissertation's analytical claims are *associational*, not causal.
    - Multi-year forecasts compound model error. The {FORECAST_YEARS[2]}
      forecast is less certain than the {FORECAST_YEARS[0]} forecast.
    - The forecasts assume the firm's 2025 financial characteristics persist.
    - Forecasts are only available for firms that filed PPR data in 2025.
    """)


elif view == "Firm explorer":
    st.title("Firm explorer")
    st.caption(
        f"Select a firm to see its payment history. For firms with {LATEST_DATA_YEAR} "
        f"data, three-year forecasts are produced for {FORECAST_YEARS[0]}-{FORECAST_YEARS[2]}."
    )

    panel = load_panel()
    analytical = load_analytical()
    regressor = load_regressor()
    classifier = load_classifier()
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
    firm_analytical = analytical[analytical["crn_clean"] == selected_crn].sort_values("year")

    company_name = firm_data["Company name"].iloc[0]
    sector = firm_data["sector_name"].iloc[0]
    years_observed = len(firm_data)
    latest_year = int(firm_data["year"].max())
    latest_payment_time = firm_data[firm_data["year"] == latest_year]["Average time to pay"].iloc[0]

    # Get the firm's stated agreed terms from the analytical panel
    latest_analytical = firm_analytical[firm_analytical["year"] == latest_year]
    if len(latest_analytical) > 0 and not pd.isna(latest_analytical["Shortest (or only) standard payment period"].iloc[0]):
        agreed_terms = float(latest_analytical["Shortest (or only) standard payment period"].iloc[0])
    else:
        agreed_terms = None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sector", sector)
    col2.metric("Years observed", years_observed)
    col3.metric("Latest year", latest_year)
    col4.metric(f"Avg time to pay ({latest_year})", f"{latest_payment_time:.1f} days")

    st.divider()

    st.subheader(f"Payment history and {FORECAST_YEARS[0]}-{FORECAST_YEARS[2]} forecast")

    has_latest_data = latest_year == LATEST_DATA_YEAR
    latest_row = firm_data[firm_data["year"] == latest_year].iloc[0]
    features_complete = not any(pd.isna(latest_row[col]) for col in feature_cols)
    can_forecast = has_latest_data and features_complete

    history_chart = firm_data[["year", "Average time to pay"]].copy()
    history_chart = history_chart.rename(columns={"Average time to pay": "Days"})
    history_chart["Type"] = "Actual"
    history_chart["Opacity"] = 1.0

    forecast_results = []

    if can_forecast:
        # Recursive multi-year forecasting.
        # Each year's prediction becomes the next year's lag-1.
        # All other features held constant at their 2025 values.
        current_features = latest_row[feature_cols].copy()
        current_lag1 = latest_row["Average time to pay"]

        for i, forecast_year in enumerate(FORECAST_YEARS):
            current_features["avg_time_to_pay_lag_1"] = current_lag1

            # Days forecast
            days_forecast = float(regressor.predict(current_features.values.reshape(1, -1))[0])

            # Probability of being late
            prob_late = float(classifier.predict_proba(current_features.values.reshape(1, -1))[0, 1])

            forecast_results.append({
                "year": forecast_year,
                "days": days_forecast,
                "prob_late": prob_late,
            })

            # The next year's lag-1 is this year's prediction.
            current_lag1 = days_forecast

        # Add forecast points to the history chart with fading opacity
        opacities = [1.0, 0.65, 0.4]
        for i, result in enumerate(forecast_results):
            history_chart = pd.concat([history_chart, pd.DataFrame({
                "year": [result["year"]],
                "Days": [result["days"]],
                "Type": ["Forecast"],
                "Opacity": [opacities[i]],
            })], ignore_index=True)

        st.caption(
            f"Forecasts use XGBoost trained on 2017-2023 and validated on 2024 "
            f"(RMSE 5.64 days, MAE 3.79 days). Multi-year forecasts are produced "
            f"recursively — each year's prediction becomes the next year's input. "
            f"Forecast points fade as the horizon extends to reflect compounding uncertainty."
        )
    elif not has_latest_data:
        st.info(
            f"**No forecast available.** This firm's most recent PPR filing is "
            f"{latest_year}. Forecasts require {LATEST_DATA_YEAR} data because "
            f"the model uses the most recent year as input."
        )
    else:
        st.info(
            f"**No forecast available.** This firm has {LATEST_DATA_YEAR} PPR "
            f"data but is missing some financial features required by the model."
        )

    # Build the chart with opacity encoding
    actual_data = history_chart[history_chart["Type"] == "Actual"]
    forecast_data = history_chart[history_chart["Type"] == "Forecast"]

    actual_chart = (
        alt.Chart(actual_data)
        .mark_line(point=True, strokeWidth=2, color="#3a5a7c")
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("Days:Q", title="Average time to pay (days)"),
            tooltip=["year:O", "Days:Q", "Type:N"],
        )
    )

    if len(forecast_data) > 0:
        forecast_points = (
            alt.Chart(forecast_data)
            .mark_circle(size=120, color="#c45a3b")
            .encode(
                x=alt.X("year:O"),
                y=alt.Y("Days:Q"),
                opacity=alt.Opacity("Opacity:Q", legend=None, scale=alt.Scale(domain=[0, 1])),
                tooltip=["year:O", "Days:Q", "Type:N"],
            )
        )

        # Dashed connecting line from last actual to forecasts
        last_actual = actual_data.iloc[-1:].copy()
        connector_data = pd.concat([last_actual, forecast_data], ignore_index=True)
        connector = (
            alt.Chart(connector_data)
            .mark_line(strokeDash=[5, 5], strokeWidth=1.5, color="#c45a3b", opacity=0.5)
            .encode(
                x=alt.X("year:O"),
                y=alt.Y("Days:Q"),
            )
        )

        combined = (actual_chart + connector + forecast_points).properties(height=350)
    else:
        combined = actual_chart.properties(height=350)

    st.altair_chart(combined, use_container_width=True)

    # Late payment risk section
    if can_forecast and forecast_results:
        st.divider()
        st.subheader("Late payment risk")

        risk_intro = (
            f"Probability of paying more than 50% of invoices outside agreed terms "
            f"in each year, predicted by the XGBoost classifier (ROC AUC 0.78 on the "
            f"2024 hold-out set)."
        )
        st.write(risk_intro)

        risk_cols = st.columns(3)
        for i, result in enumerate(forecast_results):
            with risk_cols[i]:
                prob_pct = 100 * result["prob_late"]
                if prob_pct >= 50:
                    risk_label = "High"
                elif prob_pct >= 25:
                    risk_label = "Moderate"
                else:
                    risk_label = "Low"
                st.metric(
                    f"{result['year']}",
                    f"{prob_pct:.1f}%",
                    delta=risk_label,
                    delta_color="off",
                )

        st.caption(
            f"\"Late\" here means paying more than 50% of invoices outside the firm's "
            f"own agreed terms. Risk thresholds (low/moderate/high) are dashboard "
            f"conventions only — the underlying probability is what the classifier reports."
        )

    # Days forecast vs agreed terms
    if can_forecast and forecast_results and agreed_terms is not None:
        st.divider()
        st.subheader("Forecast days vs agreed terms")

        st.write(
            f"This firm's stated agreed payment terms (from PPR) are "
            f"**{agreed_terms:.0f} days**. The expected gap between forecast "
            f"days and agreed terms is shown below."
        )

        gap_data = []
        for result in forecast_results:
            gap = result["days"] - agreed_terms
            gap_data.append({
                "Year": result["year"],
                "Forecast (days)": f"{result['days']:.1f}",
                "Agreed terms (days)": f"{agreed_terms:.0f}",
                "Gap": f"{gap:+.1f} days {'over' if gap > 0 else 'within'} terms",
            })

        gap_df = pd.DataFrame(gap_data).set_index("Year")
        st.dataframe(gap_df, use_container_width=True)

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
            y=alt.Y("sector_name:N", sort=sector_order, title=None),
        )
        .properties(height=450)
    )
    st.altair_chart(box_chart, use_container_width=True)

    st.divider()

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
            y=alt.Y("Average time to pay:Q", title="Mean average time to pay (days)"),
            color=alt.Color("sector_name:N", title="Sector", legend=alt.Legend(orient="bottom", columns=3)),
            tooltip=["year:O", "sector_name:N", "Average time to pay:Q"],
        )
        .properties(height=400)
    )
    st.altair_chart(line_chart, use_container_width=True)

    st.divider()

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
            x=alt.X("mean_capital_intensity:Q", title="Mean capital intensity (sector)"),
            y=alt.Y("mean_payment_time:Q", title="Mean average time to pay (days, sector)"),
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
    st.caption(
        "Honest reporting on both forecast models. Two models are deployed: "
        "a days regressor (XGBoost) and a late-payment classifier (XGBoost)."
    )

    metrics = load_metrics()

    st.divider()

    # Regressor section
    st.subheader("Days forecast (regressor)")

    st.markdown(f"""
    Forecasting payment behaviour at the firm-year level is genuinely hard.
    Three approaches were tested on the 2024 hold-out set
    (**{metrics['n_validation']:,} firm-years** — the full 2024 sample is
    1,106 firms; the validation set drops 79 rows with missing lag-1 or
    feature values), and the headline finding is that all three perform
    within a few days of each other.
    """)

    comparison_df = pd.DataFrame({
        "Model": ["XGBoost (production)", "Lag-1 baseline", "Panel autoregression"],
        "RMSE (days)": [
            metrics["xgboost"]["rmse"],
            metrics["baseline_lag1"]["rmse"],
            metrics["panel_ar"]["rmse"],
        ],
        "MAE (days)": [
            metrics["xgboost"]["mae"],
            metrics["baseline_lag1"]["mae"],
            metrics["panel_ar"]["mae"],
        ],
        "Description": [
            "Gradient-boosted trees with firm features and lag-1 outcome",
            "Predict next year = this year (no model)",
            "Within estimator with firm fixed effects and firm characteristics",
        ],
    })

    st.dataframe(
        comparison_df.set_index("Model").style.format({
            "RMSE (days)": "{:.2f}",
            "MAE (days)": "{:.2f}",
        }),
        use_container_width=True,
    )

    xgb_rmse_improvement = 100 * (metrics["baseline_lag1"]["rmse"] - metrics["xgboost"]["rmse"]) / metrics["baseline_lag1"]["rmse"]
    xgb_mae_difference = 100 * (metrics["xgboost"]["mae"] - metrics["baseline_lag1"]["mae"]) / metrics["baseline_lag1"]["mae"]

    st.markdown(f"""
    **The honest read:**

    - XGBoost beats the lag-1 baseline by **{xgb_rmse_improvement:.1f}% on RMSE**
      ({metrics['xgboost']['rmse']:.2f} vs {metrics['baseline_lag1']['rmse']:.2f} days).
    - XGBoost is **{xgb_mae_difference:.1f}% worse on MAE**
      ({metrics['xgboost']['mae']:.2f} vs {metrics['baseline_lag1']['mae']:.2f} days).
    - Lag-1 dominates feature importance in this model — most predictive
      power comes from "this firm tends to pay around the same time as last
      year." This is why the baseline is so hard to beat.
    """)

    importance_df = pd.DataFrame([
        {"Feature": k, "Importance": v}
        for k, v in metrics["feature_importance"].items()
    ]).sort_values("Importance", ascending=True)

    feature_labels = {
        "avg_time_to_pay_lag_1": "Lag-1 payment time",
        "capital_intensity": "Capital intensity",
        "log_total_assets": "Log total assets",
        "profit_margin": "Profit margin",
        "leverage": "Leverage",
        "net_working_capital": "Net working capital",
        "debtors_to_turnover": "Debtors / turnover",
    }
    importance_df["Feature label"] = importance_df["Feature"].map(feature_labels)

    importance_chart = (
        alt.Chart(importance_df)
        .mark_bar(color="#3a5a7c")
        .encode(
            x=alt.X("Importance:Q", title="Feature importance (gain)"),
            y=alt.Y("Feature label:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Feature label:N", title="Feature"),
                alt.Tooltip("Importance:Q", format=".4f"),
            ],
        )
        .properties(height=300, title="Regressor feature importance")
    )
    st.altair_chart(importance_chart, use_container_width=True)

    st.divider()

    # Classifier section
    st.subheader("Late-payment classifier")

    cls = metrics["classifier"]

    st.markdown(f"""
    A separate XGBoost classifier predicts the probability that a firm will
    pay more than 50% of its invoices outside its agreed terms. This is the
    model that powers the "Late payment risk" section of the Firm explorer.

    **Performance on the 2024 hold-out ({cls['n_validation']:,} firm-years):**

    - **ROC AUC: {cls['roc_auc']:.3f}** — substantively above the 0.50 chance
      baseline. The classifier has real predictive power.
    - **Accuracy: {cls['accuracy']*100:.1f}%** — but accuracy alone is
      misleading on imbalanced data. A model predicting "on time" for
      everyone would score ~90%.
    - **Recall on late firms: {cls['late_recall']*100:.0f}%** — the model
      correctly identifies 62% of firms that actually paid late.
    - **Precision on late firms: {cls['late_precision']*100:.0f}%** — of
      firms the model flags as late, 25% actually were. This is the cost
      of class weighting: more false positives, but more true late firms
      caught.

    The model was trained with `scale_pos_weight` set to balance the rare
    "late" class ({cls['pct_late_in_train']*100:.1f}% of training firm-years).
    This produces a recall-oriented classifier appropriate for the dashboard's
    purpose: flagging firms that warrant a closer look.
    """)

    cls_importance_df = pd.DataFrame([
        {"Feature": k, "Importance": v}
        for k, v in cls["feature_importance"].items()
    ]).sort_values("Importance", ascending=True)
    cls_importance_df["Feature label"] = cls_importance_df["Feature"].map(feature_labels)

    cls_importance_chart = (
        alt.Chart(cls_importance_df)
        .mark_bar(color="#c45a3b")
        .encode(
            x=alt.X("Importance:Q", title="Feature importance (gain)"),
            y=alt.Y("Feature label:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Feature label:N", title="Feature"),
                alt.Tooltip("Importance:Q", format=".4f"),
            ],
        )
        .properties(height=300, title="Classifier feature importance")
    )
    st.altair_chart(cls_importance_chart, use_container_width=True)

    st.markdown("""
    **Notable contrast with the regressor.** In the days regressor, lag-1
    dominates feature importance at ~73%. In the classifier, lag-1 is only
    ~20%, and firm characteristics (log total assets, capital intensity,
    debtors-to-turnover, profit margin, leverage, working capital) all carry
    13-15% importance each. This suggests the binary late/on-time outcome is
    substantially more responsive to firm characteristics than the continuous
    days outcome is — including capital intensity, which is consistent with
    the dissertation's main associational finding.
    """)

    st.divider()

    st.subheader("Methodology note")

    st.markdown(f"""
    **Training set.** 2017-2023 firm-years from the analytical panel, after
    dropping rows with missing lag-1 or feature values and dropping firms
    with only one usable observation. Final training set: 4,559 firm-years
    across 1,139 firms. (2017 is included here because the lag structure
    requires it — 2018's lag-1 is built from 2017's actual. 2017 is excluded
    from the Sector dashboard's time-series chart for a different reason:
    only 11 firms filed in PPR's partial first year.)

    **Validation.** 2024 firm-years held out from training. After dropping
    rows with incomplete features, **{metrics['n_validation']:,} firm-years**
    were used for both the regressor and classifier comparisons above.

    **XGBoost hyperparameters.** Both models: 300 trees, max depth 4,
    learning rate 0.05, subsample 0.8, column-subsample 0.8, random state 42.
    The classifier additionally uses `scale_pos_weight ≈ 6.0` to balance the
    rare "late" class.

    **Panel AR.** Within estimator (firm fixed effects via demeaning)
    estimated by OLS on the demeaned training data. Mathematically
    equivalent to PanelOLS with `entity_effects=True`.

    **Baseline.** Lag-1 — predict that next year's payment time equals
    this year's. No model is fitted.
    """)