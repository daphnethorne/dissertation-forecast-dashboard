"""
Build the forecast-ready panel from the analytical panel.

Run this from the repo root:
    python scripts/02_build_forecast_panel.py

Reads:  data/analytical_panel.parquet
Writes: data/forecast_panel.parquet

What this script does:
    1. Loads the analytical panel (8,370 firm-years, 90 columns).
    2. Sorts by firm and year so we can build lags reliably.
    3. Adds lagged outcome variables (Average time to pay at t-1, t-2, t-3).
    4. One-hot encodes sectors into 17 indicator columns.
    5. Marks train / validation / forecast splits per Decision C:
        - train:      2017 to 2023 inclusive
        - validation: 2024 (held out for model accuracy reporting)
        - forecast:   2025 (treated as partial data, used for projection only)
    6. Selects a tight feature set for downstream modelling.
    7. Reports row counts at each filtering step so we can see what was dropped.
    8. Writes the result to data/forecast_panel.parquet.

This is the second script in the pipeline. It is read-only against the
analytical panel - never modifies it - and produces a new derived file
that downstream model training and dashboard code will read from.
"""

from pathlib import Path

import pandas as pd

# Source and destination paths. Both relative to the repo root, where this
# script expects to be run from.
SOURCE_PATH = Path("data/analytical_panel.parquet")
DESTINATION_PATH = Path("data/forecast_panel.parquet")

# Column names confirmed from the inspection script output.
FIRM_ID_COL = "crn_clean"
YEAR_COL = "year"
SECTOR_COL = "sector_name"
OUTCOME_COL = "Average time to pay"

# Decision C: train/validation/forecast year boundaries.
# Train on the bulk of the panel, hold out 2024 for honest accuracy reporting,
# and use 2025 only as the launching point for the production forecast
# (2025 data is partial - only 980 firm-years vs ~1,100 in earlier years).
TRAIN_YEARS = list(range(2017, 2024))   # 2017..2023 inclusive
VALIDATION_YEARS = [2024]
FORECAST_YEARS = [2025]

# Features we'll feed to the forecast models. These are the columns
# the panel-AR and XGBoost models will read. Everything else gets dropped.
# Naming convention: lowercase, underscores, no spaces. Matches the panel's
# already-engineered feature columns; the lag features we add follow the same.
NUMERIC_FEATURES = [
    "capital_intensity",
    "log_total_assets",
    "profit_margin",
    "leverage",
    "net_working_capital",
    "debtors_to_turnover",
]


def load_panel() -> pd.DataFrame:
    """Read the analytical panel from disk."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find the analytical panel at {SOURCE_PATH}. "
            "Run this script from the repository root."
        )
    return pd.read_parquet(SOURCE_PATH)


def add_lagged_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged payment time variables (t-1, t-2, t-3) per firm.

    These are the most important predictors in any time-series forecast:
    the best predictor of next year's payment behaviour is usually this
    year's. We build three lags so the XGBoost model has multi-year history
    to learn from; the panel AR model uses only the t-1 lag.

    Critically, we sort by firm and year before shifting, so the lag for
    firm X year 2020 is firm X year 2019 - not firm Y from any year.
    """
    print(f"  Building lag features (t-1, t-2, t-3 for '{OUTCOME_COL}')...")

    panel = panel.sort_values([FIRM_ID_COL, YEAR_COL]).copy()

    for lag in [1, 2, 3]:
        new_col = f"avg_time_to_pay_lag_{lag}"
        panel[new_col] = panel.groupby(FIRM_ID_COL)[OUTCOME_COL].shift(lag)
        non_null = panel[new_col].notna().sum()
        print(f"    {new_col}: {non_null:,} non-null ({100*non_null/len(panel):.1f}% of panel)")

    return panel


def add_sector_dummies(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the sector column into 17 indicator columns.

    Machine learning models can't handle string-valued categorical features
    directly. We convert "Manufacturing" / "Construction" / etc. into binary
    columns: sector_Manufacturing = 1 if firm is in Manufacturing else 0.
    """
    print(f"  One-hot encoding sectors from '{SECTOR_COL}'...")

    # drop_first=False keeps all 17 sectors as columns. With firm fixed
    # effects in the panel-AR model, dropping one would conflate that
    # sector with the constant. Keeping all 17 is cleaner.
    sector_dummies = pd.get_dummies(panel[SECTOR_COL], prefix="sector", drop_first=False)
    print(f"    Created {len(sector_dummies.columns)} sector indicator columns")

    panel = pd.concat([panel, sector_dummies], axis=1)
    return panel


def add_split_flag(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Mark each firm-year as train / validation / forecast based on year.

    The split flag lets downstream model training code filter rows in one
    line instead of repeatedly hardcoding year ranges. It also makes the
    splits explicit in the saved panel - anyone inspecting the parquet
    can see exactly which firm-years are in which set.
    """
    print(f"  Marking train/validation/forecast splits...")

    def assign_split(year: int) -> str:
        if year in TRAIN_YEARS:
            return "train"
        elif year in VALIDATION_YEARS:
            return "validation"
        elif year in FORECAST_YEARS:
            return "forecast"
        else:
            return "unknown"

    panel["split"] = panel[YEAR_COL].apply(assign_split)

    split_counts = panel["split"].value_counts()
    for split, count in split_counts.items():
        print(f"    {split}: {count:,} firm-years")

    return panel


def select_feature_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns downstream code will use; drop everything else.

    The analytical panel has 90 columns but most are either redundant or
    not relevant for forecasting (e.g. retention clause details, dispute
    resolution text). We keep:
        - Identifiers (firm, year, sector for display)
        - Outcome variable (Average time to pay)
        - Lagged outcomes (3 lags we just built)
        - Numeric features (the 6 in NUMERIC_FEATURES)
        - Sector dummies (the 17 we just built)
        - Split flag
        - Plus a couple of display-friendly columns the dashboard will use
    """
    print(f"  Selecting feature columns...")

    sector_dummy_cols = [c for c in panel.columns if c.startswith("sector_")]

    # Display columns aren't features but the dashboard needs them
    # for showing firm names, sectors, and the outcome on charts.
    display_cols = ["Company name", "sector_letter", "sector_name"]

    keep_cols = (
        [FIRM_ID_COL, YEAR_COL, "split"]      # identifiers + split flag
        + display_cols                         # display columns (sector_letter, sector_name)
        + [OUTCOME_COL]                        # outcome to forecast
        + [f"avg_time_to_pay_lag_{i}" for i in [1, 2, 3]]
        + NUMERIC_FEATURES
        + sector_dummy_cols
    )

    # Deduplicate while preserving order. The display columns include
    # sector_letter and sector_name; the sector_dummy_cols also start with
    # "sector_" but are distinct (one-hot indicators). dict.fromkeys keeps
    # first occurrence which is what we want.
    keep_cols = list(dict.fromkeys(keep_cols))


    # Filter to what's actually present (defensive - some columns might
    # be missing from a future version of the panel).
    present = [c for c in keep_cols if c in panel.columns]
    missing = set(keep_cols) - set(present)
    if missing:
        print(f"    WARNING: requested columns not found: {missing}")

    panel = panel[present].copy()
    print(f"    Kept {len(panel.columns)} columns (from {len(present)} expected)")

    return panel


def report_missingness(panel: pd.DataFrame) -> None:
    """
    Print which feature columns have missing values in the training set.

    We care most about missingness in the training set because that's
    what determines how much of our panel we can actually use for fitting.
    Lagged-outcome missingness for year-1 firms is expected (~12% per lag);
    anything beyond that warrants explicit attention.
    """
    print(f"  Missingness audit (training set only)...")

    train_only = panel[panel["split"] == "train"]
    feature_cols = (
        [f"avg_time_to_pay_lag_{i}" for i in [1, 2, 3]]
        + NUMERIC_FEATURES
    )

    print(f"    Training set size: {len(train_only):,} firm-years")
    for col in feature_cols:
        if col not in train_only.columns:
            print(f"    {col:<35} NOT IN PANEL")
            continue
        non_null = train_only[col].notna().sum()
        null_pct = 100 * (1 - non_null / len(train_only))
        print(f"    {col:<35} {non_null:>6,} non-null ({null_pct:>5.1f}% missing)")


def save_panel(panel: pd.DataFrame) -> None:
    """Write the forecast panel to disk as parquet."""
    print(f"  Writing to {DESTINATION_PATH}...")
    panel.to_parquet(DESTINATION_PATH, index=False)
    size_kb = DESTINATION_PATH.stat().st_size / 1024
    print(f"    Wrote {len(panel):,} rows, {len(panel.columns)} columns ({size_kb:.0f} KB)")


def main() -> None:
    print()
    print("BUILDING FORECAST PANEL")
    print(f"Source:      {SOURCE_PATH.absolute()}")
    print(f"Destination: {DESTINATION_PATH.absolute()}")
    print()

    print("=" * 60)
    print("STEP 1: Load analytical panel")
    print("=" * 60)
    panel = load_panel()
    print(f"  Loaded {len(panel):,} rows, {len(panel.columns)} columns")
    print()

    print("=" * 60)
    print("STEP 2: Build lagged outcome variables")
    print("=" * 60)
    panel = add_lagged_outcomes(panel)
    print()

    print("=" * 60)
    print("STEP 3: One-hot encode sectors")
    print("=" * 60)
    panel = add_sector_dummies(panel)
    print()

    print("=" * 60)
    print("STEP 4: Mark train/validation/forecast splits")
    print("=" * 60)
    panel = add_split_flag(panel)
    print()

    print("=" * 60)
    print("STEP 5: Select feature columns")
    print("=" * 60)
    panel = select_feature_columns(panel)
    print()

    print("=" * 60)
    print("STEP 6: Missingness audit")
    print("=" * 60)
    report_missingness(panel)
    print()

    print("=" * 60)
    print("STEP 7: Save forecast panel")
    print("=" * 60)
    save_panel(panel)
    print()

    print("=" * 60)
    print("FORECAST PANEL BUILD COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()