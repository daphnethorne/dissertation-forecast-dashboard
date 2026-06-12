# Building the forecast-ready panel from the analytical panel.
# Adds lagged outcomes, sector dummies, and a train/val/forecast split flag.
# Running from repo root: python scripts/02_build_forecast_panel.py

from pathlib import Path
import pandas as pd

SOURCE_PATH = Path("data/analytical_panel.parquet")
DESTINATION_PATH = Path("data/forecast_panel.parquet")

# Column names in the analytical panel
FIRM_ID_COL = "crn_clean"
YEAR_COL = "year"
SECTOR_COL = "sector_name"
OUTCOME_COL = "Average time to pay"

# Train on 2017-2023, hold out 2024 for accuracy testing, use 2025 as the
# starting point for forecasts (2025 is partial data - only 980 firm-years).
TRAIN_YEARS = list(range(2017, 2024))
VALIDATION_YEARS = [2024]
FORECAST_YEARS = [2025]

# Numeric features for the forecast models
NUMERIC_FEATURES = [
    "capital_intensity",
    "log_total_assets",
    "profit_margin",
    "leverage",
    "net_working_capital",
    "debtors_to_turnover",
]


def load_panel():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"No panel at {SOURCE_PATH}. Run from repo root.")
    return pd.read_parquet(SOURCE_PATH)


def add_lagged_outcomes(panel):
    # Sort by firm and year so lags are computed correctly within each firm.
    panel = panel.sort_values([FIRM_ID_COL, YEAR_COL]).copy()

    for lag in [1, 2, 3]:
        new_col = f"avg_time_to_pay_lag_{lag}"
        panel[new_col] = panel.groupby(FIRM_ID_COL)[OUTCOME_COL].shift(lag)
        non_null = panel[new_col].notna().sum()
        print(f"  {new_col}: {non_null:,} non-null ({100*non_null/len(panel):.1f}%)")

    return panel


def add_sector_dummies(panel):
    # One-hot encode sectors. drop_first=False keeps all 17 columns.
    dummies = pd.get_dummies(panel[SECTOR_COL], prefix="sector", drop_first=False)
    print(f"  Created {len(dummies.columns)} sector dummies")
    return pd.concat([panel, dummies], axis=1)


def add_split_flag(panel):
    def assign_split(year):
        if year in TRAIN_YEARS:
            return "train"
        elif year in VALIDATION_YEARS:
            return "validation"
        elif year in FORECAST_YEARS:
            return "forecast"
        else:
            return "unknown"

    panel["split"] = panel[YEAR_COL].apply(assign_split)
    print(panel["split"].value_counts().to_string())
    return panel


def select_feature_columns(panel):
    sector_dummy_cols = [c for c in panel.columns if c.startswith("sector_")]
    display_cols = ["Company name", "sector_letter", "sector_name"]

    keep_cols = (
        [FIRM_ID_COL, YEAR_COL, "split"]
        + display_cols
        + [OUTCOME_COL]
        + [f"avg_time_to_pay_lag_{i}" for i in [1, 2, 3]]
        + NUMERIC_FEATURES
        + sector_dummy_cols
    )

    # Remove duplicates while preserving order
    keep_cols = list(dict.fromkeys(keep_cols))

    present = [c for c in keep_cols if c in panel.columns]
    missing = set(keep_cols) - set(present)
    if missing:
        print(f"  Missing columns: {missing}")

    panel = panel[present].copy()
    print(f"  Kept {len(panel.columns)} columns")
    return panel


def report_missingness(panel):
    train_only = panel[panel["split"] == "train"]
    feature_cols = [f"avg_time_to_pay_lag_{i}" for i in [1, 2, 3]] + NUMERIC_FEATURES

    print(f"  Train set: {len(train_only):,} firm-years")
    for col in feature_cols:
        if col not in train_only.columns:
            continue
        non_null = train_only[col].notna().sum()
        null_pct = 100 * (1 - non_null / len(train_only))
        print(f"  {col}: {non_null:,} non-null ({null_pct:.1f}% missing)")


def save_panel(panel):
    panel.to_parquet(DESTINATION_PATH, index=False)
    size_kb = DESTINATION_PATH.stat().st_size / 1024
    print(f"  Wrote {len(panel):,} rows, {len(panel.columns)} columns ({size_kb:.0f} KB)")


def main():
    print("Loading panel...")
    panel = load_panel()
    print(f"  Loaded {len(panel):,} rows, {len(panel.columns)} columns\n")

    print("Building lag features...")
    panel = add_lagged_outcomes(panel)
    print()

    print("Adding sector dummies...")
    panel = add_sector_dummies(panel)
    print()

    print("Marking splits...")
    panel = add_split_flag(panel)
    print()

    print("Selecting columns...")
    panel = select_feature_columns(panel)
    print()

    print("Missingness in training set:")
    report_missingness(panel)
    print()

    print("Saving forecast panel...")
    save_panel(panel)


if __name__ == "__main__":
    main()