"""
Inspect the analytical panel and print what's actually in it.

Run this from the repo root:
    python scripts/01_inspect_panel.py

This is the first script in the dashboard pipeline. It reads the parquet
file i copied across from the dissertation repo and tells me what i am
working with dimensions, columns, missingness, firms, years, sectors.

The point is to confirm the panel matches the dissertation's analytical
sample (8,370 firm years across 1,390 firms over 2017 to 2025) before i
build anything on top of it. If the numbers ever stop matching, future
runs of this script will catch it.

The script doesn't modify any data. Read only by design.
"""

from pathlib import Path

import pandas as pd

# i expect to be run from the repo root, not the scripts/ directory.
PANEL_PATH = Path("data/analytical_panel.parquet")

# Column names confirmed against the actual panel (90 columns total).
FIRM_ID_COL = "crn_clean"          # cleaned companies House registration number
YEAR_COL = "year"                  # Reporting year (2017 to 2025)
SECTOR_LETTER_COL = "sector_letter"  # ONS broad sector letter (A to S, excluding K)
SECTOR_NAME_COL = "sector_name"    


def load_panel() -> pd.DataFrame:
    """Read the parquet file and return it as a DataFrame."""
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find the analytical panel at {PANEL_PATH}. "
            "Make sure you're running this from the repository root."
        )
    return pd.read_parquet(PANEL_PATH)


def report_dimensions(panel: pd.DataFrame) -> None:
    """Print the basic shape of the panel - how many rows, columns, how big in memory."""
    print("=" * 60)
    print("PANEL DIMENSIONS")
    print("=" * 60)
    print(f"  Rows (firm-years):      {len(panel):>8,}")
    print(f"  Columns:                {len(panel.columns):>8,}")
    print(f"  Memory usage:           {panel.memory_usage(deep=True).sum() / 1024**2:>8.2f} MB")
    print()


def report_columns(panel: pd.DataFrame) -> None:
    """Print every column with its data type and how many values are missing."""
    # This is long output, but it's the only way to see what's actually in
    # a 90 column panel.
    print("=" * 60)
    print("COLUMNS AND DATA TYPES")
    print("=" * 60)
    for col in panel.columns:
        dtype = str(panel[col].dtype)
        non_null = panel[col].notna().sum()
        null_pct = 100 * (1 - non_null / len(panel))
        print(f"  {col:<55} {dtype:<18} {non_null:>6,} non-null  ({null_pct:>5.1f}% missing)")
    print()


def report_firm_coverage(panel: pd.DataFrame) -> None:
    """Tell us how many distinct firms are in the panel and how many years each appears for."""
    if FIRM_ID_COL not in panel.columns:
        print(f"WARNING: Expected firm ID column '{FIRM_ID_COL}' not found in panel.")
        print()
        return

    print("=" * 60)
    print(f"FIRM COVERAGE")
    print(f"(using firm ID column: '{FIRM_ID_COL}')")
    print("=" * 60)

    n_firms = panel[FIRM_ID_COL].nunique()
    print(f"  Unique firms:           {n_firms:>8,}")

    # how many years does each firm show up in the panel?
    # this matters for forecast feasibility: firms with only one year
    # can't be used for any lag-based forecasting.
    years_per_firm = panel.groupby(FIRM_ID_COL).size()
    print(f"  Median years per firm:  {years_per_firm.median():>8.1f}")
    print(f"  Mean years per firm:    {years_per_firm.mean():>8.2f}")
    print(f"  Min years per firm:     {years_per_firm.min():>8,}")
    print(f"  Max years per firm:     {years_per_firm.max():>8,}")
    print()

    # showing the full distribution. This tells us how many firms will
    # be in each tier of the dashboard's forecast capability.
    print("  Distribution of firms by number of years observed:")
    distribution = years_per_firm.value_counts().sort_index()
    for n_years, n_firms_at_that_count in distribution.items():
        pct = 100 * n_firms_at_that_count / n_firms
        print(f"    {n_years} year(s):  {n_firms_at_that_count:>5,} firms  ({pct:>5.1f}%)")
    print()


def report_year_coverage(panel: pd.DataFrame) -> None:
    """Print how many firm-years are in each calendar year."""
    if YEAR_COL not in panel.columns:
        print(f"WARNING: Expected year column '{YEAR_COL}' not found in panel.")
        print()
        return

    print("=" * 60)
    print(f"YEAR COVERAGE (using year column: '{YEAR_COL}')")
    print("=" * 60)
    year_counts = panel[YEAR_COL].value_counts().sort_index()
    for year, count in year_counts.items():
        # Useful to see at a glance which years are well-populated.
        # 2017 has very few because PPR only started mid-2017,
        # so most firms only reported their first full year in 2018.
        print(f"    {year}:  {count:>5,} firm-years")
    print()


def report_sector_distribution(panel: pd.DataFrame) -> None:
    """Print how firm-years are distributed across the 17 broad sectors."""
    # i have two sector columns the letter code (A-S) and a human name.
    # Both are useful; i'll show the name based breakdown since it's more readable.
    if SECTOR_NAME_COL not in panel.columns:
        print(f"WARNING: Expected sector column '{SECTOR_NAME_COL}' not found in panel.")
        print()
        return

    print("=" * 60)
    print(f"SECTOR DISTRIBUTION (using column: '{SECTOR_NAME_COL}')")
    print("=" * 60)
    sector_counts = panel[SECTOR_NAME_COL].value_counts()
    for sector, count in sector_counts.items():
        pct = 100 * count / len(panel)
        # The flagship subset for the dissertation is Manufacturing
        # (sector C) and Administrative & Support Services (sector N).
        # Worth keeping an eye on these here too.
        print(f"    {str(sector):<50} {count:>6,}  ({pct:>5.1f}%)")
    print()


def report_forecast_feasibility(panel: pd.DataFrame) -> None:
    """Show how many firms have enough history for different forecast horizons."""
    # This is dashboard-specific. i need to know upfront how many firms
    # will get a real forecast versus an "insufficient history" message.
    if FIRM_ID_COL not in panel.columns:
        return

    print("=" * 60)
    print("FORECAST FEASIBILITY")
    print("=" * 60)
    print("  (How many firms have enough history for each forecast model?)")
    print()

    years_per_firm = panel.groupby(FIRM_ID_COL).size()
    n_firms = len(years_per_firm)

    # AR(1) model needs at least 2 years (one for lag, one to forecast from)
    ar1_eligible = (years_per_firm >= 2).sum()
    # XGBoost with 3 lags needs at least 4 years
    xgb_eligible = (years_per_firm >= 4).sum()
    # Stress test forecasts (firm fixed effects) benefit from more history
    robust_eligible = (years_per_firm >= 6).sum()

    print(f"  Firms with at least 2 years (AR(1) eligible):  {ar1_eligible:>5,} of {n_firms} ({100*ar1_eligible/n_firms:.1f}%)")
    print(f"  Firms with at least 4 years (XGBoost 3-lag):   {xgb_eligible:>5,} of {n_firms} ({100*xgb_eligible/n_firms:.1f}%)")
    print(f"  Firms with at least 6 years (robust forecast): {robust_eligible:>5,} of {n_firms} ({100*robust_eligible/n_firms:.1f}%)")
    print()


def main() -> None:
    print()
    print("ANALYTICAL PANEL INSPECTION")
    print(f"Source: {PANEL_PATH.absolute()}")
    print()

    panel = load_panel()

    report_dimensions(panel)
    report_columns(panel)
    report_firm_coverage(panel)
    report_year_coverage(panel)
    report_sector_distribution(panel)
    report_forecast_feasibility(panel)

    print("=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()