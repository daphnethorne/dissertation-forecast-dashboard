# Inspecting the analytical panel: dimensions, columns, missingness, firms, years, sectors.
# Run from repo root: python scripts/01_inspect_panel.py

from pathlib import Path
import pandas as pd

PANEL_PATH = Path("data/analytical_panel.parquet")

FIRM_ID_COL = "crn_clean"
YEAR_COL = "year"
SECTOR_LETTER_COL = "sector_letter"
SECTOR_NAME_COL = "sector_name"


def load_panel():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"No panel at {PANEL_PATH}. Run from repo root.")
    return pd.read_parquet(PANEL_PATH)


def report_dimensions(panel):
    print(f"Rows: {len(panel):,}")
    print(f"Columns: {len(panel.columns)}")
    print(f"Memory: {panel.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()


def report_columns(panel):
    print("Columns and missingness:")
    for col in panel.columns:
        dtype = str(panel[col].dtype)
        non_null = panel[col].notna().sum()
        null_pct = 100 * (1 - non_null / len(panel))
        print(f"  {col:<55} {dtype:<18} {non_null:>6,} non-null  ({null_pct:.1f}% missing)")
    print()


def report_firm_coverage(panel):
    if FIRM_ID_COL not in panel.columns:
        print(f"Firm ID column '{FIRM_ID_COL}' not found")
        return

    n_firms = panel[FIRM_ID_COL].nunique()
    years_per_firm = panel.groupby(FIRM_ID_COL).size()

    print(f"Unique firms: {n_firms:,}")
    print(f"Median years per firm: {years_per_firm.median():.1f}")
    print(f"Mean years per firm: {years_per_firm.mean():.2f}")
    print(f"Min/max years per firm: {years_per_firm.min()} / {years_per_firm.max()}")
    print()

    print("Firms by number of years observed:")
    distribution = years_per_firm.value_counts().sort_index()
    for n_years, count in distribution.items():
        pct = 100 * count / n_firms
        print(f"  {n_years} years: {count:,} firms ({pct:.1f}%)")
    print()


def report_year_coverage(panel):
    if YEAR_COL not in panel.columns:
        print(f"Year column '{YEAR_COL}' not found")
        return

    print("Firm-years per calendar year:")
    year_counts = panel[YEAR_COL].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,}")
    print()


def report_sector_distribution(panel):
    if SECTOR_NAME_COL not in panel.columns:
        print(f"Sector column '{SECTOR_NAME_COL}' not found")
        return

    print("Sector distribution:")
    sector_counts = panel[SECTOR_NAME_COL].value_counts()
    for sector, count in sector_counts.items():
        pct = 100 * count / len(panel)
        print(f"  {str(sector):<50} {count:,} ({pct:.1f}%)")
    print()


def report_forecast_feasibility(panel):
    # How many firms can be forecasted at each model tier?
    if FIRM_ID_COL not in panel.columns:
        return

    years_per_firm = panel.groupby(FIRM_ID_COL).size()
    n_firms = len(years_per_firm)

    ar1_eligible = (years_per_firm >= 2).sum()
    xgb_eligible = (years_per_firm >= 4).sum()
    robust_eligible = (years_per_firm >= 6).sum()

    print("Forecast feasibility:")
    print(f"  AR(1) eligible (>= 2 years): {ar1_eligible:,} of {n_firms} ({100*ar1_eligible/n_firms:.1f}%)")
    print(f"  XGBoost eligible (>= 4 years): {xgb_eligible:,} of {n_firms} ({100*xgb_eligible/n_firms:.1f}%)")
    print(f"  Robust eligible (>= 6 years): {robust_eligible:,} of {n_firms} ({100*robust_eligible/n_firms:.1f}%)")
    print()


def main():
    print(f"Loading panel from {PANEL_PATH}\n")
    panel = load_panel()

    report_dimensions(panel)
    report_columns(panel)
    report_firm_coverage(panel)
    report_year_coverage(panel)
    report_sector_distribution(panel)
    report_forecast_feasibility(panel)


if __name__ == "__main__":
    main()