# Train forecast models from the forecast panel and save artefacts.
# Trains: lag-1 baseline (no model, just metric), panel AR (within estimator),
# and XGBoost. Saves XGBoost as the production model along with comparison metrics.
# Run from repo root: python scripts/03_train_models.py

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

PANEL_PATH = Path("data/forecast_panel.parquet")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "avg_time_to_pay_lag_1",
    "capital_intensity",
    "log_total_assets",
    "profit_margin",
    "leverage",
    "net_working_capital",
    "debtors_to_turnover",
]
OUTCOME_COL = "Average time to pay"
FIRM_ID_COL = "crn_clean"


def load_and_filter():
    panel = pd.read_parquet(PANEL_PATH)
    train = panel[panel["split"] == "train"].copy()
    validation = panel[panel["split"] == "validation"].copy()

    # dropping rows missing lag-1 or features.
    train = train.dropna(subset=FEATURE_COLS + [OUTCOME_COL]).copy()
    validation = validation.dropna(subset=FEATURE_COLS + [OUTCOME_COL]).copy()

    # dropping firms with only one observation in training (within estimator can't use them).
    firm_counts = train[FIRM_ID_COL].value_counts()
    firms_with_panel = firm_counts[firm_counts >= 2].index
    train = train[train[FIRM_ID_COL].isin(firms_with_panel)].copy()

    return train, validation


def fit_panel_ar(train):
    # Within estimator via manual demeaning: equivalent to PanelOLS with entity FE.
    train_indexed = train.set_index([FIRM_ID_COL, "year"])

    firm_means_y = train_indexed.groupby(level=FIRM_ID_COL)[OUTCOME_COL].mean()
    firm_means_X = train_indexed.groupby(level=FIRM_ID_COL)[FEATURE_COLS].mean()

    y_dm = train_indexed[OUTCOME_COL] - train_indexed.index.get_level_values(FIRM_ID_COL).map(firm_means_y)
    X_dm = train_indexed[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X_dm[col] = train_indexed[col] - train_indexed.index.get_level_values(FIRM_ID_COL).map(firm_means_X[col])

    model = LinearRegression(fit_intercept=False)
    model.fit(X_dm, y_dm)

    return model, firm_means_y, firm_means_X


def predict_panel_ar(model, validation, firm_means_y, firm_means_X, grand_mean_y, grand_mean_X):
    val_indexed = validation.set_index([FIRM_ID_COL, "year"])

    X_dm = val_indexed[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        firm_mean_col = val_indexed.index.get_level_values(FIRM_ID_COL).map(firm_means_X[col])
        firm_mean_col = firm_mean_col.fillna(grand_mean_X[col])
        X_dm[col] = val_indexed[col] - firm_mean_col

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y_dm_pred = model.predict(X_dm)

    firm_mean_y_for_val = val_indexed.index.get_level_values(FIRM_ID_COL).map(firm_means_y)
    firm_mean_y_for_val = pd.Series(firm_mean_y_for_val, index=val_indexed.index).fillna(grand_mean_y)

    return y_dm_pred + firm_mean_y_for_val.values


def fit_xgboost(train):
    X_train = train[FEATURE_COLS].values
    y_train = train[OUTCOME_COL].values

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_metrics(predictions, actuals):
    errors = predictions - actuals
    rmse = float(np.sqrt((errors ** 2).mean()))
    mae = float(np.abs(errors).mean())
    return rmse, mae


def main():
    print("Loading panel...")
    train, validation = load_and_filter()
    print(f"  Train: {len(train):,} firm-years ({train[FIRM_ID_COL].nunique():,} firms)")
    print(f"  Validation: {len(validation):,} firm-years")
    print()

    # fitting and evaluating panel AR.
    print("Fitting panel AR (within estimator)...")
    panel_ar, firm_means_y, firm_means_X = fit_panel_ar(train)
    grand_mean_y = train[OUTCOME_COL].mean()
    grand_mean_X = train[FEATURE_COLS].mean()
    panel_ar_preds = predict_panel_ar(panel_ar, validation, firm_means_y, firm_means_X, grand_mean_y, grand_mean_X)
    panel_ar_rmse, panel_ar_mae = compute_metrics(panel_ar_preds, validation[OUTCOME_COL].values)
    print(f"  Panel AR RMSE: {panel_ar_rmse:.2f}, MAE: {panel_ar_mae:.2f}")
    print()

    # fitting and evaluating XGBoost.
    print("Fitting XGBoost...")
    xgb_model = fit_xgboost(train)
    xgb_preds = xgb_model.predict(validation[FEATURE_COLS].values)
    xgb_rmse, xgb_mae = compute_metrics(xgb_preds, validation[OUTCOME_COL].values)
    print(f"  XGBoost RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}")
    print()

    # Baseline: just predict lag-1.
    baseline_preds = validation["avg_time_to_pay_lag_1"].values
    baseline_rmse, baseline_mae = compute_metrics(baseline_preds, validation[OUTCOME_COL].values)
    print(f"Baseline (lag-1) RMSE: {baseline_rmse:.2f}, MAE: {baseline_mae:.2f}")
    print()

    # Save artefacts.
    print("Saving artefacts...")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.joblib")

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    metrics = {
        "n_validation": int(len(validation)),
        "xgboost": {"rmse": xgb_rmse, "mae": xgb_mae},
        "baseline_lag1": {"rmse": baseline_rmse, "mae": baseline_mae},
        "panel_ar": {"rmse": panel_ar_rmse, "mae": panel_ar_mae},
        "feature_importance": dict(zip(FEATURE_COLS, [float(x) for x in xgb_model.feature_importances_])),
    }
    with open(MODELS_DIR / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()