# Train forecast models from the forecast panel and save artefacts.
# Trains:
#   - Lag-1 baseline (no model, just a metric)
#   - Panel AR (within estimator)
#   - XGBoost regressor (predicts payment days)
#   - XGBoost classifier (predicts probability of late payment)
# Saves the two XGBoost models as production artefacts along with
# comparison metrics.
# Run from repo root: python scripts/03_train_models.py

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

PANEL_PATH = Path("data/forecast_panel.parquet")
ANALYTICAL_PATH = Path("data/analytical_panel.parquet")
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
LATE_THRESHOLD = 50.0


def load_and_filter():
    panel = pd.read_parquet(PANEL_PATH)
    train = panel[panel["split"] == "train"].copy()
    validation = panel[panel["split"] == "validation"].copy()

    train = train.dropna(subset=FEATURE_COLS + [OUTCOME_COL]).copy()
    validation = validation.dropna(subset=FEATURE_COLS + [OUTCOME_COL]).copy()

    firm_counts = train[FIRM_ID_COL].value_counts()
    firms_with_panel = firm_counts[firm_counts >= 2].index
    train = train[train[FIRM_ID_COL].isin(firms_with_panel)].copy()

    return train, validation


def add_late_outcome(df):
    # Merge the late-payment column from the analytical panel.
    analytical = pd.read_parquet(ANALYTICAL_PATH)
    late_col = analytical[[FIRM_ID_COL, "year", "% Invoices not paid within agreed terms"]].copy()
    late_col["is_late"] = (late_col["% Invoices not paid within agreed terms"] > LATE_THRESHOLD).astype(int)
    merged = df.merge(late_col[[FIRM_ID_COL, "year", "is_late"]], on=[FIRM_ID_COL, "year"], how="left")
    merged = merged.dropna(subset=["is_late"]).copy()
    merged["is_late"] = merged["is_late"].astype(int)
    return merged


def fit_panel_ar(train):
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


def fit_xgboost_regressor(train):
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


def fit_xgboost_classifier(train_with_late):
    X_train = train_with_late[FEATURE_COLS].values
    y_train = train_with_late["is_late"].values

    n_negatives = (1 - y_train).sum()
    n_positives = y_train.sum()
    pos_weight = n_negatives / n_positives

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
    )
    model.fit(X_train, y_train)
    return model, pos_weight


def compute_regression_metrics(predictions, actuals):
    errors = predictions - actuals
    rmse = float(np.sqrt((errors ** 2).mean()))
    mae = float(np.abs(errors).mean())
    return rmse, mae


def compute_classification_metrics(probs, preds, actuals):
    auc = float(roc_auc_score(actuals, probs))
    acc = float(accuracy_score(actuals, preds))
    report = classification_report(actuals, preds, target_names=["On time", "Late"], output_dict=True)
    return auc, acc, report


def main():
    print("Loading panel...")
    train, validation = load_and_filter()
    print(f"  Train: {len(train):,} firm-years ({train[FIRM_ID_COL].nunique():,} firms)")
    print(f"  Validation: {len(validation):,} firm-years")
    print()

    # Fit and evaluate panel AR.
    print("Fitting panel AR (within estimator)...")
    panel_ar, firm_means_y, firm_means_X = fit_panel_ar(train)
    grand_mean_y = train[OUTCOME_COL].mean()
    grand_mean_X = train[FEATURE_COLS].mean()
    panel_ar_preds = predict_panel_ar(panel_ar, validation, firm_means_y, firm_means_X, grand_mean_y, grand_mean_X)
    panel_ar_rmse, panel_ar_mae = compute_regression_metrics(panel_ar_preds, validation[OUTCOME_COL].values)
    print(f"  Panel AR RMSE: {panel_ar_rmse:.2f}, MAE: {panel_ar_mae:.2f}")
    print()

    # Fit and evaluate XGBoost regressor.
    print("Fitting XGBoost regressor...")
    regressor = fit_xgboost_regressor(train)
    xgb_preds = regressor.predict(validation[FEATURE_COLS].values)
    xgb_rmse, xgb_mae = compute_regression_metrics(xgb_preds, validation[OUTCOME_COL].values)
    print(f"  XGBoost RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}")
    print()

    # Lag-1 baseline.
    baseline_preds = validation["avg_time_to_pay_lag_1"].values
    baseline_rmse, baseline_mae = compute_regression_metrics(baseline_preds, validation[OUTCOME_COL].values)
    print(f"Baseline (lag-1) RMSE: {baseline_rmse:.2f}, MAE: {baseline_mae:.2f}")
    print()

    # Build late outcome and train classifier.
    print("Building late outcome and training XGBoost classifier...")
    train_with_late = add_late_outcome(train)
    val_with_late = add_late_outcome(validation)
    print(f"  Train: {len(train_with_late):,} firm-years, {100*train_with_late['is_late'].mean():.1f}% late")
    print(f"  Validation: {len(val_with_late):,} firm-years, {100*val_with_late['is_late'].mean():.1f}% late")

    classifier, pos_weight = fit_xgboost_classifier(train_with_late)
    print(f"  scale_pos_weight: {pos_weight:.2f}")

    X_val_cls = val_with_late[FEATURE_COLS].values
    y_val_cls = val_with_late["is_late"].values
    probs = classifier.predict_proba(X_val_cls)[:, 1]
    preds = classifier.predict(X_val_cls)
    auc, acc, report = compute_classification_metrics(probs, preds, y_val_cls)
    print(f"  Classifier ROC AUC: {auc:.3f}, Accuracy: {acc:.3f}")
    print(f"  Late recall: {report['Late']['recall']:.2f}, Late precision: {report['Late']['precision']:.2f}")
    print()

    # Save artefacts.
    print("Saving artefacts...")
    joblib.dump(regressor, MODELS_DIR / "xgboost_model.joblib")
    joblib.dump(classifier, MODELS_DIR / "xgboost_classifier.joblib")

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    metrics = {
        "n_validation": int(len(validation)),
        "xgboost": {"rmse": xgb_rmse, "mae": xgb_mae},
        "baseline_lag1": {"rmse": baseline_rmse, "mae": baseline_mae},
        "panel_ar": {"rmse": panel_ar_rmse, "mae": panel_ar_mae},
        "feature_importance": dict(zip(FEATURE_COLS, [float(x) for x in regressor.feature_importances_])),
        "classifier": {
            "n_validation": int(len(val_with_late)),
            "threshold_pct_invoices_late": LATE_THRESHOLD,
            "roc_auc": auc,
            "accuracy": acc,
            "late_recall": float(report["Late"]["recall"]),
            "late_precision": float(report["Late"]["precision"]),
            "ontime_recall": float(report["On time"]["recall"]),
            "ontime_precision": float(report["On time"]["precision"]),
            "pct_late_in_train": float(train_with_late["is_late"].mean()),
            "pct_late_in_validation": float(val_with_late["is_late"].mean()),
            "feature_importance": dict(zip(FEATURE_COLS, [float(x) for x in classifier.feature_importances_])),
        },
    }
    with open(MODELS_DIR / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()