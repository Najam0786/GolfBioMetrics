"""
xgboost_model.py
Model 4: XGBoost + SHAP — Predict injury risk score with per-golfer explanations.

Rationale: Best predictive accuracy + SHAP values provide individual-level
explanations. "This golfer's injury risk is 0.72 — driven primarily by
reverse pivot severity."
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

from src.features.engineering import get_feature_names

TARGET = "injury_risk_score"


def train_xgboost(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = TARGET,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Trains an XGBoost regressor for injury risk prediction with SHAP explanations.

    Args:
        df:            DataFrame with features and target
        feature_cols:  feature names (defaults to canonical 20)
        target_col:    regression target (injury_risk_score)
        n_estimators:  number of boosting rounds
        max_depth:     tree depth (5 balances accuracy and interpretability)
        learning_rate: step size shrinkage
        test_size:     hold-out test fraction
        val_size:      validation fraction
        random_state:  reproducibility seed

    Returns:
        dict with 'model', 'scaler', 'results', 'feature_importance', 'shap_values'
    """
    if feature_cols is None:
        feature_cols = [c for c in get_feature_names() if c in df.columns]

    df_clean = df[feature_cols + [target_col]].dropna()
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False,
    )

    results = {
        "train": _evaluate(model, X_train_s, y_train),
        "val":   _evaluate(model, X_val_s,   y_val),
        "test":  _evaluate(model, X_test_s,  y_test),
    }

    feature_importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    shap_values = _compute_shap(model, X_test_s, feature_cols)

    return {
        "model":              model,
        "scaler":             scaler,
        "results":            results,
        "feature_names":      feature_cols,
        "feature_importance": feature_importance,
        "shap_values":        shap_values,
        "X_test":             X_test_s,
        "y_test":             y_test,
        "n_train":            len(X_train),
        "n_val":              len(X_val),
        "n_test":             len(X_test),
    }


def _evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "r2":   round(float(r2_score(y, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "mae":  round(float(mean_absolute_error(y, pred)), 4),
    }


def _compute_shap(model, X: np.ndarray, feature_cols: list):
    """Computes SHAP values for test set. Returns None if shap not installed."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X)
        return shap_values
    except ImportError:
        return None


def explain_golfer(
    features_row: np.ndarray,
    model,
    scaler,
    feature_names: list,
    golfer_id: int = None,
) -> str:
    """
    Generates a human-readable injury risk explanation for one golfer.

    Uses SHAP TreeExplainer to attribute risk contributions per feature.

    Args:
        features_row: 1D array of feature values (unscaled)
        model:        trained XGBoost model
        scaler:       fitted StandardScaler
        feature_names: list of feature name strings
        golfer_id:    optional identifier for the output

    Returns:
        formatted string explanation
    """
    try:
        import shap
        x_scaled = scaler.transform(features_row.reshape(1, -1))
        pred = float(model.predict(x_scaled)[0])
        risk_level = "HIGH" if pred > 0.6 else "MODERATE" if pred > 0.3 else "LOW"

        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = explainer.shap_values(x_scaled)[0]

        contributions = sorted(
            zip(feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        label = f"Golfer ID: {golfer_id}" if golfer_id else "Golfer"
        lines = [
            f"{label} | Injury Risk: {pred:.2f} ({risk_level})",
            "─" * 50,
        ]
        for fname, contrib in contributions[:5]:
            direction = "↑ (increases risk)" if contrib > 0 else "↓ (reduces risk)"
            lines.append(f"  {fname:<35}: {contrib:+.3f}  {direction}")

        if risk_level == "HIGH":
            top_risk_feature = contributions[0][0]
            lines.append("─" * 50)
            lines.append(f"Recommendation: Address {top_risk_feature} pattern immediately.")
            lines.append("Risk of injury is elevated if training volume increases.")

        return "\n".join(lines)

    except ImportError:
        x_scaled = scaler.transform(features_row.reshape(1, -1))
        pred = float(model.predict(x_scaled)[0])
        return f"Injury Risk: {pred:.2f} (SHAP not installed for detailed breakdown)"


def predict_injury_risk(
    features: np.ndarray,
    model,
    scaler,
) -> np.ndarray:
    """Predicts injury risk score [0–1] for new observations."""
    preds = model.predict(scaler.transform(features))
    return np.clip(preds, 0.0, 1.0)


def save_model(artifacts: dict, output_dir: str = "outputs/model_artifacts") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "xgboost_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": artifacts["model"], "scaler": artifacts["scaler"],
                     "feature_names": artifacts["feature_names"]}, f)
    return path


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
