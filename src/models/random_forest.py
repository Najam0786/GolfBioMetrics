"""
random_forest.py
Model 3: Random Forest Regressor — Predict carry distance (yards).

Rationale: Captures non-linear interactions between metrics.
Feature importance shows which metrics matter most for distance prediction.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.features.engineering import get_feature_names


TARGET = "carry_distance_yards"


def train_random_forest(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = TARGET,
    n_estimators: int = 200,
    max_depth: int = 12,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Trains a Random Forest regressor to predict carry distance.

    Args:
        df:           DataFrame with features and target
        feature_cols: feature names (defaults to canonical 20)
        target_col:   name of the regression target
        n_estimators: number of trees (200 is robust and interpretable)
        max_depth:    maximum tree depth
        test_size:    hold-out test fraction
        val_size:     validation fraction
        random_state: reproducibility seed

    Returns:
        dict with 'model', 'scaler', 'results', 'feature_importance'
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

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    cv_scores = cross_val_score(
        RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                               min_samples_leaf=3, random_state=random_state, n_jobs=-1),
        scaler.transform(X_temp), y_temp, cv=5, scoring="r2"
    )

    results = {
        "train": _evaluate(model, X_train_s, y_train),
        "val":   _evaluate(model, X_val_s,   y_val),
        "test":  _evaluate(model, X_test_s,  y_test),
        "cv_r2_mean": round(float(np.mean(cv_scores)), 4),
        "cv_r2_std":  round(float(np.std(cv_scores)), 4),
    }

    feature_importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
        "importance_pct": (model.feature_importances_ * 100).round(1),
    }).sort_values("importance", ascending=False)

    return {
        "model":              model,
        "scaler":             scaler,
        "results":            results,
        "feature_names":      feature_cols,
        "feature_importance": feature_importance,
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


def predict_carry_distance(
    features: np.ndarray,
    model,
    scaler,
) -> np.ndarray:
    """Predicts carry distance for new observations."""
    return model.predict(scaler.transform(features))


def save_model(artifacts: dict, output_dir: str = "outputs/model_artifacts") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "random_forest.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": artifacts["model"], "scaler": artifacts["scaler"],
                     "feature_names": artifacts["feature_names"]}, f)
    return path


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
