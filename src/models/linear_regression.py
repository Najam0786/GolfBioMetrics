"""
linear_regression.py
Model 1: Linear Regression — Predict ball speed (mph) from biomechanics metrics.

Rationale: Baseline model. Coefficients have direct physical meaning and
can be audited by coaches and scientists without statistics background.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.features.engineering import get_feature_names


TARGET = "ball_speed_mph"


def train_linear_regression(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = TARGET,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Trains a Linear Regression model to predict ball speed from biomechanics metrics.

    Args:
        df:           DataFrame with feature columns and target column
        feature_cols: list of feature names to use (defaults to canonical 20)
        target_col:   name of the target column
        test_size:    fraction for hold-out test set
        val_size:     fraction for validation set
        random_state: seed for reproducibility

    Returns:
        dict with keys: 'model', 'scaler', 'results', 'feature_names',
                        'coefficients', 'intercept'
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

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    cv_scores = cross_val_score(
        LinearRegression(), scaler.transform(X_temp), y_temp,
        cv=5, scoring="r2"
    )

    results = {
        "train": _evaluate(model, X_train_s, y_train, "train"),
        "val":   _evaluate(model, X_val_s,   y_val,   "val"),
        "test":  _evaluate(model, X_test_s,  y_test,  "test"),
        "cv_r2_mean": round(float(np.mean(cv_scores)), 4),
        "cv_r2_std":  round(float(np.std(cv_scores)), 4),
    }

    coef_df = pd.DataFrame({
        "feature":     feature_cols,
        "coefficient": model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)

    return {
        "model":         model,
        "scaler":        scaler,
        "results":       results,
        "feature_names": feature_cols,
        "coefficients":  coef_df,
        "intercept":     float(model.intercept_),
        "n_train":       len(X_train),
        "n_val":         len(X_val),
        "n_test":        len(X_test),
    }


def _evaluate(model, X, y, split_name: str) -> dict:
    pred = model.predict(X)
    return {
        "r2":   round(float(r2_score(y, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "mae":  round(float(mean_absolute_error(y, pred)), 4),
    }


def predict_ball_speed(
    features: np.ndarray,
    model,
    scaler,
) -> np.ndarray:
    """
    Predicts ball speed for new observations.

    Args:
        features: shape (n_swings, n_features)
        model:    trained LinearRegression
        scaler:   fitted StandardScaler

    Returns:
        1D array of predicted ball speeds in mph
    """
    return model.predict(scaler.transform(features))


def save_model(artifacts: dict, output_dir: str = "outputs/model_artifacts") -> str:
    """Saves model and scaler to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "linear_regression.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": artifacts["model"], "scaler": artifacts["scaler"],
                     "feature_names": artifacts["feature_names"]}, f)
    return path


def load_model(path: str) -> dict:
    """Loads model artifacts from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
