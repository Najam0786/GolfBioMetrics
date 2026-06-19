"""
svm_model.py
Model 5: Support Vector Machine — Binary classification: Efficient vs Inefficient swing.

Rationale: SVM is robust to outliers and works well with smaller, high-dimensional
feature sets. Provides fast binary screening — "Is this swing worth detailed
analysis or does it need fundamental correction first?"
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

from src.features.engineering import get_feature_names


TARGET = "swing_efficient"
EFFICIENCY_THRESHOLD = 0.70


def create_efficiency_labels(
    df: pd.DataFrame,
    source_col: str = "kinematic_sequence_score",
    threshold: float = EFFICIENCY_THRESHOLD,
) -> pd.Series:
    """
    Creates binary efficiency labels from kinematic sequence score.

    Args:
        df:         DataFrame with source column
        source_col: metric to threshold
        threshold:  score >= threshold → 'Efficient' (1), else 'Inefficient' (0)

    Returns:
        Series of integer labels (1 = Efficient, 0 = Inefficient)
    """
    return (df[source_col] >= threshold).astype(int)


def train_svm(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = TARGET,
    label_source_col: str = "kinematic_sequence_score",
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Trains an SVM binary classifier for swing efficiency screening.

    Args:
        df:               DataFrame with features and target
        feature_cols:     feature names (defaults to canonical 20)
        target_col:       binary target column name
        label_source_col: column used to threshold the efficiency label — excluded
                          from features to prevent the model from trivially learning
                          the threshold boundary
        kernel:           SVM kernel ('rbf' is standard for non-linear separation)
        C:                regularisation parameter
        gamma:            kernel coefficient
        test_size:        hold-out test fraction
        val_size:         validation fraction
        random_state:     reproducibility seed

    Returns:
        dict with 'model', 'scaler', 'results', 'feature_names'
    """
    if feature_cols is None:
        feature_cols = [c for c in get_feature_names() if c in df.columns]

    # Exclude the column used to define the binary label — keeping it would let
    # the SVM trivially learn the threshold and inflate accuracy.
    feature_cols = [c for c in feature_cols if c != label_source_col]

    if target_col not in df.columns:
        df = df.copy()
        df[target_col] = create_efficiency_labels(df)

    df_clean = df[feature_cols + [target_col]].dropna()
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values.astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=random_state,
        stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train_s, y_train)

    cv_scores = cross_val_score(
        SVC(kernel=kernel, C=C, gamma=gamma, probability=True,
            random_state=random_state, class_weight="balanced"),
        scaler.transform(X_temp), y_temp, cv=5, scoring="accuracy"
    )

    results = {
        "train": _evaluate_svm(model, X_train_s, y_train),
        "val":   _evaluate_svm(model, X_val_s,   y_val),
        "test":  _evaluate_svm(model, X_test_s,  y_test),
        "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
        "cv_accuracy_std":  round(float(np.std(cv_scores)), 4),
        "efficiency_threshold": EFFICIENCY_THRESHOLD,
    }

    return {
        "model":         model,
        "scaler":        scaler,
        "results":       results,
        "feature_names": feature_cols,
        "class_names":   ["Inefficient", "Efficient"],
        "n_train":       len(X_train),
        "n_val":         len(X_val),
        "n_test":        len(X_test),
    }


def _evaluate_svm(model, X, y) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy":  round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc_roc":   round(float(roc_auc_score(y, prob)), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def screen_swing(
    features: np.ndarray,
    model,
    scaler,
) -> dict:
    """
    Fast binary screen for a single swing.

    Args:
        features: 1D array of feature values (unscaled)
        model:    trained SVC
        scaler:   fitted StandardScaler

    Returns:
        dict with 'label', 'probability_efficient', 'recommend_detailed_analysis'
    """
    x_s = scaler.transform(features.reshape(1, -1))
    pred = int(model.predict(x_s)[0])
    prob = float(model.predict_proba(x_s)[0, 1])

    return {
        "label":                      "Efficient" if pred == 1 else "Inefficient",
        "probability_efficient":      round(prob, 4),
        "recommend_detailed_analysis": pred == 0 or prob < 0.65,
    }


def save_model(artifacts: dict, output_dir: str = "outputs/model_artifacts") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "svm_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": artifacts["model"], "scaler": artifacts["scaler"],
                     "feature_names": artifacts["feature_names"]}, f)
    return path


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
