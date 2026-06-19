"""
decision_tree.py
Model 2: Decision Tree Classifier — Classify swing quality (Poor/Average/Good/Elite).

Rationale: Decision rules are directly interpretable by coaches.
The tree can be printed as a human-readable flowchart.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

from src.features.engineering import get_feature_names


TARGET = "swing_quality_class"

QUALITY_BINS = [0.0, 0.55, 0.70, 0.85, 1.01]
QUALITY_LABELS = ["Poor", "Average", "Good", "Elite"]


def create_quality_labels(df: pd.DataFrame, source_col: str = "kinematic_sequence_score") -> pd.Series:
    """
    Derives swing quality class from kinematic sequence score.

    Bins:
        0.00–0.55: Poor
        0.55–0.70: Average
        0.70–0.85: Good
        0.85–1.00: Elite

    Args:
        df:         DataFrame containing source_col
        source_col: column to bin (defaults to kinematic_sequence_score)

    Returns:
        Series of string labels
    """
    return pd.cut(
        df[source_col],
        bins=QUALITY_BINS,
        labels=QUALITY_LABELS,
        include_lowest=True,
    ).astype(str)


def train_decision_tree(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = TARGET,
    label_source_col: str = "kinematic_sequence_score",
    max_depth: int = 8,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Trains a Decision Tree classifier for swing quality classification.

    Args:
        df:               DataFrame with features and target column
        feature_cols:     feature names (defaults to canonical 20)
        target_col:       name of the target column
        label_source_col: column used to derive the quality label — excluded
                          from features to prevent trivial circular accuracy
        max_depth:        maximum tree depth (8 keeps rules interpretable)
        test_size:        test fraction
        val_size:         validation fraction
        random_state:     reproducibility seed

    Returns:
        dict with 'model', 'encoder', 'results', 'feature_names',
                  'class_names', 'tree_rules'
    """
    if feature_cols is None:
        feature_cols = [c for c in get_feature_names() if c in df.columns]

    # Exclude the column used to bin the label — keeping it would let the tree
    # trivially recover the bin boundaries and report 100% accuracy.
    feature_cols = [c for c in feature_cols if c != label_source_col]

    if target_col not in df.columns:
        df = df.copy()
        df[target_col] = create_quality_labels(df)

    df_clean = df[feature_cols + [target_col]].dropna()
    X = df_clean[feature_cols].values
    y_raw = df_clean[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=random_state,
        stratify=y_temp
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(
        DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=10,
            random_state=random_state, class_weight="balanced",
        ),
        X_temp, y_temp, cv=10, scoring="accuracy"
    )

    results = {
        "train": _evaluate_clf(model, X_train, y_train),
        "val":   _evaluate_clf(model, X_val,   y_val),
        "test":  _evaluate_clf(model, X_test,  y_test),
        "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
        "cv_accuracy_std":  round(float(np.std(cv_scores)), 4),
    }

    tree_rules = export_text(model, feature_names=feature_cols, max_depth=max_depth)

    feature_importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model":              model,
        "encoder":            le,
        "results":            results,
        "feature_names":      feature_cols,
        "class_names":        list(le.classes_),
        "tree_rules":         tree_rules,
        "feature_importance": feature_importance,
        "n_train":            len(X_train),
        "n_val":              len(X_val),
        "n_test":             len(X_test),
    }


def _evaluate_clf(model, X, y) -> dict:
    pred = model.predict(X)
    avg = "weighted"
    return {
        "accuracy":  round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, average=avg, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, pred, average=avg, zero_division=0)), 4),
        "f1":        round(float(f1_score(y, pred, average=avg, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def predict_quality(
    features: np.ndarray,
    model,
    encoder,
) -> np.ndarray:
    """
    Predicts swing quality class labels.

    Returns:
        1D array of string labels ('Poor', 'Average', 'Good', 'Elite')
    """
    numeric_preds = model.predict(features)
    return encoder.inverse_transform(numeric_preds)


def save_model(artifacts: dict, output_dir: str = "outputs/model_artifacts") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "decision_tree.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "model":         artifacts["model"],
            "encoder":       artifacts["encoder"],
            "feature_names": artifacts["feature_names"],
            "class_names":   artifacts["class_names"],
        }, f)
    return path


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
