"""
test_models.py
Smoke tests for all 5 ML models — verify they train and predict without errors.
"""

import numpy as np
import pandas as pd
import pytest


def _make_mock_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Creates a minimal mock feature DataFrame for model training tests."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kinematic_sequence_score":    rng.uniform(0.4, 0.98, n),
        "lag_angle_mid_downswing":     rng.uniform(35, 90, n),
        "lag_angle_impact":            rng.uniform(5, 35, n),
        "xfactor_top_backswing":       rng.uniform(15, 55, n),
        "weight_transfer_timing_ms":   rng.uniform(-200, 100, n),
        "club_path_consistency":        rng.uniform(0.5, 0.99, n),
        "swing_tempo_ratio":            rng.uniform(1.2, 3.5, n),
        "early_cast_severity":          rng.uniform(0, 0.8, n),
        "reverse_pivot_severity":       rng.uniform(0, 0.8, n),
        "sway_severity":                rng.uniform(0, 0.8, n),
        "early_extension_severity":     rng.uniform(0, 0.8, n),
        "over_top_severity":            rng.uniform(0, 0.8, n),
        "sequence_efficiency_index":    rng.uniform(0.3, 0.95, n),
        "power_potential_score":        rng.uniform(0.2, 0.9, n),
        "release_efficiency":           rng.uniform(1.0, 6.0, n),
        "tempo_sequence_alignment":     rng.uniform(0.5, 2.5, n),
        "backswing_duration":           rng.uniform(0.5, 1.2, n),
        "downswing_duration":           rng.uniform(0.2, 0.55, n),
        "xfactor_at_impact":            rng.uniform(10, 40, n),
        "lag_release_rate":             rng.uniform(100, 400, n),
        "ball_speed_mph":               rng.uniform(75, 130, n),
        "carry_distance_yards":         rng.uniform(150, 320, n),
        "injury_risk_score":            rng.uniform(0.0, 1.0, n),
        "skill_level":                  rng.choice(["elite", "semi_pro", "amateur"], n),
    })
    return df


class TestLinearRegression:
    def test_trains_and_predicts(self):
        from src.models.linear_regression import train_linear_regression, predict_ball_speed
        df = _make_mock_df()
        artifacts = train_linear_regression(df, test_size=0.2, val_size=0.1)
        assert "model" in artifacts
        assert artifacts["results"]["test"]["r2"] is not None
        X = df[[c for c in artifacts["feature_names"]]].dropna().values[:5]
        preds = predict_ball_speed(X, artifacts["model"], artifacts["scaler"])
        assert len(preds) == 5
        assert not np.any(np.isnan(preds))

    def test_coefficients_shape(self):
        from src.models.linear_regression import train_linear_regression
        df = _make_mock_df()
        artifacts = train_linear_regression(df, test_size=0.2, val_size=0.1)
        assert len(artifacts["coefficients"]) == len(artifacts["feature_names"])


class TestDecisionTree:
    def test_trains_and_predicts(self):
        from src.models.decision_tree import train_decision_tree, create_quality_labels, predict_quality
        df = _make_mock_df()
        df["swing_quality_class"] = create_quality_labels(df)
        artifacts = train_decision_tree(df, test_size=0.2, val_size=0.1)
        assert "model" in artifacts
        assert "tree_rules" in artifacts
        X = df[[c for c in artifacts["feature_names"]]].dropna().values[:5]
        preds = predict_quality(X, artifacts["model"], artifacts["encoder"])
        assert len(preds) == 5

    def test_class_names_are_strings(self):
        from src.models.decision_tree import train_decision_tree, create_quality_labels
        df = _make_mock_df()
        df["swing_quality_class"] = create_quality_labels(df)
        artifacts = train_decision_tree(df, test_size=0.2, val_size=0.1)
        assert all(isinstance(c, str) for c in artifacts["class_names"])


class TestRandomForest:
    def test_trains_and_predicts(self):
        from src.models.random_forest import train_random_forest, predict_carry_distance
        df = _make_mock_df()
        artifacts = train_random_forest(df, n_estimators=20, test_size=0.2, val_size=0.1)
        assert "model" in artifacts
        assert artifacts["results"]["test"]["r2"] is not None
        X = df[[c for c in artifacts["feature_names"]]].dropna().values[:5]
        preds = predict_carry_distance(X, artifacts["model"], artifacts["scaler"])
        assert len(preds) == 5

    def test_feature_importance_sums_to_one(self):
        from src.models.random_forest import train_random_forest
        df = _make_mock_df()
        artifacts = train_random_forest(df, n_estimators=20, test_size=0.2, val_size=0.1)
        total = artifacts["feature_importance"]["importance"].sum()
        assert abs(total - 1.0) < 0.01


class TestXGBoost:
    def test_trains_and_predicts(self):
        from src.models.xgboost_model import train_xgboost, predict_injury_risk
        df = _make_mock_df()
        artifacts = train_xgboost(df, n_estimators=20, test_size=0.2, val_size=0.1)
        assert "model" in artifacts
        X = df[[c for c in artifacts["feature_names"]]].dropna().values[:5]
        preds = predict_injury_risk(X, artifacts["model"], artifacts["scaler"])
        assert len(preds) == 5
        assert np.all(preds >= 0) and np.all(preds <= 1)


class TestSVM:
    def test_trains_and_screens(self):
        from src.models.svm_model import train_svm, create_efficiency_labels, screen_swing
        df = _make_mock_df()
        df["swing_efficient"] = create_efficiency_labels(df)
        artifacts = train_svm(df, test_size=0.2, val_size=0.1)
        assert "model" in artifacts
        X = df[[c for c in artifacts["feature_names"]]].dropna().iloc[0].values
        result = screen_swing(X, artifacts["model"], artifacts["scaler"])
        assert result["label"] in ("Efficient", "Inefficient")
        assert 0.0 <= result["probability_efficient"] <= 1.0
