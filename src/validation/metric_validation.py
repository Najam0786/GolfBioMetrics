"""
metric_validation.py
Statistical validation of biomechanics metrics against skill-level groups.

Validates that metrics:
  1. Discriminate between skill levels (ANOVA / Kruskal-Wallis)
  2. Correlate with performance outcomes (Pearson r)
  3. Meet expected ranges per skill group
"""

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_CORRELATIONS = {
    "kinematic_sequence_score": ("ball_speed_mph",     0.75),
    "xfactor_degrees":          ("ball_speed_mph",     0.65),
    "lag_angle_mid_downswing":  ("ball_speed_mph",     0.60),
    "early_cast_severity":      ("ball_speed_mph",    -0.55),
}

SKILL_RANGES = {
    "kinematic_sequence_score": {
        "elite":    (0.85, 0.98),
        "semi_pro": (0.65, 0.85),
        "amateur":  (0.40, 0.65),
    },
    "xfactor_degrees": {
        "elite":    (40, 55),
        "semi_pro": (30, 45),
        "amateur":  (15, 30),
    },
    "lag_angle_mid_downswing": {
        "elite":    (75, 90),
        "semi_pro": (60, 80),
        "amateur":  (35, 65),
    },
    "swing_tempo_ratio": {
        "elite":    (2.3, 3.0),
        "semi_pro": (2.0, 2.5),
        "amateur":  (1.2, 2.0),
    },
}


def test_discriminant_validity(
    df: pd.DataFrame,
    metric_cols: list = None,
    group_col: str = "skill_level",
    alpha: float = 0.001,
) -> pd.DataFrame:
    """
    Tests whether each metric significantly discriminates between skill groups.

    Uses Kruskal-Wallis H-test (non-parametric, robust to non-normality).

    Args:
        df:          DataFrame with metric columns and group column
        metric_cols: list of metric names to test
        group_col:   column with group labels ('elite', 'semi_pro', 'amateur')
        alpha:       significance threshold

    Returns:
        DataFrame with columns: metric, H_statistic, p_value, significant, effect_size
    """
    if metric_cols is None:
        metric_cols = list(SKILL_RANGES.keys())

    results = []
    valid_groups = df[group_col].isin(["elite", "semi_pro", "amateur"])
    df_valid = df[valid_groups].copy()

    for metric in metric_cols:
        if metric not in df_valid.columns:
            continue

        groups = [
            df_valid[df_valid[group_col] == g][metric].dropna().values
            for g in ["elite", "semi_pro", "amateur"]
            if g in df_valid[group_col].values
        ]

        if len(groups) < 2 or any(len(g) == 0 for g in groups):
            continue

        try:
            h_stat, p_value = stats.kruskal(*groups)
        except ValueError:
            continue

        n_total = sum(len(g) for g in groups)
        eta_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))
        eta_sq = max(0.0, float(eta_sq))

        results.append({
            "metric":      metric,
            "H_statistic": round(float(h_stat), 3),
            "p_value":     float(p_value),
            "significant": bool(p_value < alpha),
            "eta_squared": round(eta_sq, 4),
        })

    return pd.DataFrame(results)


def test_outcome_correlations(
    df: pd.DataFrame,
    expected: dict = None,
) -> pd.DataFrame:
    """
    Tests Pearson correlations between metrics and performance outcomes.

    Args:
        df:       DataFrame with metric and outcome columns
        expected: dict mapping metric_name → (outcome_col, min_r_expected)

    Returns:
        DataFrame with correlation results and pass/fail for expected thresholds
    """
    if expected is None:
        expected = EXPECTED_CORRELATIONS

    results = []

    for metric, (outcome, expected_r) in expected.items():
        if metric not in df.columns or outcome not in df.columns:
            continue

        clean = df[[metric, outcome]].dropna()
        if len(clean) < 10:
            continue

        r, p = stats.pearsonr(clean[metric], clean[outcome])
        meets_threshold = (abs(r) >= abs(expected_r)) if expected_r >= 0 else (r <= expected_r)

        results.append({
            "metric":          metric,
            "outcome":         outcome,
            "pearson_r":       round(float(r), 4),
            "p_value":         float(p),
            "expected_min_r":  expected_r,
            "meets_threshold": bool(meets_threshold),
        })

    return pd.DataFrame(results)


def validate_skill_level_ranges(
    df: pd.DataFrame,
    group_col: str = "skill_level",
) -> pd.DataFrame:
    """
    Checks that median metric values fall within expected ranges per skill group.

    Args:
        df:        DataFrame with metric columns and skill_level column
        group_col: column with skill group labels

    Returns:
        DataFrame with per-metric, per-group median and pass/fail
    """
    rows = []

    for metric, ranges_by_skill in SKILL_RANGES.items():
        if metric not in df.columns:
            continue

        for skill, (lo, hi) in ranges_by_skill.items():
            subset = df[df[group_col] == skill][metric].dropna()
            if len(subset) == 0:
                continue

            median_val = float(np.median(subset))
            mean_val   = float(np.mean(subset))
            in_range   = lo <= median_val <= hi

            rows.append({
                "metric":       metric,
                "skill_level":  skill,
                "median":       round(median_val, 3),
                "mean":         round(mean_val, 3),
                "expected_lo":  lo,
                "expected_hi":  hi,
                "in_range":     in_range,
            })

    return pd.DataFrame(rows)


def run_full_validation(df: pd.DataFrame) -> dict:
    """
    Runs the complete validation suite and prints a summary.

    Args:
        df: swing-level metrics DataFrame

    Returns:
        dict with 'discriminant', 'correlations', 'ranges' DataFrames
    """
    print("=" * 60)
    print("GolfBioMetrics — Metric Validation Report")
    print("=" * 60)

    disc = test_discriminant_validity(df)
    print("\n[1] Discriminant Validity (Kruskal-Wallis)")
    print(disc.to_string(index=False))
    n_sig = disc["significant"].sum() if len(disc) > 0 else 0
    print(f"\n    {n_sig}/{len(disc)} metrics significantly discriminate skill levels (p < 0.001)")

    corr = test_outcome_correlations(df)
    print("\n[2] Outcome Correlations (Pearson r)")
    print(corr.to_string(index=False))
    n_pass = corr["meets_threshold"].sum() if len(corr) > 0 else 0
    print(f"\n    {n_pass}/{len(corr)} correlations meet expected thresholds")

    ranges = validate_skill_level_ranges(df)
    print("\n[3] Skill-Level Range Validation")
    print(ranges.to_string(index=False))
    n_in = ranges["in_range"].sum() if len(ranges) > 0 else 0
    print(f"\n    {n_in}/{len(ranges)} metric-skill combinations within expected ranges")

    print("=" * 60)

    return {
        "discriminant":  disc,
        "correlations":  corr,
        "ranges":        ranges,
    }
