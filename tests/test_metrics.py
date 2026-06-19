"""
test_metrics.py
Unit tests for src/biomechanics/metrics.py

Tests use synthetic ground-truth inputs so expected values can be
verified analytically. See MASTER_PLAN.md Section 9.1 for test criteria.
"""

import numpy as np
import pytest
from src.biomechanics.metrics import (
    lag_angle,
    xfactor,
    swing_tempo,
    swing_tempo_full,
    club_path_consistency,
    weight_transfer_timing,
    kinematic_sequence_score,
    KP,
)


# ── Helper: build minimal keypoints tensor ────────────────────────────────────

def make_static_keypoints(n_frames: int = 30, n_kp: int = 17) -> np.ndarray:
    """Creates a stationary keypoints array — all keypoints at origin."""
    return np.zeros((n_frames, n_kp, 3))


def make_keypoints_with_positions(positions: dict, n_frames: int = 30,
                                  n_kp: int = 17) -> np.ndarray:
    """Creates a uniform keypoints array with specific positions per keypoint index."""
    kps = np.zeros((n_frames, n_kp, 3))
    for idx, pos in positions.items():
        kps[:, idx, :] = np.array(pos)
    return kps


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 2: Lag Angle
# ═══════════════════════════════════════════════════════════════════════════════

class TestLagAngle:

    def test_90_degree_known_angle(self):
        wrist     = np.array([0.0, 0.0, 0.0])
        elbow     = np.array([0.0, 0.4, 0.0])   # forearm: +Y
        club_head = np.array([0.0, 0.0, -0.8])  # club:    -Z → 90° to forearm
        angle, conf = lag_angle(wrist, elbow, club_head)
        assert abs(angle - 90.0) < 0.001, f"Expected 90°, got {angle}"

    def test_180_degree_straight_wrist(self):
        wrist     = np.array([0.0, 0.0, 0.0])
        elbow     = np.array([0.0, 0.4, 0.0])
        club_head = np.array([0.0, -0.8, 0.0])  # opposite direction = 180°
        angle, conf = lag_angle(wrist, elbow, club_head)
        assert abs(angle - 180.0) < 0.001

    def test_0_degree_fully_extended(self):
        wrist     = np.array([0.0, 0.0, 0.0])
        elbow     = np.array([0.0, 0.4, 0.0])
        club_head = np.array([0.0, 0.8, 0.0])  # same direction = 0°
        angle, conf = lag_angle(wrist, elbow, club_head)
        assert abs(angle - 0.0) < 0.001

    def test_45_degree_angle(self):
        wrist     = np.array([0.0, 0.0, 0.0])
        elbow     = np.array([1.0, 0.0, 0.0])
        club_head = np.array([1.0, 1.0, 0.0])  # 45° from forearm
        angle, conf = lag_angle(wrist, elbow, club_head)
        assert abs(angle - 45.0) < 0.01

    def test_zero_vectors_no_nan_no_crash(self):
        wrist     = np.array([0.0, 0.0, 0.0])
        elbow     = np.array([0.0, 0.0, 0.0])  # zero-length forearm
        club_head = np.array([0.0, 0.0, -0.8])
        angle, conf = lag_angle(wrist, elbow, club_head)
        assert not np.isnan(angle)
        assert conf == 0.0  # should flag as unreliable

    def test_returns_tuple_float_float(self):
        wrist     = np.array([0.0, 0.0, 0.0])
        elbow     = np.array([0.0, 0.4, 0.0])
        club_head = np.array([0.0, 0.0, -0.8])
        result = lag_angle(wrist, elbow, club_head)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 3: X-Factor
# ═══════════════════════════════════════════════════════════════════════════════

class TestXFactor:

    def test_zero_xfactor_aligned_segments(self):
        left_hip      = np.array([-0.12, 1.0, 0.0])
        right_hip     = np.array([ 0.12, 1.0, 0.0])
        left_shoulder = np.array([-0.22, 1.5, 0.0])
        right_shoulder = np.array([0.22, 1.5, 0.0])
        angle, conf = xfactor(left_hip, right_hip, left_shoulder, right_shoulder)
        assert abs(angle) < 1.0, f"Aligned segments should give ~0°, got {angle}"

    def test_known_45_degree_xfactor(self):
        left_hip      = np.array([-0.12, 1.0, 0.0])
        right_hip     = np.array([ 0.12, 1.0, 0.0])
        angle_rad = np.radians(45)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        left_shoulder  = np.array([-c * 0.22, 1.5, s * 0.22])
        right_shoulder = np.array([ c * 0.22, 1.5, -s * 0.22])
        angle, conf = xfactor(left_hip, right_hip, left_shoulder, right_shoulder)
        assert abs(abs(angle) - 45.0) < 2.0, f"Expected ~±45°, got {angle}"

    def test_returns_tuple(self):
        lh = np.array([-0.12, 1.0, 0.0])
        rh = np.array([ 0.12, 1.0, 0.0])
        ls = np.array([-0.22, 1.5, 0.0])
        rs = np.array([ 0.22, 1.5, 0.0])
        result = xfactor(lh, rh, ls, rs)
        assert len(result) == 2

    def test_confidence_in_0_1(self):
        lh = np.array([-0.12, 1.0, 0.0])
        rh = np.array([ 0.12, 1.0, 0.0])
        ls = np.array([-0.22, 1.5, 0.0])
        rs = np.array([ 0.22, 1.5, 0.0])
        _, conf = xfactor(lh, rh, ls, rs)
        assert 0.0 <= conf <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 6: Swing Tempo
# ═══════════════════════════════════════════════════════════════════════════════

class TestSwingTempo:

    def test_elite_tempo_ratio_2_5(self):
        t_address    = 0.0
        t_transition = 0.625   # backswing 0.625s
        t_impact     = 0.875   # downswing 0.25s → ratio = 2.5
        ratio, conf = swing_tempo(t_address, t_transition, t_impact)
        assert abs(ratio - 2.5) < 0.01
        assert conf >= 0.85

    def test_rushing_amateur_ratio_1_5(self):
        t_address    = 0.0
        t_transition = 0.75    # backswing 0.75s
        t_transition = 0.6
        t_impact     = 1.0     # downswing 0.4s → ratio = 1.5
        ratio, conf = swing_tempo(t_address, t_transition, t_impact)
        assert ratio < 2.0

    def test_negative_time_handled(self):
        ratio, conf = swing_tempo(0.5, 0.3, 0.8)
        assert conf == 0.0

    def test_full_dict_contains_expected_keys(self):
        result = swing_tempo_full(0.0, 0.6, 0.85)
        for key in ["tempo_ratio", "confidence", "backswing_time_s",
                    "downswing_time_s", "is_rushing", "is_optimal"]:
            assert key in result

    def test_optimal_flag_correct(self):
        result = swing_tempo_full(0.0, 0.625, 0.875)  # ratio = 2.5
        assert result["is_optimal"] is True
        assert result["is_rushing"] is False

    def test_rush_flag_correct(self):
        result = swing_tempo_full(0.0, 0.3, 0.5)  # ratio = 1.5
        assert result["is_rushing"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 5: Club Path Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestClubPathConsistency:

    def test_perfect_plane_score_near_1(self):
        t = np.linspace(0, 1, 20)
        perfect_plane = np.column_stack([t, t * 0.5, np.zeros(20)])
        score, conf = club_path_consistency(perfect_plane)
        assert score >= 0.95, f"Perfect plane should score >= 0.95, got {score}"

    def test_scattered_points_lower_score(self):
        rng = np.random.default_rng(1)
        noisy = rng.uniform(-1, 1, (20, 3))
        score, conf = club_path_consistency(noisy)
        assert score < 0.9

    def test_fewer_than_3_points_no_crash(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        score, conf = club_path_consistency(pts)
        assert not np.isnan(score)
        assert conf <= 0.5

    def test_score_in_0_1(self):
        pts = np.random.randn(15, 3)
        score, conf = club_path_consistency(pts)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= conf <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 1: Kinematic Sequence Score
# ═══════════════════════════════════════════════════════════════════════════════

class TestKinematicSequenceScore:

    def _make_av(self, t_peaks: dict, timestamps: np.ndarray) -> dict:
        """Helper: build synthetic angular velocity curves peaking at given times."""
        av = {}
        sigma = 0.04
        for seg, t_peak in t_peaks.items():
            av[seg] = 1000.0 * np.exp(-((timestamps - t_peak) ** 2) / (2 * sigma ** 2))
        return av

    def test_perfect_elite_sequence_high_score(self):
        t = np.linspace(0.0, 0.3, 120)
        t_peaks = {"pelvis": 0.060, "thorax": 0.090, "arm": 0.130, "club": 0.200}
        av = self._make_av(t_peaks, t)
        score, conf = kinematic_sequence_score(av, t, 0, 120)
        assert score >= 0.85, f"Elite sequence should score >= 0.85, got {score}"

    def test_reversed_sequence_low_score(self):
        t = np.linspace(0.0, 0.3, 120)
        t_peaks = {"pelvis": 0.200, "thorax": 0.150, "arm": 0.100, "club": 0.060}
        av = self._make_av(t_peaks, t)
        score, conf = kinematic_sequence_score(av, t, 0, 120)
        assert score <= 0.40, f"Reversed sequence should score <= 0.40, got {score}"

    def test_missing_key_returns_zero(self):
        t = np.linspace(0.0, 0.3, 60)
        av = {"pelvis": np.ones(60), "thorax": np.ones(60)}
        score, conf = kinematic_sequence_score(av, t, 0, 60)
        assert score == 0.0
        assert conf == 0.0

    def test_score_in_0_1(self):
        t = np.linspace(0.0, 0.3, 120)
        t_peaks = {"pelvis": 0.08, "thorax": 0.10, "arm": 0.14, "club": 0.20}
        av = {seg: np.exp(-((t - tp) ** 2) / 0.002) for seg, tp in t_peaks.items()}
        score, _ = kinematic_sequence_score(av, t, 0, 120)
        assert 0.0 <= score <= 1.0

    def test_empty_downswing_window_returns_zero(self):
        t = np.linspace(0.0, 0.3, 120)
        av = {k: np.ones(120) for k in ["pelvis", "thorax", "arm", "club"]}
        score, conf = kinematic_sequence_score(av, t, 50, 50)
        assert score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# NaN safety tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNaNSafety:

    def test_lag_angle_nan_inputs_no_nan_output(self):
        w = np.array([np.nan, 0.0, 0.0])
        e = np.array([0.0, 0.4, 0.0])
        c = np.array([0.0, 0.0, -0.8])
        angle, conf = lag_angle(w, e, c)
        assert not np.isnan(angle)

    def test_xfactor_zero_hips_no_crash(self):
        lh = np.array([0.0, 0.0, 0.0])
        rh = np.array([0.0, 0.0, 0.0])
        ls = np.array([-0.22, 1.5, 0.0])
        rs = np.array([ 0.22, 1.5, 0.0])
        angle, conf = xfactor(lh, rh, ls, rs)
        assert not np.isnan(angle)
        assert conf == 0.0
