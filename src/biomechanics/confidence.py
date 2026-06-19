"""
confidence.py
Composite confidence scoring for all GolfBioMetrics biomechanics metrics.

Design principle:
  - Every metric returns a (value, confidence) tuple
  - Confidence < 0.6  → flag as unreliable, suppress from user output
  - Confidence 0.6–0.8 → display with warning
  - Confidence > 0.8  → display as reliable
"""

import numpy as np


def compute_confidence(
    keypoint_confidences: np.ndarray,
    missing_frame_fraction: float,
    value: float,
    valid_range: tuple,
    temporal_jitter: float,
) -> float:
    """
    Composite confidence score (0.0–1.0) for any single metric value.

    Sub-scores:
      - Keypoint quality (0.35 weight): mean pose estimation confidence
      - Data completeness (0.30 weight): fraction of non-missing frames
      - Physiological validity (0.20 weight): is value in expected range?
      - Temporal stability (0.15 weight): low jitter = more reliable

    Args:
        keypoint_confidences:   1D array of keypoint confidence values (0–1)
        missing_frame_fraction: fraction of frames that were missing/interpolated
        value:                  the computed metric value (in native units)
        valid_range:            (min, max) physiologically plausible range
        temporal_jitter:        std-dev of metric across adjacent frames

    Returns:
        composite confidence in [0.0, 1.0]

    Hard rule:
        If value is outside valid_range, confidence is capped at 0.3.
    """
    kp_score  = float(np.mean(keypoint_confidences)) if len(keypoint_confidences) > 0 else 0.0
    complete  = max(0.0, 1.0 - missing_frame_fraction)
    range_ok  = 1.0 if valid_range[0] <= value <= valid_range[1] else 0.0
    range_width = valid_range[1] - valid_range[0]
    stability = max(0.0, 1.0 - (temporal_jitter / (range_width + 1e-8)))

    confidence = (
        0.35 * kp_score
        + 0.30 * complete
        + 0.20 * range_ok
        + 0.15 * stability
    )

    if range_ok == 0.0:
        confidence = min(confidence, 0.3)

    return float(np.clip(confidence, 0.0, 1.0))


def confidence_from_keypoints(
    keypoint_confidences: np.ndarray,
    missing_frame_fraction: float = 0.0,
) -> float:
    """
    Simplified confidence when no range-check or jitter data is available.
    Used for intermediary steps where full compute_confidence is not yet feasible.

    Args:
        keypoint_confidences:   1D array of confidence values
        missing_frame_fraction: fraction of frames missing

    Returns:
        confidence in [0.0, 1.0]
    """
    kp_score = float(np.mean(keypoint_confidences)) if len(keypoint_confidences) > 0 else 0.0
    complete = max(0.0, 1.0 - missing_frame_fraction)
    return float(np.clip(0.60 * kp_score + 0.40 * complete, 0.0, 1.0))


def classify_confidence(confidence: float) -> str:
    """
    Maps a numeric confidence value to a display classification.

    Args:
        confidence: float in [0.0, 1.0]

    Returns:
        'reliable'   if confidence >= 0.80
        'uncertain'  if 0.60 <= confidence < 0.80
        'unreliable' if confidence < 0.60
    """
    if confidence >= 0.80:
        return "reliable"
    elif confidence >= 0.60:
        return "uncertain"
    else:
        return "unreliable"


def aggregate_swing_confidence(metric_confidences: dict) -> float:
    """
    Computes an overall swing-level reliability score from all metric confidences.

    Args:
        metric_confidences: dict mapping metric name → confidence float

    Returns:
        mean confidence across all metrics
    """
    values = list(metric_confidences.values())
    if not values:
        return 0.0
    return float(np.mean(values))


# ── Physiological valid ranges (used by compute_confidence) ──────────────────

VALID_RANGES = {
    "kinematic_sequence_score":  (0.0,   1.0),
    "lag_angle_mid_downswing":   (10.0,  120.0),   # degrees
    "lag_angle_impact":          (0.0,   60.0),    # degrees
    "xfactor_degrees":           (-10.0, 70.0),    # degrees
    "weight_transfer_timing_ms": (-300.0, 300.0),  # milliseconds
    "club_path_consistency":     (0.0,   1.0),
    "swing_tempo_ratio":         (0.5,   6.0),
    "early_cast_severity":       (0.0,   1.0),
    "reverse_pivot_severity":    (0.0,   1.0),
    "sway_severity":             (0.0,   1.0),
    "early_extension_severity":  (0.0,   1.0),
    "over_top_severity":         (0.0,   1.0),
    "injury_risk_score":         (0.0,   1.0),
}
