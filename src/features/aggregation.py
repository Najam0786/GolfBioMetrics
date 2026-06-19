"""
aggregation.py
Aggregates frame-level keypoint data into swing-level feature vectors.

This module bridges the biomechanics computation layer and the ML model layer.
It reads from the swing frames tensor and calls metrics.py / compensations.py
to produce a single flat feature dict per swing.

Design principle:
  - Receives numpy arrays (no pandas or file IO)
  - Returns flat dict of numeric features, suitable for ML
  - Every feature name matches the schema in MASTER_PLAN.md Section 6.3
"""

import numpy as np

from src.biomechanics.metrics import compute_all_metrics, KP
from src.biomechanics.compensations import (
    detect_early_cast,
    detect_reverse_pivot,
    detect_lateral_sway,
    detect_early_extension,
    detect_over_the_top,
)
from src.biomechanics.signal_processing import (
    detect_swing_phases,
    smooth_signal,
)
from src.biomechanics.confidence import aggregate_swing_confidence


def aggregate_swing(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    fps: float = 60.0,
    golfer_id: int = None,
    swing_id: int = None,
    skill_level: str = None,
) -> dict:
    """
    Computes all biomechanics metrics for a single swing and returns
    a flat feature dict ready for the ML feature matrix.

    Args:
        keypoints_over_time: shape (n_frames, 17, 3) — all keypoints per frame
        timestamps:          shape (n_frames,)
        fps:                 capture frame rate in Hz
        golfer_id:           optional identifier for output dict
        swing_id:            optional identifier for output dict
        skill_level:         optional label for output dict

    Returns:
        flat dict of all computed features and metadata
    """
    n = len(timestamps)
    club_kp = KP["club_head"]

    # ── Detect swing phases ──────────────────────────────────────────────────
    club_z = keypoints_over_time[:, club_kp, 2]
    club_vz = np.gradient(
        smooth_signal(club_z, fps=fps, cutoff_hz=10.0),
        timestamps
    )

    phases = detect_swing_phases(club_vz, timestamps)

    addr_frame  = int(phases["address_end"])
    top_frame   = int(phases["top_of_swing"])
    impact_frame = int(phases["impact"])
    ds_frame    = top_frame

    phase_frames = {
        "address_end":     addr_frame,
        "top_of_swing":    top_frame,
        "downswing_start": ds_frame,
        "impact":          impact_frame,
    }

    # ── Core metric computation ─────────────────────────────────────────────
    metrics = compute_all_metrics(
        keypoints_over_time, timestamps, phase_frames, fps
    )

    # ── Compensation detection ──────────────────────────────────────────────
    from src.biomechanics.signal_processing import compute_angular_velocity_from_keypoints
    from src.biomechanics.metrics import xfactor_series, lag_angle_series

    xf_series  = xfactor_series(keypoints_over_time, timestamps, fps)
    lag_series = lag_angle_series(keypoints_over_time, timestamps, fps)

    hip_angles = xf_series
    hip_av = compute_angular_velocity_from_keypoints(hip_angles, timestamps, fps)

    downswing_lag = lag_series[ds_frame:impact_frame + 1]
    downswing_hip_av = hip_av[ds_frame:impact_frame + 1]
    downswing_ts = timestamps[ds_frame:impact_frame + 1]

    if len(downswing_lag) < 3:
        early_cast = {"early_cast_detected": False, "severity": 0.0, "timing_error_ms": 0.0}
    else:
        early_cast = detect_early_cast(downswing_lag, downswing_hip_av, downswing_ts)

    rev_pivot = detect_reverse_pivot(
        keypoints_over_time, timestamps,
        KP["left_ankle"], KP["right_ankle"], top_frame
    )

    sway = detect_lateral_sway(keypoints_over_time, timestamps, top_frame)

    early_ext = detect_early_extension(
        keypoints_over_time, timestamps,
        KP["left_hip"], KP["right_hip"], ds_frame, impact_frame
    )

    club_positions = keypoints_over_time[:, club_kp, :]
    over_top = detect_over_the_top(club_positions, timestamps, ds_frame)

    # ── Build flat feature dict ─────────────────────────────────────────────
    feature = {}

    if swing_id is not None:
        feature["swing_id"] = swing_id
    if golfer_id is not None:
        feature["golfer_id"] = golfer_id
    if skill_level is not None:
        feature["skill_level"] = skill_level

    feature.update(metrics)

    feature["early_cast_severity"]      = round(float(early_cast["severity"]), 4)
    feature["reverse_pivot_severity"]   = round(float(rev_pivot["severity"]), 4)
    feature["sway_severity"]            = round(float(sway["severity"]), 4)
    feature["early_extension_severity"] = round(float(early_ext["severity"]), 4)
    feature["over_top_severity"]        = round(float(over_top["severity"]), 4)

    feature["early_cast_flag"]      = int(early_cast["early_cast_detected"])
    feature["reverse_pivot_flag"]   = int(rev_pivot["reverse_pivot_detected"])
    feature["sway_flag"]            = int(sway["sway_detected"])
    feature["early_extension_flag"] = int(early_ext["early_extension_detected"])
    feature["over_top_flag"]        = int(over_top["over_top_detected"])

    compensation_count = sum([
        feature["early_cast_flag"],
        feature["reverse_pivot_flag"],
        feature["sway_flag"],
        feature["early_extension_flag"],
        feature["over_top_flag"],
    ])
    feature["compensation_count"] = compensation_count

    confidence_map = {
        "kinematic_sequence": metrics.get("kinematic_sequence_confidence", 0.5),
        "lag":               metrics.get("lag_confidence", 0.5),
        "xfactor":           metrics.get("xfactor_confidence", 0.5),
        "weight_transfer":   metrics.get("weight_transfer_confidence", 0.5),
        "club_path":         metrics.get("club_path_confidence", 0.5),
        "tempo":             metrics.get("tempo_confidence", 0.5),
    }
    feature["overall_confidence"] = round(aggregate_swing_confidence(confidence_map), 4)

    return feature


def aggregate_dataset(
    frames_df,
    fps: float = 60.0,
) -> list:
    """
    Applies aggregate_swing() to every swing in a frames DataFrame.

    Args:
        frames_df: pandas DataFrame with columns
                   [swing_id, frame_id, timestamp_s, golfer_id, skill_level,
                    keypoint, x, y, z, confidence]
        fps: default frame rate (overridden per swing if detectable)

    Returns:
        list of feature dicts, one per swing_id
    """
    from src.biomechanics.metrics import KP as KP_IDX

    KEYPOINT_ORDER = list(KP.keys())
    N_KP = len(KEYPOINT_ORDER)

    results = []

    for swing_id, swing_df in frames_df.groupby("swing_id"):
        pivot = swing_df.pivot_table(
            index=["frame_id", "timestamp_s"],
            columns="keypoint",
            values=["x", "y", "z"],
            aggfunc="first",
        )
        pivot.columns = [f"{v}_{kp}" for v, kp in pivot.columns]
        pivot = pivot.reset_index().sort_values("frame_id")

        n_frames = len(pivot)
        timestamps = pivot["timestamp_s"].values.astype(float)

        kps = np.zeros((n_frames, N_KP, 3))
        for ki, kname in enumerate(KEYPOINT_ORDER):
            for ci, comp in enumerate(["x", "y", "z"]):
                col = f"{comp}_{kname}"
                if col in pivot.columns:
                    kps[:, ki, ci] = pivot[col].values.astype(float)

        swing_meta = swing_df.iloc[0]
        golfer_id  = int(swing_meta["golfer_id"]) if "golfer_id" in swing_df.columns else None
        skill      = str(swing_meta["skill_level"]) if "skill_level" in swing_df.columns else None

        n = n_frames
        dt = np.mean(np.diff(timestamps)) if n > 1 else 1.0 / fps
        actual_fps = 1.0 / (dt + 1e-8)

        features = aggregate_swing(
            kps, timestamps, fps=actual_fps,
            golfer_id=golfer_id, swing_id=swing_id, skill_level=skill
        )
        results.append(features)

    return results
