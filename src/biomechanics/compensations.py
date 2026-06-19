"""
compensations.py
Detection of biomechanically inefficient compensatory movement patterns.

Each detector returns a dict with:
  - '<pattern>_detected': bool
  - 'severity': float 0.0–1.0 (0 = absent, 1 = severe)
  - 'timing_error_ms' or other diagnostic float (where applicable)

All functions are pure: no side effects, no IO.
"""

import numpy as np
from src.biomechanics.geometry_utils import centre_of_mass_approx


def detect_early_cast(
    lag_angles: np.ndarray,
    hip_angular_velocity: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """
    Early arm cast: lag angle decreases BEFORE pelvis reaches peak velocity.

    Research threshold: lag release starts >10° drop before hip peak.

    Args:
        lag_angles:            1D array of lag angle (degrees) during downswing
        hip_angular_velocity:  1D array of pelvis angular velocity (deg/s)
        timestamps:            1D time array (seconds), same length

    Returns:
        dict with 'early_cast_detected', 'severity' (0–1), 'timing_error_ms'
    """
    if len(lag_angles) < 3 or len(hip_angular_velocity) < 3:
        return {"early_cast_detected": False, "severity": 0.0, "timing_error_ms": 0.0}

    hip_peak_idx = int(np.argmax(hip_angular_velocity))
    hip_peak_time = float(timestamps[hip_peak_idx])

    lag_diff = np.diff(lag_angles)
    release_indices = np.where(lag_diff < -2.0)[0]

    if len(release_indices) == 0:
        return {"early_cast_detected": False, "severity": 0.0, "timing_error_ms": 0.0}

    lag_release_idx = int(release_indices[0])
    lag_release_time = float(timestamps[lag_release_idx])

    is_early = lag_release_time < hip_peak_time
    timing_error_s = hip_peak_time - lag_release_time
    severity = float(np.clip(timing_error_s / 0.10, 0.0, 1.0)) if is_early else 0.0

    return {
        "early_cast_detected": bool(is_early),
        "severity": round(severity, 4),
        "timing_error_ms": round(timing_error_s * 1000, 2),
    }


def detect_reverse_pivot(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    left_ankle_idx: int = 11,
    right_ankle_idx: int = 12,
    top_of_swing_frame: int = None,
) -> dict:
    """
    Reverse pivot: golfer's weight remains on the lead (front) foot at the top
    of the backswing instead of loading the trail foot.

    Threshold: normalised COM_x > 0.55 (front-foot side) at top of backswing.

    Args:
        keypoints_over_time: shape (n_frames, n_keypoints, 3)
        timestamps:          shape (n_frames,)
        left_ankle_idx:      keypoint index for left ankle (lead foot)
        right_ankle_idx:     keypoint index for right ankle (trail foot)
        top_of_swing_frame:  frame index for top of backswing

    Returns:
        dict with 'reverse_pivot_detected', 'severity', 'com_position_normalised'
    """
    if top_of_swing_frame is None:
        top_of_swing_frame = len(timestamps) // 3

    top_frame = min(top_of_swing_frame, len(keypoints_over_time) - 1)
    kps = keypoints_over_time[top_frame]

    com = centre_of_mass_approx(kps)
    back_foot_x  = float(kps[right_ankle_idx, 0])
    front_foot_x = float(kps[left_ankle_idx, 0])

    foot_span = abs(front_foot_x - back_foot_x)
    if foot_span < 1e-6:
        return {
            "reverse_pivot_detected": False,
            "severity": 0.0,
            "com_position_normalised": 0.5,
        }

    com_normalised = (com[0] - back_foot_x) / (foot_span + 1e-8)

    THRESHOLD = 0.55
    is_reverse = float(com_normalised) > THRESHOLD
    severity = float(np.clip((com_normalised - THRESHOLD) / (1.0 - THRESHOLD), 0.0, 1.0)) if is_reverse else 0.0

    return {
        "reverse_pivot_detected": bool(is_reverse),
        "severity": round(float(severity), 4),
        "com_position_normalised": round(float(com_normalised), 4),
    }


def detect_lateral_sway(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    backswing_end_frame: int = None,
) -> dict:
    """
    Lateral sway: COM moves > 15 cm sideways during the backswing.

    Args:
        keypoints_over_time: shape (n_frames, n_keypoints, 3)
        timestamps:          shape (n_frames,)
        backswing_end_frame: frame index for top of backswing

    Returns:
        dict with 'sway_detected', 'severity', 'max_lateral_displacement_m'
    """
    if backswing_end_frame is None:
        backswing_end_frame = len(timestamps) // 3

    end_frame = min(backswing_end_frame, len(keypoints_over_time) - 1)

    com_x = np.array([
        float(centre_of_mass_approx(keypoints_over_time[f])[0])
        for f in range(end_frame + 1)
    ])

    if len(com_x) < 2:
        return {"sway_detected": False, "severity": 0.0, "max_lateral_displacement_m": 0.0}

    max_displacement = float(np.max(np.abs(com_x - com_x[0])))

    THRESHOLD_M = 0.15
    is_sway = max_displacement > THRESHOLD_M
    severity = float(np.clip((max_displacement - THRESHOLD_M) / 0.15, 0.0, 1.0)) if is_sway else 0.0

    return {
        "sway_detected": bool(is_sway),
        "severity": round(severity, 4),
        "max_lateral_displacement_m": round(max_displacement, 4),
    }


def detect_early_extension(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    left_hip_idx: int = 7,
    right_hip_idx: int = 8,
    downswing_start_frame: int = None,
    impact_frame: int = None,
) -> dict:
    """
    Early extension: hips thrust toward the ball (decrease Z distance to target)
    during the downswing. Indicates spine angle loss.

    Args:
        keypoints_over_time: shape (n_frames, n_keypoints, 3)
        timestamps:          shape (n_frames,)
        left_hip_idx:        keypoint index for left hip
        right_hip_idx:       keypoint index for right hip
        downswing_start_frame: frame index for start of downswing
        impact_frame:        frame index for impact

    Returns:
        dict with 'early_extension_detected', 'severity', 'hip_forward_displacement_m'
    """
    n = len(timestamps)
    if downswing_start_frame is None:
        downswing_start_frame = n // 3
    if impact_frame is None:
        impact_frame = int(n * 0.75)

    ds_start = min(downswing_start_frame, n - 1)
    imp = min(impact_frame, n - 1)

    mid_hip_z_start = float(
        np.mean([keypoints_over_time[ds_start, left_hip_idx, 2],
                 keypoints_over_time[ds_start, right_hip_idx, 2]])
    )
    mid_hip_z_impact = float(
        np.mean([keypoints_over_time[imp, left_hip_idx, 2],
                 keypoints_over_time[imp, right_hip_idx, 2]])
    )

    # Golfer faces -Z; hips moving in +Z direction = thrusting forward
    hip_forward_disp = mid_hip_z_impact - mid_hip_z_start

    THRESHOLD_M = 0.06
    is_ee = hip_forward_disp > THRESHOLD_M
    severity = float(np.clip((hip_forward_disp - THRESHOLD_M) / 0.10, 0.0, 1.0)) if is_ee else 0.0

    return {
        "early_extension_detected": bool(is_ee),
        "severity": round(severity, 4),
        "hip_forward_displacement_m": round(float(hip_forward_disp), 4),
    }


def detect_over_the_top(
    club_positions: np.ndarray,
    timestamps: np.ndarray,
    downswing_start_frame: int = None,
) -> dict:
    """
    Over-the-top: club path angle at mid-downswing crosses outside-in
    (positive-to-negative X trajectory when facing -Z).

    Args:
        club_positions:       shape (n_frames, 3) — club head x,y,z
        timestamps:           shape (n_frames,)
        downswing_start_frame: start frame for downswing analysis

    Returns:
        dict with 'over_top_detected', 'severity', 'path_angle_deg'
    """
    n = len(timestamps)
    if downswing_start_frame is None:
        downswing_start_frame = n // 3

    ds_start = min(downswing_start_frame, n - 2)
    ds_end = min(n - 1, ds_start + max(2, (n - ds_start) // 2))

    club_ds = club_positions[ds_start:ds_end]
    if len(club_ds) < 2:
        return {"over_top_detected": False, "severity": 0.0, "path_angle_deg": 0.0}

    delta_x = float(club_ds[-1, 0] - club_ds[0, 0])
    delta_z = float(club_ds[-1, 2] - club_ds[0, 2])

    path_angle = float(np.degrees(np.arctan2(delta_x, abs(delta_z) + 1e-8)))

    THRESHOLD_DEG = 3.0
    is_ott = path_angle > THRESHOLD_DEG
    severity = float(np.clip((path_angle - THRESHOLD_DEG) / 15.0, 0.0, 1.0)) if is_ott else 0.0

    return {
        "over_top_detected": bool(is_ott),
        "severity": round(severity, 4),
        "path_angle_deg": round(path_angle, 2),
    }
