"""
metrics.py
All 7 core biomechanics metrics for the GolfBioMetrics system.

Design principles:
  - Pure geometry and physics — NO machine learning
  - Every public function returns (value: float, confidence: float)
  - np.clip before arccos, atan2 for signed angles, 1e-8 in denominators
  - Separation of concerns: no file IO, no pandas — receives np.ndarray only

Metrics implemented:
  1. kinematic_sequence_score
  2. lag_angle
  3. xfactor
  4. weight_transfer_timing
  5. club_path_consistency
  6. swing_tempo
  7. (compensatory patterns are in compensations.py)

Each metric also has a *_from_swing() convenience wrapper that accepts
the swing's full keypoint tensor and returns (value, confidence).
"""

import numpy as np

from src.biomechanics.geometry_utils import (
    normalize,
    segment_axis_horizontal,
    fit_plane_svd,
    centre_of_mass_approx,
)
from src.biomechanics.signal_processing import (
    smooth_signal,
    rms_deviation_from_plane,
)
from src.biomechanics.confidence import compute_confidence, VALID_RANGES


# ── MediaPipe keypoint indices (subset used) ──────────────────────────────────
KP = {
    "nose":           0,
    "left_shoulder":  1,
    "right_shoulder": 2,
    "left_elbow":     3,
    "right_elbow":    4,
    "left_wrist":     5,
    "right_wrist":    6,
    "left_hip":       7,
    "right_hip":      8,
    "left_knee":      9,
    "right_knee":     10,
    "left_ankle":     11,
    "right_ankle":    12,
    "left_index":     13,
    "right_index":    14,
    "club_grip":      15,
    "club_head":      16,
}


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 1: Kinematic Sequence Quality Score
# ═══════════════════════════════════════════════════════════════════════════════

def kinematic_sequence_score(
    angular_velocities: dict,
    timestamps: np.ndarray,
    downswing_start_idx: int,
    downswing_end_idx: int,
) -> tuple:
    """
    Measures how well the golfer follows the proximal-to-distal energy transfer
    pattern during the downswing: Pelvis → Thorax → Lead Arm → Club.

    Scientific basis:
        Elite golfers peak each segment 20–50ms apart in strict order.
        This multiplies energy at the clubhead like a whip crack.
        (Sheffield Hallam University, TPI Biomechanics, ASMI)

    Args:
        angular_velocities: dict with keys ['pelvis', 'thorax', 'arm', 'club'],
                            each a 1D array of angular velocity (deg/s) over ALL frames
        timestamps:         1D array of time values (seconds) for all frames
        downswing_start_idx: frame index where downswing begins
        downswing_end_idx:   frame index where downswing ends (impact)

    Returns:
        (score: float 0–1, confidence: float 0–1)

    Scoring:
        - +0.70: proportion of correctly ordered pairs out of 3
        - +0.30: timing gaps in optimal 15–60ms range
    """
    required_keys = ["pelvis", "thorax", "arm", "club"]
    for k in required_keys:
        if k not in angular_velocities:
            return 0.0, 0.0

    ds_start = max(0, downswing_start_idx)
    ds_end = min(len(timestamps), downswing_end_idx)

    if ds_end <= ds_start + 2:
        return 0.0, 0.0

    peak_times = {}
    peak_values = {}

    for seg in required_keys:
        av = angular_velocities[seg]
        if len(av) < ds_end:
            return 0.0, 0.0

        window = av[ds_start:ds_end]
        if len(window) == 0:
            return 0.0, 0.0

        local_idx = int(np.argmax(window))
        peak_times[seg]  = float(timestamps[ds_start + local_idx])
        peak_values[seg] = float(window[local_idx])

    pairs = [("pelvis", "thorax"), ("thorax", "arm"), ("arm", "club")]
    correct_order = sum(1 for a, b in pairs if peak_times[a] < peak_times[b])

    timing_gaps_ms = {
        f"{a}_to_{b}": (peak_times[b] - peak_times[a]) * 1000
        for a, b in pairs
    }
    gap_score = float(np.mean([
        1.0 if 15 <= gap <= 60 else 0.5
        for gap in timing_gaps_ms.values()
    ]))

    score = (correct_order / 3.0) * 0.70 + gap_score * 0.30

    all_positive = all(v > 0 for v in peak_values.values())
    confidence = 0.85 if all_positive else 0.55

    return float(np.clip(score, 0.0, 1.0)), float(confidence)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 2: Lag Angle
# ═══════════════════════════════════════════════════════════════════════════════

def lag_angle(
    wrist_pos: np.ndarray,
    elbow_pos: np.ndarray,
    club_head_pos: np.ndarray,
) -> tuple:
    """
    Computes the wrist-cock (lag) angle between the forearm and club shaft.

    Scientific basis:
        The lag angle stores potential energy during downswing.
        Releasing early ('casting') loses 10–25 mph ball speed.
        Elite: 70–90° at mid-downswing, 20–35° at impact.

    Args:
        wrist_pos:    3D position of wrist keypoint (metres)
        elbow_pos:    3D position of elbow keypoint (metres)
        club_head_pos: 3D position of club head keypoint (metres)

    Returns:
        (lag_angle_degrees: float, confidence: float)

    Note:
        np.clip is mandatory before arccos — floating point errors cause NaN.
    """
    if np.any(np.isnan(wrist_pos)) or np.any(np.isnan(elbow_pos)) or np.any(np.isnan(club_head_pos)):
        return 0.0, 0.0

    v_forearm = elbow_pos - wrist_pos
    v_club    = club_head_pos - wrist_pos

    norm_f = np.linalg.norm(v_forearm)
    norm_c = np.linalg.norm(v_club)

    if norm_f < 1e-6 or norm_c < 1e-6:
        return 0.0, 0.0

    v_forearm_n = v_forearm / norm_f
    v_club_n    = v_club / norm_c

    dot = np.clip(np.dot(v_forearm_n, v_club_n), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(dot)))

    confidence = 0.9 if (VALID_RANGES["lag_angle_mid_downswing"][0]
                         <= angle_deg <=
                         VALID_RANGES["lag_angle_mid_downswing"][1]) else 0.4

    return angle_deg, confidence


def lag_angle_series(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    fps: float = 60.0,
) -> np.ndarray:
    """
    Computes lag angle at every frame.

    Args:
        keypoints_over_time: shape (n_frames, n_keypoints, 3)
        timestamps:          shape (n_frames,)
        fps:                 sampling rate

    Returns:
        1D array of lag angles in degrees, length n_frames
    """
    angles = np.zeros(len(timestamps))
    for fi in range(len(timestamps)):
        kps = keypoints_over_time[fi]
        wrist    = kps[KP["right_wrist"]]
        elbow    = kps[KP["right_elbow"]]
        club_hd  = kps[KP["club_head"]]
        angles[fi], _ = lag_angle(wrist, elbow, club_hd)

    return smooth_signal(angles, fps=fps, cutoff_hz=10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 3: Hip-Shoulder Separation (X-Factor)
# ═══════════════════════════════════════════════════════════════════════════════

def xfactor(
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> tuple:
    """
    Computes hip-shoulder separation (X-Factor) in the transverse (horizontal) plane.

    Scientific basis:
        X-Factor at the top of the backswing correlates strongly with clubhead speed.
        Elite average: 40–50°. Amateurs: 15–30°. (Cheetham et al., McTeigue et al.)

    Args:
        left_hip, right_hip:           3D positions of hip keypoints
        left_shoulder, right_shoulder: 3D positions of shoulder keypoints

    Returns:
        (xfactor_degrees: float, confidence: float)

    Note:
        Uses atan2 for signed angle — gives direction and is stable near 0° and 180°.
    """
    pelvis_axis = segment_axis_horizontal(left_hip, right_hip)
    thorax_axis = segment_axis_horizontal(left_shoulder, right_shoulder)

    if np.linalg.norm(pelvis_axis) < 1e-8 or np.linalg.norm(thorax_axis) < 1e-8:
        return 0.0, 0.0

    cross = pelvis_axis[0] * thorax_axis[2] - pelvis_axis[2] * thorax_axis[0]
    dot   = float(np.dot(pelvis_axis, thorax_axis))
    angle_deg = float(np.degrees(np.arctan2(cross, dot)))

    in_range = VALID_RANGES["xfactor_degrees"][0] <= angle_deg <= VALID_RANGES["xfactor_degrees"][1]
    confidence = 0.88 if in_range else 0.35

    return angle_deg, confidence


def xfactor_series(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    fps: float = 60.0,
) -> np.ndarray:
    """
    Computes X-Factor at every frame.

    Returns:
        1D array of X-Factor values in degrees, length n_frames
    """
    angles = np.zeros(len(timestamps))
    for fi in range(len(timestamps)):
        kps = keypoints_over_time[fi]
        angles[fi], _ = xfactor(
            kps[KP["left_hip"]],
            kps[KP["right_hip"]],
            kps[KP["left_shoulder"]],
            kps[KP["right_shoulder"]],
        )
    return smooth_signal(angles, fps=fps, cutoff_hz=6.0)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 4: Weight Transfer Timing
# ═══════════════════════════════════════════════════════════════════════════════

def weight_transfer_timing(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    downswing_start_frame: int,
    left_ankle_idx: int = None,
    right_ankle_idx: int = None,
) -> tuple:
    """
    Measures when the golfer's COM shifts from trail foot to lead foot,
    relative to the start of the downswing.

    Scientific basis:
        Weight should begin transferring slightly BEFORE downswing initiation.
        Optimal: −80 to −120ms (shifts early to enable kinetic chain).
        Late transfer ('hanging back') disrupts sequencing and reduces power.

    Args:
        keypoints_over_time: shape (n_frames, n_keypoints, 3)
        timestamps:          shape (n_frames,)
        downswing_start_frame: frame index where downswing begins
        left_ankle_idx:      index for left (lead) ankle
        right_ankle_idx:     index for right (trail) ankle

    Returns:
        (transfer_timing_ms: float, confidence: float)
        Negative = transfer before downswing start (good), positive = after (bad)
    """
    if left_ankle_idx is None:
        left_ankle_idx = KP["left_ankle"]
    if right_ankle_idx is None:
        right_ankle_idx = KP["right_ankle"]

    n = len(timestamps)
    if n < 4 or downswing_start_frame >= n:
        return 0.0, 0.0

    com_x = np.array([
        float(centre_of_mass_approx(keypoints_over_time[f])[0])
        for f in range(n)
    ])

    back_foot_x  = keypoints_over_time[:, right_ankle_idx, 0]
    front_foot_x = keypoints_over_time[:, left_ankle_idx, 0]
    foot_span    = np.abs(front_foot_x - back_foot_x) + 1e-8

    com_normalised = (com_x - back_foot_x) / foot_span

    transfer_frames = np.where(com_normalised > 0.5)[0]
    if len(transfer_frames) == 0:
        transfer_ms = 300.0
        conf = 0.40
    else:
        transfer_frame = int(transfer_frames[0])
        transfer_ms = (
            timestamps[transfer_frame] - timestamps[downswing_start_frame]
        ) * 1000.0

        in_optimal = -150.0 <= transfer_ms <= -50.0
        conf = 0.85 if in_optimal else 0.65

    return float(transfer_ms), float(conf)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 5: Club Path Consistency
# ═══════════════════════════════════════════════════════════════════════════════

def club_path_consistency(
    club_positions: np.ndarray,
) -> tuple:
    """
    Measures how consistently the club follows a single swing plane.

    Scientific basis:
        Deviation from the swing plane predicts shot direction inconsistency.
        Uses SVD to fit a best-fit plane and measures RMS deviation from it.

    Args:
        club_positions: shape (n_frames, 3) — club head x,y,z during downswing

    Returns:
        (consistency_score: float 0–1, confidence: float 0–1)
        1.0 = perfect plane, decreasing with deviation

    Additionally accessible via return dict in club_path_consistency_full().
    """
    if len(club_positions) < 3:
        return 0.5, 0.3

    plane_normal, centroid = fit_plane_svd(club_positions)
    rms_cm = rms_deviation_from_plane(club_positions, plane_normal, centroid) * 100.0

    consistency = float(np.clip(1.0 - (rms_cm / 10.0), 0.0, 1.0))

    in_range = VALID_RANGES["club_path_consistency"][0] <= consistency <= VALID_RANGES["club_path_consistency"][1]
    confidence = 0.88 if (in_range and len(club_positions) >= 8) else 0.55

    return consistency, confidence


def club_path_consistency_full(
    club_positions: np.ndarray,
) -> dict:
    """
    Extended version returning diagnostic details alongside the score.

    Returns:
        dict with 'consistency_score', 'confidence', 'rms_deviation_cm',
                  'swing_plane_normal'
    """
    consistency, confidence = club_path_consistency(club_positions)

    if len(club_positions) < 3:
        return {
            "consistency_score": consistency,
            "confidence": confidence,
            "rms_deviation_cm": 0.0,
            "swing_plane_normal": np.array([0.0, 1.0, 0.0]),
        }

    plane_normal, centroid = fit_plane_svd(club_positions)
    rms_cm = rms_deviation_from_plane(club_positions, plane_normal, centroid) * 100.0

    return {
        "consistency_score": consistency,
        "confidence": confidence,
        "rms_deviation_cm": round(float(rms_cm), 3),
        "swing_plane_normal": plane_normal,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 6: Swing Tempo & Rhythm
# ═══════════════════════════════════════════════════════════════════════════════

def swing_tempo(
    t_address: float,
    t_transition: float,
    t_impact: float,
) -> tuple:
    """
    Computes the backswing-to-downswing time ratio (tempo ratio).

    Scientific basis:
        Tempo ratio is a fundamental measure of swing rhythm and repeatability.
        Elite average: 2.3–2.8 (backswing ~3× longer than downswing).
        Rushing (< 1.8): downswing starts before kinetic chain is loaded.

    Args:
        t_address:    time (seconds) at start of swing / address position
        t_transition: time (seconds) at top of backswing
        t_impact:     time (seconds) at ball impact

    Returns:
        (tempo_ratio: float, confidence: float)
    """
    backswing_time = t_transition - t_address
    downswing_time = t_impact - t_transition

    if backswing_time <= 0 or downswing_time <= 0:
        return 1.0, 0.0

    tempo_ratio = backswing_time / (downswing_time + 1e-8)

    in_optimal = VALID_RANGES["swing_tempo_ratio"][0] <= tempo_ratio <= VALID_RANGES["swing_tempo_ratio"][1]
    confidence = 0.90 if in_optimal else 0.65

    return float(tempo_ratio), float(confidence)


def swing_tempo_full(
    t_address: float,
    t_transition: float,
    t_impact: float,
) -> dict:
    """
    Extended version returning full diagnostic detail.

    Returns:
        dict with 'tempo_ratio', 'confidence', 'backswing_time_s',
                  'downswing_time_s', 'is_rushing', 'is_optimal', 'is_slow'
    """
    tempo_ratio, confidence = swing_tempo(t_address, t_transition, t_impact)
    backswing_time = t_transition - t_address
    downswing_time = t_impact - t_transition

    return {
        "tempo_ratio":       round(tempo_ratio, 4),
        "confidence":        round(confidence, 4),
        "backswing_time_s":  round(float(backswing_time), 4),
        "downswing_time_s":  round(float(downswing_time), 4),
        "is_rushing":        tempo_ratio < 1.8,
        "is_optimal":        2.0 <= tempo_ratio <= 3.0,
        "is_slow":           tempo_ratio > 3.5,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC 7 (helper): Angular Velocity Profiles for Kinematic Sequence
# ═══════════════════════════════════════════════════════════════════════════════

def compute_segment_angular_velocities(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    fps: float = 60.0,
) -> dict:
    """
    Computes angular velocity profiles for all four kinematic sequence segments:
    pelvis, thorax, lead arm, club.

    Args:
        keypoints_over_time: shape (n_frames, n_keypoints, 3)
        timestamps:          shape (n_frames,)
        fps:                 sampling rate

    Returns:
        dict with keys 'pelvis', 'thorax', 'arm', 'club',
        each a 1D angular velocity array (deg/s)
    """
    from src.biomechanics.signal_processing import compute_angular_velocity_from_keypoints

    n = len(timestamps)

    pelvis_angles = np.zeros(n)
    thorax_angles = np.zeros(n)
    arm_angles    = np.zeros(n)
    club_angles   = np.zeros(n)

    for fi in range(n):
        kps = keypoints_over_time[fi]

        pelvis_ax = segment_axis_horizontal(kps[KP["left_hip"]], kps[KP["right_hip"]])
        ref_z = np.array([0.0, 0.0, -1.0])
        pelvis_angles[fi] = float(np.degrees(np.arctan2(
            pelvis_ax[0] * ref_z[2] - pelvis_ax[2] * ref_z[0],
            np.dot(pelvis_ax, ref_z),
        )))

        thorax_ax = segment_axis_horizontal(kps[KP["left_shoulder"]], kps[KP["right_shoulder"]])
        thorax_angles[fi] = float(np.degrees(np.arctan2(
            thorax_ax[0] * ref_z[2] - thorax_ax[2] * ref_z[0],
            np.dot(thorax_ax, ref_z),
        )))

        arm_vec = kps[KP["right_wrist"]] - kps[KP["right_shoulder"]]
        arm_vec_h = arm_vec.copy()
        arm_vec_h[1] = 0.0
        arm_norm = np.linalg.norm(arm_vec_h)
        if arm_norm > 1e-8:
            arm_vec_h /= arm_norm
            arm_angles[fi] = float(np.degrees(np.arctan2(arm_vec_h[0], arm_vec_h[2] + 1e-8)))

        club_vec = kps[KP["club_head"]] - kps[KP["club_grip"]]
        club_norm = np.linalg.norm(club_vec)
        if club_norm > 1e-8:
            club_vec /= club_norm
            club_angles[fi] = float(np.degrees(np.arctan2(club_vec[0], club_vec[2] + 1e-8)))

    return {
        "pelvis": compute_angular_velocity_from_keypoints(pelvis_angles, timestamps, fps),
        "thorax": compute_angular_velocity_from_keypoints(thorax_angles, timestamps, fps),
        "arm":    compute_angular_velocity_from_keypoints(arm_angles,    timestamps, fps),
        "club":   compute_angular_velocity_from_keypoints(club_angles,   timestamps, fps),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ALL METRICS — single-swing convenience entry point
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    keypoints_over_time: np.ndarray,
    timestamps: np.ndarray,
    phase_frames: dict,
    fps: float = 60.0,
) -> dict:
    """
    Computes all 7 biomechanics metrics for a single swing.

    Args:
        keypoints_over_time: shape (n_frames, n_keypoints, 3)
        timestamps:          shape (n_frames,)
        phase_frames: dict with integer keys:
            'address_end'   — last frame of address
            'top_of_swing'  — frame at top of backswing
            'downswing_start' — same as top_of_swing (or slightly after)
            'impact'        — frame at ball impact
        fps: sampling rate in Hz

    Returns:
        dict with all metric values and confidence scores
    """
    n = len(timestamps)
    addr = min(phase_frames.get("address_end", 0), n - 1)
    top  = min(phase_frames.get("top_of_swing", n // 3), n - 1)
    ds   = min(phase_frames.get("downswing_start", top), n - 1)
    imp  = min(phase_frames.get("impact", int(n * 0.75)), n - 1)

    results = {}

    # ── Metric 1: Kinematic Sequence ─────────────────────────────────────────
    av = compute_segment_angular_velocities(keypoints_over_time, timestamps, fps)
    ks_score, ks_conf = kinematic_sequence_score(av, timestamps, ds, imp)
    results["kinematic_sequence_score"]      = round(float(ks_score), 4)
    results["kinematic_sequence_confidence"] = round(float(ks_conf), 4)

    # ── Metric 2: Lag Angle ───────────────────────────────────────────────────
    lag_angles_all = lag_angle_series(keypoints_over_time, timestamps, fps)

    mid_ds_frame = (ds + imp) // 2
    lag_mid_val, lag_mid_conf = lag_angle(
        keypoints_over_time[mid_ds_frame, KP["right_wrist"]],
        keypoints_over_time[mid_ds_frame, KP["right_elbow"]],
        keypoints_over_time[mid_ds_frame, KP["club_head"]],
    )
    lag_imp_val, lag_imp_conf = lag_angle(
        keypoints_over_time[imp, KP["right_wrist"]],
        keypoints_over_time[imp, KP["right_elbow"]],
        keypoints_over_time[imp, KP["club_head"]],
    )
    results["lag_angle_mid_downswing"] = round(float(lag_mid_val), 2)
    results["lag_angle_impact"]        = round(float(lag_imp_val), 2)
    results["lag_confidence"]          = round(float((lag_mid_conf + lag_imp_conf) / 2), 4)

    # ── Metric 3: X-Factor ───────────────────────────────────────────────────
    top_kps = keypoints_over_time[top]
    xf_val, xf_conf = xfactor(
        top_kps[KP["left_hip"]],
        top_kps[KP["right_hip"]],
        top_kps[KP["left_shoulder"]],
        top_kps[KP["right_shoulder"]],
    )
    imp_kps = keypoints_over_time[imp]
    xf_impact_val, _ = xfactor(
        imp_kps[KP["left_hip"]],
        imp_kps[KP["right_hip"]],
        imp_kps[KP["left_shoulder"]],
        imp_kps[KP["right_shoulder"]],
    )
    results["xfactor_top_backswing"] = round(float(xf_val), 2)
    results["xfactor_at_impact"]     = round(float(xf_impact_val), 2)
    results["xfactor_confidence"]    = round(float(xf_conf), 4)

    # ── Metric 4: Weight Transfer Timing ────────────────────────────────────
    wt_ms, wt_conf = weight_transfer_timing(
        keypoints_over_time, timestamps, ds
    )
    results["weight_transfer_timing_ms"]   = round(float(wt_ms), 2)
    results["weight_transfer_confidence"]  = round(float(wt_conf), 4)
    results["weight_transfer_optimal"]     = bool(-150.0 <= wt_ms <= -50.0)

    # ── Metric 5: Club Path Consistency ─────────────────────────────────────
    club_kp_idx = KP["club_head"]
    club_pos_ds = keypoints_over_time[ds:imp + 1, club_kp_idx, :]
    cp_score, cp_conf = club_path_consistency(club_pos_ds)
    results["club_path_consistency"] = round(float(cp_score), 4)
    results["club_path_confidence"]  = round(float(cp_conf), 4)

    # ── Metric 6: Swing Tempo ────────────────────────────────────────────────
    t_addr = float(timestamps[addr])
    t_top  = float(timestamps[top])
    t_imp  = float(timestamps[imp])
    tempo_ratio, tempo_conf = swing_tempo(t_addr, t_top, t_imp)
    results["swing_tempo_ratio"] = round(float(tempo_ratio), 4)
    results["tempo_confidence"]  = round(float(tempo_conf), 4)
    results["tempo_is_optimal"]  = bool(2.0 <= tempo_ratio <= 3.0)
    results["backswing_duration"]  = round(float(t_top - t_addr), 4)
    results["downswing_duration"]  = round(float(t_imp - t_top), 4)

    # ── Metric 7: Lag Release Rate (derived) ────────────────────────────────
    lag_mid_ds = float(lag_angles_all[mid_ds_frame])
    lag_at_imp = float(lag_angles_all[imp])
    lag_release_rate = (lag_mid_ds - lag_at_imp) / (
        (timestamps[imp] - timestamps[mid_ds_frame]) + 1e-8
    )
    results["lag_release_rate"] = round(float(lag_release_rate), 2)

    return results
