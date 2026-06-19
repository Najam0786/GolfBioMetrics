"""
signal_processing.py
Signal processing utilities for golf swing time-series data.

Includes: smoothing, event detection, gap interpolation, peak finding.
All functions are pure — no side effects, no IO.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def smooth_signal(
    signal: np.ndarray,
    fps: float = 60.0,
    cutoff_hz: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.

    No phase distortion — critical for kinematic timing analysis.

    Args:
        signal:     1D time-series array
        fps:        sampling rate in Hz
        cutoff_hz:  cut-off frequency (6Hz for gross body, 10Hz for fast limbs)
        order:      filter order (4 is standard for biomechanics)

    Returns:
        filtered signal, same shape as input

    Note:
        Returns raw signal if too short to filter (< 3 × filter order).
    """
    min_length = 3 * order
    if len(signal) < min_length:
        return signal.copy()

    nyquist = fps / 2.0
    normal_cutoff = cutoff_hz / nyquist

    if normal_cutoff >= 1.0:
        return signal.copy()

    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def find_peak_in_window(
    signal: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> tuple:
    """
    Finds the index and value of the maximum in a time window.

    Args:
        signal:    1D array
        start_idx: window start index (inclusive)
        end_idx:   window end index (exclusive)

    Returns:
        (peak_value, peak_global_index) or (None, None) if window is empty
    """
    if start_idx >= end_idx or end_idx > len(signal):
        return None, None

    window = signal[start_idx:end_idx]
    if len(window) == 0:
        return None, None

    local_idx = int(np.argmax(window))
    return float(window[local_idx]), start_idx + local_idx


def detect_zero_crossing(
    signal: np.ndarray,
    direction: str = "falling",
) -> np.ndarray:
    """
    Finds indices where signal crosses zero.

    Args:
        signal:    1D array
        direction: 'falling' (positive→negative), 'rising' (negative→positive),
                   or 'both'

    Returns:
        array of crossing indices
    """
    sign_changes = np.diff(np.sign(signal))

    if direction == "falling":
        return np.where(sign_changes < 0)[0]
    elif direction == "rising":
        return np.where(sign_changes > 0)[0]
    else:
        return np.where(sign_changes != 0)[0]


def detect_swing_phases(
    club_velocity_z: np.ndarray,
    timestamps: np.ndarray,
    static_threshold_ms: float = 0.5,
) -> dict:
    """
    Detects golf swing phase boundaries from club Z-axis velocity.

    Phase detection thresholds (from SKILLS.md):
      - Address:         club velocity < 0.5 m/s for > 0.5s
      - Backswing start: velocity > 0.5 m/s moving away from ball
      - Top of swing:    velocity direction reverses (zero crossing)
      - Downswing start: same frame as top of swing end
      - Impact:          peak club velocity frame
      - Follow-through:  post-impact deceleration

    Args:
        club_velocity_z: 1D velocity of club head in Z direction (m/s)
        timestamps:      corresponding time array (seconds)
        static_threshold_ms: speed below which club is considered static (m/s)

    Returns:
        dict with keys: 'address_end', 'top_of_swing', 'impact', each as frame index
    """
    speed = np.abs(club_velocity_z)

    # Address end: first frame where speed exceeds threshold
    moving_frames = np.where(speed > static_threshold_ms)[0]
    address_end = int(moving_frames[0]) if len(moving_frames) > 0 else 0

    # Top of swing: first zero crossing of club_velocity_z after address
    post_address = club_velocity_z[address_end:]
    crossings = detect_zero_crossing(post_address, direction="falling")
    if len(crossings) > 0:
        top_of_swing = address_end + int(crossings[0])
    else:
        top_of_swing = address_end + len(post_address) // 2

    # Impact: peak speed after top_of_swing
    post_top = speed[top_of_swing:]
    if len(post_top) > 0:
        impact_local = int(np.argmax(post_top))
        impact = top_of_swing + impact_local
    else:
        impact = top_of_swing + 1

    return {
        "address_end":   address_end,
        "top_of_swing":  top_of_swing,
        "impact":        impact,
    }


def interpolate_missing(
    signal: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.3,
    max_gap: int = 5,
) -> tuple:
    """
    Linearly interpolates frames where confidence is below threshold.

    Short gaps (≤ max_gap frames) are interpolated.
    Long gaps are left as-is but flagged in the mask.

    Args:
        signal:     1D signal array
        confidence: 1D confidence array (0–1), same length
        threshold:  confidence below this value is treated as missing
        max_gap:    maximum consecutive frames to interpolate

    Returns:
        (interpolated_signal, was_interpolated_mask)
    """
    interpolated = signal.copy().astype(float)
    was_interpolated = np.zeros(len(signal), dtype=bool)

    low_conf = confidence < threshold
    i = 0

    while i < len(signal):
        if low_conf[i]:
            start = i
            while i < len(signal) and low_conf[i]:
                i += 1
            end = i
            gap_length = end - start

            if gap_length <= max_gap and start > 0 and end < len(signal):
                fill = np.linspace(signal[start - 1], signal[end], gap_length + 2)[1:-1]
                interpolated[start:end] = fill

            was_interpolated[start:end] = True
        else:
            i += 1

    return interpolated, was_interpolated


def compute_angular_velocity_from_keypoints(
    angle_series: np.ndarray,
    timestamps: np.ndarray,
    fps: float = 60.0,
) -> np.ndarray:
    """
    Computes smoothed angular velocity (deg/s) from a joint angle time series.

    Pipeline: smooth → differentiate → smooth again.

    Args:
        angle_series: 1D array of angles in degrees
        timestamps:   1D array of time in seconds
        fps:          sampling rate in Hz

    Returns:
        angular velocity array in degrees/second
    """
    smoothed = smooth_signal(angle_series, fps=fps, cutoff_hz=6.0)
    velocity = np.gradient(smoothed, timestamps)
    velocity_smooth = smooth_signal(velocity, fps=fps, cutoff_hz=10.0)
    return velocity_smooth


def rms_deviation_from_plane(
    points: np.ndarray,
    plane_normal: np.ndarray,
    plane_centroid: np.ndarray,
) -> float:
    """
    Computes the RMS distance of a set of 3D points from a fitted plane.

    Args:
        points:         shape (N, 3)
        plane_normal:   unit normal of the plane
        plane_centroid: centroid of the plane

    Returns:
        RMS deviation in metres
    """
    centred = points - plane_centroid
    distances = np.abs(centred @ plane_normal)
    return float(np.sqrt(np.mean(distances ** 2)))
