"""
noise_injection.py
Adds realistic sensor/pose-estimation noise to synthetic keypoint data.

Used to simulate:
  - Markerless pose estimation jitter
  - Occlusion during fast downswing
  - Camera frame-rate drop
  - Extreme body-type scaling artefacts
"""

import numpy as np
import pandas as pd


def add_gaussian_jitter(
    keypoints: np.ndarray,
    sigma_metres: float = 0.008,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Adds zero-mean Gaussian jitter to all keypoints.

    Args:
        keypoints: shape (n_frames, n_keypoints, 3)
        sigma_metres: std-dev of noise in metres (0.008 ≈ 8mm, typical markerless)
        rng: numpy random Generator for reproducibility

    Returns:
        noisy keypoints same shape as input
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma_metres, keypoints.shape)
    return keypoints + noise


def inject_occlusion(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    occlude_prob: float = 0.10,
    max_gap_frames: int = 8,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Randomly zeros out keypoint positions and sets confidence to near-zero,
    simulating occlusion during fast downswing or limb crossing.

    Args:
        keypoints:   shape (n_frames, n_keypoints, 3)
        confidences: shape (n_frames, n_keypoints)
        occlude_prob: probability that any given keypoint gets an occlusion event
        max_gap_frames: maximum consecutive frames to occlude

    Returns:
        (keypoints_out, confidences_out) — same shapes, with occlusion applied
    """
    if rng is None:
        rng = np.random.default_rng()

    kp_out = keypoints.copy()
    conf_out = confidences.copy()
    n_frames, n_kp, _ = keypoints.shape

    for ki in range(n_kp):
        if rng.random() < occlude_prob:
            gap = int(rng.integers(2, max_gap_frames + 1))
            start = int(rng.integers(0, max(1, n_frames - gap)))
            end = min(n_frames, start + gap)
            kp_out[start:end, ki, :] = 0.0
            conf_out[start:end, ki] = rng.uniform(0.0, 0.15, end - start)

    return kp_out, conf_out


def add_temporal_dropout(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    dropout_prob: float = 0.05,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Randomly drops entire frames (sets all keypoints to zero and confidence to 0).
    Simulates frame-rate drops or missed detections.

    Args:
        keypoints:   shape (n_frames, n_keypoints, 3)
        confidences: shape (n_frames, n_keypoints)
        dropout_prob: probability of dropping each frame

    Returns:
        (keypoints_out, confidences_out)
    """
    if rng is None:
        rng = np.random.default_rng()

    kp_out = keypoints.copy()
    conf_out = confidences.copy()
    n_frames = keypoints.shape[0]

    drop_mask = rng.random(n_frames) < dropout_prob
    kp_out[drop_mask] = 0.0
    conf_out[drop_mask] = 0.0

    return kp_out, conf_out


def scale_for_body_type(
    keypoints: np.ndarray,
    height_m: float = 1.80,
    wingspan_ratio: float = 1.02,
) -> np.ndarray:
    """
    Scales keypoints to match a specific body type.

    Args:
        keypoints:       shape (n_frames, n_keypoints, 3)
        height_m:        golfer height in metres (default 1.80)
        wingspan_ratio:  arm length relative to standard (1.0 = average)

    Returns:
        scaled keypoints
    """
    height_scale = height_m / 1.80
    arm_indices = [3, 4, 5, 6, 13, 14]  # elbows, wrists, index fingers

    scaled = keypoints * height_scale
    scaled[:, arm_indices, :] *= wingspan_ratio

    return scaled


def apply_all_noise(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    skill_level: str,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Applies the full realistic noise pipeline appropriate to the skill level's
    capture quality.

    Args:
        keypoints:   shape (n_frames, n_keypoints, 3)
        confidences: shape (n_frames, n_keypoints)
        skill_level: 'elite', 'semi_pro', or 'amateur'

    Returns:
        (keypoints_noisy, confidences_noisy)
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_params = {
        "elite":    {"sigma": 0.004, "occlude_prob": 0.03, "dropout_prob": 0.01},
        "semi_pro": {"sigma": 0.008, "occlude_prob": 0.07, "dropout_prob": 0.03},
        "amateur":  {"sigma": 0.015, "occlude_prob": 0.12, "dropout_prob": 0.06},
    }
    p = noise_params.get(skill_level, noise_params["amateur"])

    kp = add_gaussian_jitter(keypoints, sigma_metres=p["sigma"], rng=rng)
    kp, conf = inject_occlusion(kp, confidences.copy(),
                                occlude_prob=p["occlude_prob"], rng=rng)
    kp, conf = add_temporal_dropout(kp, conf,
                                    dropout_prob=p["dropout_prob"], rng=rng)

    return kp, conf
