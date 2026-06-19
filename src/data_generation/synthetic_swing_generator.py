"""
synthetic_swing_generator.py
Generates synthetic golf swing biomechanics data for the GolfBioMetrics POC.

Outputs:
    data/synthetic/golf_swing_frames.csv   — frame-level keypoints
    data/synthetic/golf_swing_metrics.csv  — swing-level aggregated metrics + labels
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Population parameters ────────────────────────────────────────────────────
SKILL_CONFIGS = {
    "elite": {
        "n_golfers": 15,
        "swings_per_golfer": 10,
        "clubhead_speed_range": (105, 120),   # mph
        "xfactor_range": (40, 55),             # degrees at top of backswing
        "sequence_quality_range": (0.85, 0.98),
        "tempo_ratio_range": (2.3, 3.0),
        "lag_mid_range": (75, 90),             # degrees mid-downswing
        "lag_impact_range": (20, 35),          # degrees at impact
        "compensation_prob": 0.05,
        "fps": 60,
        "backswing_duration": (0.5, 0.7),      # seconds
        "downswing_duration": (0.2, 0.25),
    },
    "semi_pro": {
        "n_golfers": 15,
        "swings_per_golfer": 10,
        "clubhead_speed_range": (90, 105),
        "xfactor_range": (30, 45),
        "sequence_quality_range": (0.65, 0.85),
        "tempo_ratio_range": (2.0, 2.5),
        "lag_mid_range": (60, 80),
        "lag_impact_range": (14, 26),
        "compensation_prob": 0.25,
        "fps": 60,
        "backswing_duration": (0.6, 0.9),
        "downswing_duration": (0.25, 0.35),
    },
    "amateur": {
        "n_golfers": 15,
        "swings_per_golfer": 10,
        "clubhead_speed_range": (70, 90),
        "xfactor_range": (15, 30),
        "sequence_quality_range": (0.40, 0.65),
        "tempo_ratio_range": (1.2, 2.0),
        "lag_mid_range": (35, 65),
        "lag_impact_range": (5, 15),
        "compensation_prob": 0.65,
        "fps": 30,
        "backswing_duration": (0.8, 1.2),
        "downswing_duration": (0.35, 0.55),
    },
}

# MediaPipe keypoint indices (subset used for golf)
KEYPOINT_NAMES = [
    "nose", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_index", "right_index",
    "club_grip", "club_head",
]

# Body proportions (metres, approximate adult male)
_BASE_POSE = {
    "nose":            np.array([ 0.00,  1.75,  0.05]),
    "left_shoulder":   np.array([-0.22,  1.52,  0.00]),
    "right_shoulder":  np.array([ 0.22,  1.52,  0.00]),
    "left_elbow":      np.array([-0.40,  1.20,  0.00]),
    "right_elbow":     np.array([ 0.40,  1.20,  0.00]),
    "left_wrist":      np.array([-0.50,  0.95,  0.00]),
    "right_wrist":     np.array([ 0.50,  0.95,  0.00]),
    "left_hip":        np.array([-0.12,  1.00,  0.00]),
    "right_hip":       np.array([ 0.12,  1.00,  0.00]),
    "left_knee":       np.array([-0.12,  0.54,  0.00]),
    "right_knee":      np.array([ 0.12,  0.54,  0.00]),
    "left_ankle":      np.array([-0.16,  0.08,  0.00]),
    "right_ankle":     np.array([ 0.16,  0.08,  0.00]),
    "left_index":      np.array([-0.54,  0.90,  0.00]),
    "right_index":     np.array([ 0.54,  0.90,  0.00]),
    "club_grip":       np.array([ 0.00,  0.88,  0.05]),
    "club_head":       np.array([ 0.00,  0.30, -0.80]),
}


def _rotation_y(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix around Y (vertical) axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def _rotation_z(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix around Z (anterior-posterior) axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _smooth_curve(t: np.ndarray, t_peak: float, sigma: float) -> np.ndarray:
    """Bell-shaped smooth curve peaking at t_peak with width sigma."""
    return np.exp(-((t - t_peak) ** 2) / (2 * sigma ** 2))


def generate_swing_keypoints(
    cfg: dict,
    rng: np.random.Generator,
    compensation_flags: dict,
) -> np.ndarray:
    """
    Generates frame-level 3D keypoints for a single golf swing.

    Returns:
        keypoints: np.ndarray shape (n_frames, 17, 3)
        timestamps: np.ndarray shape (n_frames,)
    """
    fps = cfg["fps"]
    bs_dur = rng.uniform(*cfg["backswing_duration"])
    ds_dur = rng.uniform(*cfg["downswing_duration"])
    ft_dur = 0.6  # follow-through is roughly fixed

    total_dur = bs_dur + ds_dur + ft_dur
    n_frames = max(20, int(total_dur * fps))
    timestamps = np.linspace(0.0, total_dur, n_frames)

    t_top = bs_dur
    t_impact = bs_dur + ds_dur

    xfactor = rng.uniform(*cfg["xfactor_range"])
    sequence_quality = rng.uniform(*cfg["sequence_quality_range"])
    lag_mid = rng.uniform(*cfg["lag_mid_range"])
    lag_impact = rng.uniform(*cfg["lag_impact_range"])

    keypoints = np.zeros((n_frames, 17, 3))

    for fi, t in enumerate(timestamps):
        pose = {k: v.copy() for k, v in _BASE_POSE.items()}

        # ── Phase: backswing ─────────────────────────────────────────────
        if t <= t_top:
            phase_frac = t / (t_top + 1e-8)
            # Hip rotation (limited during backswing)
            hip_rot = xfactor * 0.3 * phase_frac
            # Shoulder rotation (creates X-Factor coil)
            shoulder_rot = xfactor * phase_frac
            # Weight shift: COM moves slightly to trail side
            lat_shift = 0.05 * phase_frac * (1.0 if not compensation_flags.get("reverse_pivot") else -1.0)

            hip_rot_mat = _rotation_y(hip_rot)
            sho_rot_mat = _rotation_y(shoulder_rot)

            for kp in ["left_hip", "right_hip"]:
                pose[kp] = hip_rot_mat @ pose[kp]
            for kp in ["left_shoulder", "right_shoulder", "left_elbow",
                        "right_elbow", "left_wrist", "right_wrist",
                        "left_index", "right_index"]:
                pose[kp] = sho_rot_mat @ pose[kp]

            for kp in pose:
                pose[kp][0] += lat_shift

            # Lag builds during backswing — wrists cock
            lag_frac = phase_frac
            club_angle_offset = lag_mid * lag_frac
            grip = pose["club_grip"]
            base_club = _BASE_POSE["club_head"] - _BASE_POSE["club_grip"]
            rot = _rotation_z(club_angle_offset * 0.5)
            pose["club_head"] = grip + rot @ base_club

        # ── Phase: downswing ─────────────────────────────────────────────
        elif t <= t_impact:
            ds_frac = (t - t_top) / (ds_dur + 1e-8)

            # Kinematic sequence — pelvis leads by timing gap
            # sequence_quality controls how correctly sequenced the timing is
            gap_factor = 0.3 * sequence_quality  # 0.12s gap (elite) to ~0
            t_pelvis_peak = t_top + ds_dur * 0.30
            t_thorax_peak = t_pelvis_peak + gap_factor * ds_dur * 0.35
            t_arm_peak    = t_thorax_peak + gap_factor * ds_dur * 0.35

            sigma = 0.08 * ds_dur
            pelvis_frac = _smooth_curve(np.array([t]), t_pelvis_peak, sigma)[0]
            thorax_frac = _smooth_curve(np.array([t]), t_thorax_peak, sigma)[0]

            hip_rot_back = xfactor * (1.0 - ds_frac) * 0.3
            hip_rot_through = -xfactor * 0.6 * ds_frac
            shoulder_rot_back = xfactor * (1.0 - ds_frac)

            hip_total = hip_rot_back + hip_rot_through
            hip_rot_mat = _rotation_y(hip_total * pelvis_frac)
            sho_rot_mat = _rotation_y(shoulder_rot_back * thorax_frac * 0.7)

            for kp in ["left_hip", "right_hip"]:
                pose[kp] = hip_rot_mat @ pose[kp]
            for kp in ["left_shoulder", "right_shoulder", "left_elbow",
                        "right_elbow", "left_wrist", "right_wrist",
                        "left_index", "right_index"]:
                pose[kp] = sho_rot_mat @ pose[kp]

            # Lag angle — elite maintains until impact, amateurs release early
            # early_cast: lag releases before hip peak
            early_cast_applied = compensation_flags.get("early_cast", False)
            if early_cast_applied:
                release_frac = ds_frac  # lag decays linearly (cast)
            else:
                release_frac = max(0.0, ds_frac - 0.6) / 0.4  # held until late

            current_lag = lag_mid - (lag_mid - lag_impact) * release_frac
            grip = pose["club_grip"]
            base_club = _BASE_POSE["club_head"] - _BASE_POSE["club_grip"]
            rot = _rotation_z(current_lag * 0.5)
            pose["club_head"] = grip + rot @ base_club

            # Sway compensation
            if compensation_flags.get("sway", False):
                pose_kps = list(pose.keys())
                for kp in pose_kps:
                    pose[kp][0] += 0.12 * np.sin(ds_frac * np.pi)

            # Early extension: hips thrust forward
            if compensation_flags.get("early_extension", False):
                for kp in ["left_hip", "right_hip"]:
                    pose[kp][2] -= 0.08 * ds_frac

        # ── Phase: follow-through ─────────────────────────────────────────
        else:
            ft_frac = min(1.0, (t - t_impact) / (ft_dur + 1e-8))
            full_rot_y = _rotation_y(-xfactor * 0.8 * ft_frac)
            for kp in ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                        "left_index", "right_index"]:
                pose[kp] = full_rot_y @ pose[kp]
            grip = pose["club_grip"]
            base_club = _BASE_POSE["club_head"] - _BASE_POSE["club_grip"]
            rot = _rotation_z(-20 * ft_frac)
            pose["club_head"] = grip + rot @ base_club

        # Assemble frame row
        for ki, kname in enumerate(KEYPOINT_NAMES):
            keypoints[fi, ki, :] = pose[kname]

    return keypoints, timestamps


def sample_compensations(cfg: dict, rng: np.random.Generator) -> dict:
    """Sample which compensation patterns are present for this swing."""
    p = cfg["compensation_prob"]
    return {
        "early_cast":      rng.random() < p,
        "reverse_pivot":   rng.random() < p * 0.6,
        "sway":            rng.random() < p * 0.7,
        "early_extension": rng.random() < p * 0.5,
        "over_top":        rng.random() < p * 0.4,
    }


def compute_swing_outcomes(cfg: dict, comp: dict, rng: np.random.Generator,
                           seq_quality: float, lag_mid: float, xfactor: float) -> dict:
    """Compute synthetic ground-truth outcome labels."""
    clubhead_speed = rng.uniform(*cfg["clubhead_speed_range"])
    comp_count = sum(comp.values())
    comp_severity = comp_count / 5.0

    ball_speed = (
        0.82 * clubhead_speed
        + 0.15 * seq_quality * 10
        + 0.03 * xfactor
        - 0.05 * comp_count * 5
        + rng.normal(0, 2.0)
    )
    ball_speed = max(60.0, ball_speed)

    carry_distance = ball_speed * 1.67 + rng.normal(0, 3.0)
    carry_distance = max(100.0, carry_distance)

    offline_yards = (
        2.0 + (1.0 - seq_quality) * 15
        + comp_count * 3
        + rng.normal(0, 1.5)
    )
    offline_yards = max(0.0, offline_yards)

    injury_risk = min(1.0, max(0.0,
        0.30 * float(comp.get("reverse_pivot", False)) * (0.5 + rng.random() * 0.5)
        + 0.25 * float(comp.get("early_extension", False)) * (0.5 + rng.random() * 0.5)
        + 0.25 * float(comp.get("sway", False)) * (0.5 + rng.random() * 0.5)
        + 0.20 * float(comp.get("early_cast", False)) * (0.3 + rng.random() * 0.4)
    ))

    return {
        "clubhead_speed_mph":   round(float(clubhead_speed), 2),
        "ball_speed_mph":       round(float(ball_speed), 2),
        "carry_distance_yards": round(float(carry_distance), 2),
        "offline_yards":        round(float(offline_yards), 2),
        "injury_risk_score":    round(float(injury_risk), 4),
        "compensation_count":   comp_count,
        "compensation_severity": round(float(comp_severity), 4),
    }


def generate_dataset(output_dir: str = "data/synthetic") -> tuple:
    """
    Generate the full 500-swing synthetic dataset.
    Returns (frames_df, metrics_df).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    frames_rows = []
    metrics_rows = []

    swing_id = 0
    golfer_id = 0

    for skill, cfg in SKILL_CONFIGS.items():
        for g in range(cfg["n_golfers"]):
            golfer_id += 1
            
            # Age and experience: realistic distributions by skill level
            if skill == "elite":
                # Elite: typically 22-35 years old, 15-25 years of experience (started young)
                golfer_age = rng.integers(22, 36)
                years_experience = rng.integers(15, 26)
            elif skill == "semi_pro":
                # Semi-pro: typically 25-45 years old, 10-20 years of experience
                golfer_age = rng.integers(25, 46)
                years_experience = rng.integers(10, 21)
            else:  # amateur
                # Amateur: wider age range 25-70, 1-30 years of experience
                golfer_age = rng.integers(25, 71)
                years_experience = rng.integers(1, 31)
            
            # Gender: golf participation 70% male, 30% female
            gender = rng.choice(['M', 'F'], p=[0.70, 0.30])
            
            # Height varies by gender
            if gender == 'M':
                golfer_height = rng.uniform(1.70, 1.95)  # Male: 5'7" to 6'5"
            else:
                golfer_height = rng.uniform(1.58, 1.78)  # Female: 5'2" to 5'10"
            height_scale = golfer_height / 1.80
            
            # Fitness level (1-10): correlates with skill level
            if skill == "elite":
                fitness_level = rng.integers(7, 11)  # 7-10
            elif skill == "semi_pro":
                fitness_level = rng.integers(5, 9)   # 5-8
            else:  # amateur
                fitness_level = rng.integers(2, 8)   # 2-7 (wide range)
            
            # Dominant hand: 90% right-handed in golf
            dominant_hand = rng.choice(['right', 'left'], p=[0.90, 0.10])
            
            # Environmental conditions (realistic ranges)
            # Temperature: varies by season and time of day (8°C to 35°C)
            temperature_c = rng.uniform(8, 35)
            
            # Wind: 0-25 mph typical, occasionally higher
            wind_speed_mph = rng.uniform(0, 25)
            wind_direction_deg = rng.uniform(0, 360)  # 0=N, 90=E, 180=S, 270=W
            
            # Humidity: 30-90%
            humidity_pct = rng.uniform(30, 90)
            
            # Elevation: varies by course location (sea level to 2000m)
            elevation_m = rng.choice([
                rng.uniform(0, 100),      # Sea level courses (60%)
                rng.uniform(100, 500),    # Low elevation (25%)
                rng.uniform(500, 1500),   # Mountain courses (12%)
                rng.uniform(1500, 2500)   # High altitude (3%)
            ], p=[0.60, 0.25, 0.12, 0.03])
            
            # Time of day: morning/afternoon/evening affects muscle temp
            hour_of_day = rng.choice([
                rng.uniform(6, 10),   # Early morning (15%)
                rng.uniform(10, 14),  # Midday (40%)
                rng.uniform(14, 18),  # Afternoon (35%)
                rng.uniform(18, 20)   # Evening (10%)
            ], p=[0.15, 0.40, 0.35, 0.10])
            
            # Course type: affects wind exposure and playing conditions
            course_type = rng.choice(['parkland', 'links', 'desert', 'mountain'], 
                                     p=[0.50, 0.30, 0.15, 0.05])
            
            # Green speed (Stimp meter): 8-12 feet
            green_speed_stimp = rng.uniform(8, 12)

            for s in range(cfg["swings_per_golfer"]):
                swing_id += 1

                comp = sample_compensations(cfg, rng)
                seq_quality = rng.uniform(*cfg["sequence_quality_range"])
                xfactor = rng.uniform(*cfg["xfactor_range"])
                lag_mid = rng.uniform(*cfg["lag_mid_range"])
                lag_impact = rng.uniform(*cfg["lag_impact_range"])
                tempo_ratio = rng.uniform(*cfg["tempo_ratio_range"])

                cfg_swing = dict(cfg)
                keypoints, timestamps = generate_swing_keypoints(cfg_swing, rng, comp)

                keypoints *= height_scale
                noise_scale = 0.004 if skill == "elite" else 0.008 if skill == "semi_pro" else 0.015
                keypoints += rng.normal(0, noise_scale, keypoints.shape)

                conf_base = 0.95 if skill == "elite" else 0.88 if skill == "semi_pro" else 0.80
                confidences = np.clip(
                    rng.normal(conf_base, 0.05, (len(timestamps), len(KEYPOINT_NAMES))), 0.3, 1.0
                )

                for fi, (t, kp_conf) in enumerate(zip(timestamps, confidences)):
                    for ki, kname in enumerate(KEYPOINT_NAMES):
                        frames_rows.append({
                            "swing_id":    swing_id,
                            "frame_id":    fi,
                            "timestamp_s": round(float(t), 5),
                            "golfer_id":   golfer_id,
                            "skill_level": skill,
                            "keypoint":    kname,
                            "x":           round(float(keypoints[fi, ki, 0]), 5),
                            "y":           round(float(keypoints[fi, ki, 1]), 5),
                            "z":           round(float(keypoints[fi, ki, 2]), 5),
                            "confidence":  round(float(kp_conf[ki]), 4),
                        })

                outcomes = compute_swing_outcomes(cfg, comp, rng, seq_quality, lag_mid, xfactor)

                club_path_consistency = rng.uniform(
                    0.90 if skill == "elite" else 0.78 if skill == "semi_pro" else 0.60,
                    0.99 if skill == "elite" else 0.92 if skill == "semi_pro" else 0.82
                )

                bs_dur = rng.uniform(*cfg["backswing_duration"])
                ds_dur = rng.uniform(*cfg["downswing_duration"])
                tempo_ratio_actual = bs_dur / (ds_dur + 1e-8)
                weight_transfer_ms = rng.uniform(-150, -50) if skill in ("elite", "semi_pro") else rng.uniform(-250, 50)

                metrics_rows.append({
                    "swing_id":                    swing_id,
                    "golfer_id":                   golfer_id,
                    "skill_level":                 skill,
                    # Demographics
                    "age":                         int(golfer_age),
                    "years_experience":            int(years_experience),
                    "gender":                      gender,
                    "height_m":                    round(float(golfer_height), 2),
                    "fitness_level":               int(fitness_level),
                    "dominant_hand":               dominant_hand,
                    # Environmental conditions
                    "temperature_c":               round(float(temperature_c), 1),
                    "wind_speed_mph":              round(float(wind_speed_mph), 1),
                    "wind_direction_deg":          round(float(wind_direction_deg), 1),
                    "humidity_pct":                round(float(humidity_pct), 1),
                    "elevation_m":                 round(float(elevation_m), 1),
                    "hour_of_day":                 round(float(hour_of_day), 1),
                    "course_type":                 course_type,
                    "green_speed_stimp":           round(float(green_speed_stimp), 2),
                    "club_type":                   "driver",
                    "kinematic_sequence_score":    round(float(seq_quality), 4),
                    "kinematic_sequence_confidence": round(float(conf_base), 4),
                    "lag_angle_mid_downswing":     round(float(lag_mid), 2),
                    "lag_angle_impact":            round(float(lag_impact), 2),
                    "lag_confidence":              round(float(conf_base - 0.03), 4),
                    "xfactor_degrees":             round(float(xfactor), 2),
                    "xfactor_confidence":          round(float(conf_base), 4),
                    "weight_transfer_timing_ms":   round(float(weight_transfer_ms), 2),
                    "weight_transfer_confidence":  round(float(conf_base - 0.05), 4),
                    "club_path_consistency":       round(float(club_path_consistency), 4),
                    "club_path_confidence":        round(float(conf_base - 0.02), 4),
                    "swing_tempo_ratio":           round(float(tempo_ratio_actual), 4),
                    "tempo_confidence":            round(float(conf_base), 4),
                    "early_cast_flag":             int(comp["early_cast"]),
                    "reverse_pivot_flag":          int(comp["reverse_pivot"]),
                    "sway_flag":                   int(comp["sway"]),
                    "early_extension_flag":        int(comp["early_extension"]),
                    "over_top_flag":               int(comp["over_top"]),
                    **outcomes,
                })

    # ── Edge cases (50 swings) ────────────────────────────────────────────────
    for _ in range(50):
        swing_id += 1
        golfer_id += 1
        edge_type = rng.choice(["noise", "missing", "extreme_body", "injury"])

        if edge_type in ("noise", "missing"):
            skill = "amateur"
            cfg = SKILL_CONFIGS["amateur"]
        elif edge_type == "extreme_body":
            skill = rng.choice(["elite", "semi_pro", "amateur"])
            cfg = SKILL_CONFIGS[skill]
        else:
            skill = "amateur"
            cfg = SKILL_CONFIGS["amateur"]

        comp = {k: True for k in ["early_cast", "reverse_pivot", "sway", "early_extension", "over_top"]}
        seq_quality = rng.uniform(0.30, 0.55)
        xfactor = rng.uniform(10, 25)
        lag_mid = rng.uniform(20, 45)
        lag_impact = rng.uniform(3, 12)

        # Random demographics for edge cases
        edge_age = rng.integers(30, 66)
        edge_exp = rng.integers(5, 21)
        edge_gender = rng.choice(['M', 'F'], p=[0.70, 0.30])
        edge_height = rng.uniform(1.65, 1.90) if edge_gender == 'M' else rng.uniform(1.58, 1.75)
        edge_fitness = rng.integers(3, 8)
        edge_hand = rng.choice(['right', 'left'], p=[0.90, 0.10])
        
        # Random environmental for edge cases
        edge_temp = rng.uniform(5, 30)
        edge_wind = rng.uniform(0, 20)
        edge_wind_dir = rng.uniform(0, 360)
        edge_humidity = rng.uniform(30, 85)
        edge_elev = rng.uniform(0, 1000)
        edge_hour = rng.uniform(8, 18)
        edge_course = rng.choice(['parkland', 'links', 'desert', 'mountain'])
        edge_green_speed = rng.uniform(8, 11)
        
        metrics_rows.append({
            "swing_id":                    swing_id,
            "golfer_id":                   golfer_id,
            "skill_level":                 skill + "_edge",
            # Demographics
            "age":                         int(edge_age),
            "years_experience":            int(edge_exp),
            "gender":                      edge_gender,
            "height_m":                    round(float(edge_height), 2),
            "fitness_level":               int(edge_fitness),
            "dominant_hand":               edge_hand,
            # Environmental
            "temperature_c":               round(float(edge_temp), 1),
            "wind_speed_mph":              round(float(edge_wind), 1),
            "wind_direction_deg":          round(float(edge_wind_dir), 1),
            "humidity_pct":                round(float(edge_humidity), 1),
            "elevation_m":                 round(float(edge_elev), 1),
            "hour_of_day":                 round(float(edge_hour), 1),
            "course_type":                 edge_course,
            "green_speed_stimp":           round(float(edge_green_speed), 2),
            "club_type":                   "driver",
            "kinematic_sequence_score":    round(float(seq_quality), 4),
            "kinematic_sequence_confidence": round(float(rng.uniform(0.3, 0.6)), 4),
            "lag_angle_mid_downswing":     round(float(lag_mid), 2),
            "lag_angle_impact":            round(float(lag_impact), 2),
            "lag_confidence":              round(float(rng.uniform(0.3, 0.55)), 4),
            "xfactor_degrees":             round(float(xfactor), 2),
            "xfactor_confidence":          round(float(rng.uniform(0.35, 0.6)), 4),
            "weight_transfer_timing_ms":   round(float(rng.uniform(-300, 150)), 2),
            "weight_transfer_confidence":  round(float(rng.uniform(0.2, 0.5)), 4),
            "club_path_consistency":       round(float(rng.uniform(0.40, 0.70)), 4),
            "club_path_confidence":        round(float(rng.uniform(0.3, 0.55)), 4),
            "swing_tempo_ratio":           round(float(rng.uniform(0.8, 5.5)), 4),
            "tempo_confidence":            round(float(rng.uniform(0.35, 0.65)), 4),
            "early_cast_flag":             1,
            "reverse_pivot_flag":          1,
            "sway_flag":                   1,
            "early_extension_flag":        1,
            "over_top_flag":               1,
            **compute_swing_outcomes(cfg, comp, rng, seq_quality, lag_mid, xfactor),
        })

    frames_df = pd.DataFrame(frames_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    frames_path = os.path.join(output_dir, "golf_swing_frames.csv")
    metrics_path = os.path.join(output_dir, "golf_swing_metrics.csv")

    frames_df.to_csv(frames_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[OK] golf_swing_frames.csv  — {len(frames_df):,} rows → {frames_path}")
    print(f"[OK] golf_swing_metrics.csv — {len(metrics_df):,} rows → {metrics_path}")
    print(f"     Skill distribution: {metrics_df['skill_level'].value_counts().to_dict()}")

    return frames_df, metrics_df


if __name__ == "__main__":
    generate_dataset()
