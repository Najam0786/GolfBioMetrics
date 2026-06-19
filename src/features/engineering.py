"""
engineering.py
Derived feature computation for the GolfBioMetrics ML feature matrix.

Takes the flat swing-level feature dict (from aggregation.py) and
computes 72 engineered features including demographics, environmental,
and advanced time-series statistical features.

Updated to include time-series analysis from frame-level data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# Import time-series feature extraction
try:
    from src.features.time_series_features import extract_swing_phase_features, extract_lag_features
except ImportError:
    from time_series_features import extract_swing_phase_features, extract_lag_features


def compute_derived_features(swing: dict) -> dict:
    """
    Computes 8 derived features from the core 7 metrics.

    Args:
        swing: flat feature dict produced by aggregation.aggregate_swing()

    Returns:
        the same dict with 8 new keys added (in-place modification + return)
    """
    ks  = float(swing.get("kinematic_sequence_score",    0.0))
    lag = float(swing.get("lag_angle_mid_downswing",     0.0))
    xf  = float(swing.get("xfactor_top_backswing",       0.0))
    cp  = float(swing.get("club_path_consistency",        0.0))
    tr  = float(swing.get("swing_tempo_ratio",            1.0))
    wt  = float(swing.get("weight_transfer_timing_ms",    0.0))
    cs  = float(swing.get("compensation_count",           0.0)) / 5.0
    lag_imp = float(swing.get("lag_angle_impact",         0.0))
    lag_mid = float(swing.get("lag_angle_mid_downswing",  0.0))

    wt_ok = 1.0 if -150.0 <= wt <= -50.0 else 0.0

    swing["sequence_efficiency_index"] = round(
        float(ks * (1.0 - cs)), 4
    )

    lag_norm = lag / 90.0 if lag > 0 else 0.0
    xf_norm  = xf / 55.0 if xf > 0 else 0.0
    swing["power_potential_score"] = round(
        float(xf_norm * lag_norm * (0.5 + 0.5 * wt_ok)), 4
    )

    swing["release_efficiency"] = round(
        float(lag_mid / (lag_imp + 1e-8)), 4
    ) if lag_mid > 0 else 0.0

    swing["tempo_sequence_alignment"] = round(
        float(tr * ks / (1.0 + abs(tr - 2.5))), 4
    )

    bs = float(swing.get("backswing_duration",  0.0))
    ds = float(swing.get("downswing_duration",  0.0))
    swing["swing_speed_index"] = round(
        float(1.0 / (ds + 1e-4)), 4
    )

    lr = float(swing.get("lag_release_rate", 0.0))
    swing["lag_release_smoothness"] = round(
        float(np.clip(1.0 - abs(lr) / 1000.0, 0.0, 1.0)), 4
    )

    conf = float(swing.get("overall_confidence", 0.5))
    swing["weighted_quality_score"] = round(
        float(ks * conf * (1.0 - 0.5 * cs)), 4
    )

    xf_i = float(swing.get("xfactor_at_impact", 0.0))
    swing["xfactor_stretch"] = round(
        float(xf - xf_i), 4
    )

    # ── Age and Experience Features (NEW) ───────────────────────────────────
    age = float(swing.get("age", 35.0))
    years_exp = float(swing.get("years_experience", 10.0))
    
    # Age factor: 1.0 at age 25, declines to ~0.7 at age 65
    # Models physical capability decline
    swing["age_capability_factor"] = round(
        float(1.0 - 0.0075 * max(0, age - 25)), 4
    )
    
    # Experience benefit: plateaus after ~20 years
    # Models motor pattern engrainment
    swing["experience_engrainment"] = round(
        float(min(1.0, years_exp / 20.0)), 4
    )
    
    # Age-adjusted X-Factor: what % of age-expected maximum?
    # Elite at age 25: 55°, at age 55: 45°
    age_expected_max_xf = 55.0 - 0.33 * max(0, age - 25)
    swing["xfactor_age_adjusted"] = round(
        float(xf / age_expected_max_xf) if age_expected_max_xf > 0 else 0.0, 4
    )
    
    # Career stage: early (0), prime (1), veteran (2)
    if years_exp < 5:
        swing["career_stage"] = 0  # Early
    elif years_exp < 15:
        swing["career_stage"] = 1  # Prime
    else:
        swing["career_stage"] = 2  # Veteran

    # ── Gender, Height, Fitness Features (NEW) ────────────────────────────────
    gender = swing.get("gender", "M")
    height_m = float(swing.get("height_m", 1.75))
    fitness = int(swing.get("fitness_level", 5))
    dominant_hand = swing.get("dominant_hand", "right")
    
    # Gender encoding: 0 = Female, 1 = Male (simplified binary for ML)
    swing["gender_encoded"] = 1 if gender == "M" else 0
    
    # Height factor: normalized around 1.75m (average male)
    # Taller = longer lever arm = speed potential
    swing["height_factor"] = round(height_m / 1.75, 4)
    
    # Fitness capability: non-linear (plateaus at high fitness)
    # Low fitness (1-3): significant limitation
    # Medium fitness (4-6): moderate benefit
    # High fitness (7-10): diminishing returns
    if fitness <= 3:
        swing["fitness_capability"] = round(0.5 + 0.1 * fitness, 4)  # 0.6-0.8
    elif fitness <= 6:
        swing["fitness_capability"] = round(0.8 + 0.067 * (fitness - 3), 4)  # 0.8-1.0
    else:
        swing["fitness_capability"] = round(1.0 + 0.02 * (fitness - 6), 4)  # 1.0-1.08
    
    # Dominant hand encoding: 0 = Left, 1 = Right
    swing["dominant_hand_encoded"] = 1 if dominant_hand == "right" else 0
    
    # Combined physical profile score
    swing["physical_profile_score"] = round(
        (swing["age_capability_factor"] * 0.25 +
         swing["fitness_capability"] * 0.35 +
         swing["height_factor"] * 0.25 +
         swing["experience_engrainment"] * 0.15),
        4
    )

    # ── Environmental Features (NEW) ─────────────────────────────────────────
    temperature = float(swing.get("temperature_c", 20.0))
    wind_speed = float(swing.get("wind_speed_mph", 0.0))
    wind_dir = float(swing.get("wind_direction_deg", 0.0))
    humidity = float(swing.get("humidity_pct", 50.0))
    elevation = float(swing.get("elevation_m", 100.0))
    hour = float(swing.get("hour_of_day", 12.0))
    course_type = swing.get("course_type", "parkland")
    
    # Air density factor (elevation effect on ball flight)
    # Sea level = 1.225 kg/m³, decreases ~12% per 1000m elevation
    sea_level_density = 1.225
    altitude_density = sea_level_density * (1 - 2.25577e-5 * elevation)**5.25588
    swing["air_density_factor"] = round(altitude_density / sea_level_density, 4)
    
    # Temperature efficiency (quadratic, optimal ~22°C)
    # Cold muscles stiff, hot causes fatigue
    swing["temperature_efficiency"] = round(
        max(0.5, 1.0 - 0.0015 * (temperature - 22)**2), 4
    )
    
    # Wind components (assume golfer faces north/0° for simplicity)
    # In real app, would use actual shot direction
    wind_dir_rad = np.radians(wind_dir)
    swing["wind_headwind_component"] = round(wind_speed * np.cos(wind_dir_rad), 2)
    swing["wind_crosswind_component"] = round(wind_speed * np.sin(wind_dir_rad), 2)
    
    # Wind effect severity (normalized 0-1)
    swing["wind_effect_severity"] = round(min(1.0, wind_speed / 30.0), 4)
    
    # Circadian rhythm factor (peak performance ~10am-4pm)
    if 10 <= hour <= 16:
        swing["circadian_factor"] = 1.0  # Optimal
    elif 8 <= hour < 10 or 16 < hour <= 18:
        swing["circadian_factor"] = 0.95  # Good
    else:
        swing["circadian_factor"] = 0.88  # Early/late (muscle temp/stiffness)
    
    # Course type encodings (affects wind exposure, playing conditions)
    course_encoding = {
        'parkland': 0,   # Sheltered, moderate conditions
        'links': 1,      # Exposed, windy, unpredictable
        'desert': 2,     # Hot, firm, less wind
        'mountain': 3    # Thin air, dramatic elevation changes
    }
    swing["course_type_encoded"] = course_encoding.get(course_type, 0)
    
    # Links course flag (high wind exposure)
    swing["links_course_flag"] = 1 if course_type == 'links' else 0
    
    # Environmental difficulty index (0 = easy, 1 = extreme)
    swing["env_difficulty_index"] = round(
        0.25 * swing["wind_effect_severity"] +
        0.20 * abs(temperature - 22) / 25 +
        0.20 * (1 - swing["air_density_factor"]) +
        0.15 * (1 - swing["circadian_factor"]) +
        0.20 * swing["links_course_flag"],
        4
    )
    
    # Normalized distance factor (what drive distance would be at sea level, 22°C, no wind)
    # This allows "apples to apples" comparison across conditions
    swing["normalized_distance_factor"] = round(
        (1.0 / swing["air_density_factor"]) *  # Elevation effect
        (1.0 + 0.02 * swing["wind_headwind_component"]) *  # Wind effect
        (1.0 / swing["temperature_efficiency"]),  # Temperature effect
        4
    )

    return swing


def build_feature_matrix(swings: list, include_labels: bool = False) -> pd.DataFrame:
    """
    Converts a list of swing feature dicts into a pandas DataFrame.

    Applies compute_derived_features() to each swing first.

    Args:
        swings:         list of feature dicts from aggregation.aggregate_dataset()
        include_labels: if True, retains label columns in output

    Returns:
        DataFrame with one row per swing, columns = all features
    """
    enriched = [compute_derived_features(dict(s)) for s in swings]
    df = pd.DataFrame(enriched)

    LABEL_COLS = [
        "ball_speed_mph", "carry_distance_yards", "offline_yards",
        "injury_risk_score", "clubhead_speed_mph",
        "swing_quality_class",
    ]
    META_COLS = ["swing_id", "golfer_id", "skill_level", "club_type"]

    feature_cols = [
        c for c in df.columns
        if c not in LABEL_COLS + META_COLS
        and df[c].dtype in [np.float64, np.int64, float, int]
    ]

    output_cols = META_COLS + feature_cols
    if include_labels:
        output_cols += [c for c in LABEL_COLS if c in df.columns]

    return df[[c for c in output_cols if c in df.columns]]


def build_complete_feature_matrix(frames_df: pd.DataFrame, metrics_df: pd.DataFrame, 
                                  include_labels: bool = False) -> pd.DataFrame:
    """
    Builds complete feature matrix including time-series features from frame data.
    
    This is the MASTER function that creates the full 72-feature matrix by:
    1. Starting with swing-level metrics (demographics, environmental, biomechanics)
    2. Extracting time-series features from frame-level data (velocity, jerk, FFT, etc.)
    3. Computing lag features for consistency analysis
    4. Merging everything into the final feature matrix
    
    Args:
        frames_df: Frame-level time-series data (626,000+ rows)
        metrics_df: Swing-level metrics (500 rows)
        include_labels: If True, includes target variables (ball_speed, etc.)
        
    Returns:
        DataFrame with all 72 features per swing
    """
    print("Building complete feature matrix with time-series analysis...")
    
    # Step 1: Start with swing-level derived features
    swing_dicts = metrics_df.to_dict('records')
    enriched_swings = [compute_derived_features(dict(s)) for s in swing_dicts]
    base_df = pd.DataFrame(enriched_swings)
    
    # Step 2: Extract time-series features for each swing from frame data
    print("  Extracting time-series features from frame data...")
    ts_features_list = []
    
    for swing_id in metrics_df['swing_id'].unique():
        # Extract phase features from frame data
        phase_feats = extract_swing_phase_features(frames_df, swing_id)
        
        # Extract lag features for consistency
        golfer_id = metrics_df[metrics_df['swing_id'] == swing_id]['golfer_id'].iloc[0]
        lag_feats = extract_lag_features(metrics_df, golfer_id, n_lags=5)
        
        # Combine
        combined = {'swing_id': swing_id}
        combined.update(phase_feats)
        combined.update(lag_feats)
        ts_features_list.append(combined)
    
    # Convert to DataFrame
    ts_df = pd.DataFrame(ts_features_list)
    
    # Step 3: Merge time-series features with base features
    print(f"  Merging {len(ts_df.columns)-1} time-series features...")
    complete_df = base_df.merge(ts_df, on='swing_id', how='left', suffixes=('', '_ts'))
    
    # Step 4: Handle missing values (some swings may not have time-series data)
    ts_cols = [c for c in ts_df.columns if c != 'swing_id']
    for col in ts_cols:
        if col in complete_df.columns:
            complete_df[col] = complete_df[col].fillna(complete_df[col].median())
    
    # Step 5: Organize columns
    LABEL_COLS = [
        "ball_speed_mph", "carry_distance_yards", "offline_yards",
        "injury_risk_score", "clubhead_speed_mph", "swing_quality_class",
    ]
    META_COLS = ["swing_id", "golfer_id", "skill_level", "club_type"]
    
    # Get all feature columns (numerical only for ML)
    feature_cols = [
        c for c in complete_df.columns
        if c not in LABEL_COLS + META_COLS
        and complete_df[c].dtype in [np.float64, np.int64, float, int]
    ]
    
    # Reorder: meta + features + labels
    output_cols = META_COLS + sorted(feature_cols)
    if include_labels:
        output_cols += [c for c in LABEL_COLS if c in complete_df.columns]
    
    final_df = complete_df[[c for c in output_cols if c in complete_df.columns]]
    
    print(f"  Complete feature matrix: {len(final_df)} swings × {len(feature_cols)} features")
    
    return final_df


def get_feature_names() -> list:
    """
    Returns the canonical ordered list of 72 feature names used by ML models.
    
    Includes biomechanics + demographics + environmental + time-series statistics
    for the most comprehensive golf performance prediction system available.
    """
    return [
        # Core biomechanics (14 features)
        "kinematic_sequence_score",
        "lag_angle_mid_downswing",
        "lag_angle_impact",
        "xfactor_top_backswing",
        "weight_transfer_timing_ms",
        "club_path_consistency",
        "swing_tempo_ratio",
        "early_cast_severity",
        "reverse_pivot_severity",
        "sway_severity",
        "early_extension_severity",
        "over_top_severity",
        "backswing_duration",
        "downswing_duration",
        # Derived cross-metrics (6 features)
        "sequence_efficiency_index",
        "power_potential_score",
        "release_efficiency",
        "tempo_sequence_alignment",
        "xfactor_at_impact",
        "lag_release_rate",
        # Demographic features (12 features)
        "age",
        "years_experience",
        "age_capability_factor",
        "experience_engrainment",
        "xfactor_age_adjusted",
        "gender_encoded",
        "height_m",
        "height_factor",
        "fitness_level",
        "fitness_capability",
        "dominant_hand_encoded",
        "physical_profile_score",
        # Environmental conditions (13 features)
        "temperature_c",
        "temperature_efficiency",
        "wind_speed_mph",
        "wind_direction_deg",
        "wind_headwind_component",
        "wind_crosswind_component",
        "wind_effect_severity",
        "humidity_pct",
        "elevation_m",
        "air_density_factor",
        "hour_of_day",
        "circadian_factor",
        "course_type_encoded",
        "links_course_flag",
        "green_speed_stimp",
        "env_difficulty_index",
        "normalized_distance_factor",
        # Time-Series Statistical Features (27 features) - NEW
        # Velocity & Acceleration Profile
        "club_speed_max",
        "club_speed_mean",
        "club_speed_at_impact",
        "accel_max",
        "accel_mean",
        "time_to_peak_speed",
        "peak_speed_pct_of_swing",
        # Jerk & Smoothness Metrics
        "jerk_max",
        "jerk_mean",
        "jerk_std",
        "motion_smoothness",
        # Phase-Specific Statistics
        "downswing_speed_variance",
        "downswing_speed_skewness",
        "downswing_speed_range",
        # Frequency Domain Features
        "swing_dominant_frequency",
        "swing_spectral_entropy",
        "swing_high_freq_noise",
        # Coordination Timing
        "xfactor_max_ts",
        "xfactor_timing",
        "xfactor_rate_of_change",
        # Lag & Consistency Features
        "seq_consistency",
        "seq_trend",
        "speed_variance_recent",
        "speed_trend",
        "session_fatigue",
        "swings_in_session",
    ]
