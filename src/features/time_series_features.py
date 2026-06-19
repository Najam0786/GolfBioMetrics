"""
time_series_features.py
Advanced statistical feature engineering from frame-level time-series data.

Extracts velocity, acceleration, jerk, rolling statistics, and frequency-domain
features from swing kinematics. Demonstrates sophisticated time-series analysis
capabilities for GolfBioMetrics.

Mathematical foundations:
- Numerical differentiation for velocity/acceleration
- Rolling window statistics for phase analysis
- Statistical moments (variance, skewness, kurtosis)
- Smoothness metrics (jerk minimization = efficient motion)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from typing import Dict, List, Tuple, Optional


def compute_velocity_acceleration(positions: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes velocity and acceleration from position time-series.
    
    Uses central differences for numerical differentiation:
    - Velocity: v(t) = [p(t+Δt) - p(t-Δt)] / (2Δt)
    - Acceleration: a(t) = [v(t+Δt) - v(t-Δt)] / (2Δt)
    
    Args:
        positions: Array of positions (can be 1D or 2D for multiple dimensions)
        timestamps: Array of timestamps in seconds
        
    Returns:
        (velocity, acceleration) arrays
    """
    dt = np.diff(timestamps)
    dt = np.append(dt, dt[-1])  # Extend last timestep
    
    # Central difference for velocity
    velocity = np.zeros_like(positions)
    velocity[1:-1] = (positions[2:] - positions[:-2]) / (timestamps[2:] - timestamps[:-2])[:, None] if positions.ndim > 1 else (positions[2:] - positions[:-2]) / (timestamps[2:] - timestamps[:-2])
    
    # Forward/backward difference at boundaries
    velocity[0] = (positions[1] - positions[0]) / (timestamps[1] - timestamps[0]) if positions.ndim > 1 else (positions[1] - positions[0]) / (timestamps[1] - timestamps[0])
    velocity[-1] = (positions[-1] - positions[-2]) / (timestamps[-1] - timestamps[-2]) if positions.ndim > 1 else (positions[-1] - positions[-2]) / (timestamps[-1] - timestamps[-2])
    
    # Central difference for acceleration
    acceleration = np.zeros_like(positions)
    acceleration[1:-1] = (velocity[2:] - velocity[:-2]) / (timestamps[2:] - timestamps[:-2])[:, None] if velocity.ndim > 1 else (velocity[2:] - velocity[:-2]) / (timestamps[2:] - timestamps[:-2])
    acceleration[0] = (velocity[1] - velocity[0]) / (timestamps[1] - timestamps[0]) if velocity.ndim > 1 else (velocity[1] - velocity[0]) / (timestamps[1] - timestamps[0])
    acceleration[-1] = (velocity[-1] - velocity[-2]) / (timestamps[-1] - timestamps[-2]) if velocity.ndim > 1 else (velocity[-1] - velocity[-2]) / (timestamps[-1] - timestamps[-2])
    
    return velocity, acceleration


def compute_jerk(acceleration: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Computes jerk (rate of change of acceleration).
    
    Jerk minimization indicates smooth, efficient motion.
    High jerk = abrupt changes = compensatory movements or inefficiency.
    
    Args:
        acceleration: Acceleration array
        timestamps: Time array
        
    Returns:
        Jerk array
    """
    jerk = np.zeros_like(acceleration)
    jerk[1:-1] = (acceleration[2:] - acceleration[:-2]) / (timestamps[2:] - timestamps[:-2])[:, None] if acceleration.ndim > 1 else (acceleration[2:] - acceleration[:-2]) / (timestamps[2:] - timestamps[:-2])
    jerk[0] = (acceleration[1] - acceleration[0]) / (timestamps[1] - timestamps[0]) if acceleration.ndim > 1 else (acceleration[1] - acceleration[0]) / (timestamps[1] - timestamps[0])
    jerk[-1] = (acceleration[-1] - acceleration[-2]) / (timestamps[-1] - timestamps[-2]) if acceleration.ndim > 1 else (acceleration[-1] - acceleration[-2]) / (timestamps[-1] - timestamps[-2])
    
    return jerk


def rolling_statistics(values: np.ndarray, window_size: int = 5) -> Dict[str, np.ndarray]:
    """
    Computes rolling window statistics.
    
    Captures local variability and trends within swing phases.
    
    Args:
        values: Input array
        window_size: Rolling window size (in frames)
        
    Returns:
        Dictionary of rolling statistics
    """
    # Pad for edge effects
    padded = np.pad(values, (window_size//2, window_size//2), mode='edge')
    
    rolling_mean = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    
    # Variance requires 2D convolution for efficiency
    rolling_var = np.array([
        np.var(padded[i:i+window_size]) 
        for i in range(len(values))
    ])
    
    return {
        'rolling_mean': rolling_mean,
        'rolling_var': rolling_var,
        'rolling_std': np.sqrt(rolling_var)
    }


def compute_statistical_moments(values: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical moments: mean, variance, skewness, kurtosis.
    
    Higher moments reveal distribution shape characteristics:
    - Skewness > 0: Right tail (sudden accelerations)
    - Kurtosis > 3: Heavy tails (outliers, compensations)
    
    Args:
        values: Input array
        
    Returns:
        Dictionary of moment statistics
    """
    return {
        'mean': np.mean(values),
        'variance': np.var(values),
        'std': np.std(values),
        'skewness': stats.skew(values),
        'kurtosis': stats.kurtosis(values),
        'range': np.max(values) - np.min(values),
        'cv': np.std(values) / (np.mean(values) + 1e-8)  # Coefficient of variation
    }


def compute_frequency_features(signal: np.ndarray, sample_rate: float = 60.0) -> Dict[str, float]:
    """
    Extracts frequency-domain features using FFT.
    
    Reveals rhythmic patterns and smoothness:
    - Dominant frequency: Primary oscillation rate
    - Spectral entropy: Randomness vs periodicity
    - High frequency ratio: Noise/jitter vs smooth motion
    
    Args:
        signal: Time-series signal
        sample_rate: Frames per second (default 60fps)
        
    Returns:
        Dictionary of frequency features
    """
    if len(signal) < 8:
        return {
            'dominant_freq': 0.0,
            'spectral_entropy': 0.0,
            'high_freq_ratio': 0.0,
            'spectral_centroid': 0.0
        }
    
    # FFT
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Only positive frequencies
    positive_mask = freqs > 0
    fft_vals = fft_vals[positive_mask]
    freqs = freqs[positive_mask]
    
    # Power spectrum
    power = fft_vals ** 2
    power_sum = np.sum(power) + 1e-8
    
    # Dominant frequency (peak)
    dominant_idx = np.argmax(fft_vals)
    dominant_freq = freqs[dominant_idx]
    
    # Spectral entropy (randomness measure)
    power_norm = power / power_sum
    spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-8))
    
    # High frequency ratio (noise vs signal)
    high_freq_mask = freqs > 5  # Above 5Hz considered high frequency
    high_freq_power = np.sum(power[high_freq_mask]) if np.any(high_freq_mask) else 0
    high_freq_ratio = high_freq_power / power_sum
    
    # Spectral centroid ("center of mass" of spectrum)
    spectral_centroid = np.sum(freqs * power) / power_sum
    
    return {
        'dominant_freq': dominant_freq,
        'spectral_entropy': spectral_entropy,
        'high_freq_ratio': high_freq_ratio,
        'spectral_centroid': spectral_centroid
    }


def extract_swing_phase_features(frames_df: pd.DataFrame, swing_id: int) -> Dict[str, float]:
    """
    Extracts time-series features for a single swing from frame data.
    
    Computes sophisticated kinematic features:
    - Club head velocity profile
    - Hip/shoulder coordination timing
    - Smoothness metrics (jerk minimization)
    - Phase-specific statistics
    
    Args:
        frames_df: DataFrame with frame-level data
        swing_id: Swing ID to analyze
        
    Returns:
        Dictionary of time-series features
    """
    swing_frames = frames_df[frames_df['swing_id'] == swing_id].sort_values('frame_id')
    
    if len(swing_frames) < 10:
        return {}
    
    # Get timestamps
    timestamps = swing_frames['timestamp_s'].values
    
    # Club head trajectory (primary motion)
    club_data = swing_frames[swing_frames['keypoint'] == 'club_head']
    if len(club_data) < 10:
        return {}
    
    club_positions = np.column_stack([
        club_data['x'].values,
        club_data['y'].values,
        club_data['z'].values
    ])
    
    # Compute velocity and acceleration
    club_velocity, club_acceleration = compute_velocity_acceleration(club_positions, timestamps[:len(club_positions)])
    club_jerk = compute_jerk(club_acceleration, timestamps[:len(club_acceleration)])
    
    # Club speed magnitude
    club_speed = np.linalg.norm(club_velocity, axis=1)
    club_accel_mag = np.linalg.norm(club_acceleration, axis=1)
    club_jerk_mag = np.linalg.norm(club_jerk, axis=1)
    
    # Identify swing phases based on club position
    # Top of backswing: minimum y (highest point behind golfer)
    # Impact: minimum z (closest to ground)
    top_backswing_idx = np.argmin(club_data['y'].values)
    impact_idx = np.argmin(club_data['z'].values)
    
    # Split into phases
    backswing = club_speed[:top_backswing_idx] if top_backswing_idx > 0 else club_speed[:1]
    downswing = club_speed[top_backswing_idx:impact_idx] if impact_idx > top_backswing_idx else club_speed[top_backswing_idx:top_backswing_idx+1]
    follow_through = club_speed[impact_idx:] if impact_idx < len(club_speed) else club_speed[-1:]
    
    features = {}
    
    # Overall swing statistics
    features['club_speed_max'] = float(np.max(club_speed))
    features['club_speed_mean'] = float(np.mean(club_speed))
    features['club_speed_at_impact'] = float(club_speed[impact_idx]) if impact_idx < len(club_speed) else 0.0
    
    # Acceleration features
    features['accel_max'] = float(np.max(club_accel_mag))
    features['accel_mean'] = float(np.mean(club_accel_mag))
    
    # Jerk features (smoothness)
    features['jerk_max'] = float(np.max(club_jerk_mag))
    features['jerk_mean'] = float(np.mean(club_jerk_mag))
    features['jerk_std'] = float(np.std(club_jerk_mag))
    features['motion_smoothness'] = float(1.0 / (1.0 + np.mean(club_jerk_mag)))  # Higher = smoother
    
    # Phase-specific features
    if len(downswing) > 5:
        ds_moments = compute_statistical_moments(downswing)
        features['downswing_speed_variance'] = float(ds_moments['variance'])
        features['downswing_speed_skewness'] = float(ds_moments['skewness'])
        features['downswing_speed_range'] = float(ds_moments['range'])
    
    # Time to peak speed (efficiency metric)
    peak_speed_idx = np.argmax(club_speed)
    features['time_to_peak_speed'] = float(timestamps[peak_speed_idx] - timestamps[0]) if peak_speed_idx < len(timestamps) else 0.0
    features['peak_speed_pct_of_swing'] = float(peak_speed_idx / len(club_speed)) if len(club_speed) > 0 else 0.0
    
    # Frequency analysis of speed profile
    if len(club_speed) > 16:
        freq_features = compute_frequency_features(club_speed, sample_rate=60.0)
        features['swing_dominant_frequency'] = float(freq_features['dominant_freq'])
        features['swing_spectral_entropy'] = float(freq_features['spectral_entropy'])
        features['swing_high_freq_noise'] = float(freq_features['high_freq_ratio'])
    
    # Coordination features (if we have body data)
    hip_data = swing_frames[swing_frames['keypoint'] == 'left_hip']
    shoulder_data = swing_frames[swing_frames['keypoint'] == 'left_shoulder']
    
    if len(hip_data) > 10 and len(shoulder_data) > 10:
        hip_y = hip_data['y'].values[:len(timestamps)]
        shoulder_y = shoulder_data['y'].values[:len(timestamps)]
        
        # Hip-shoulder separation (X-factor) time series
        x_factor_ts = np.abs(shoulder_y - hip_y)
        
        features['xfactor_max_ts'] = float(np.max(x_factor_ts))
        features['xfactor_timing'] = float(timestamps[np.argmax(x_factor_ts)] - timestamps[0])
        features['xfactor_rate_of_change'] = float(np.max(np.abs(np.diff(x_factor_ts))))
    
    return features


def compute_rolling_window_features(values: np.ndarray, window: int = 5) -> Dict[str, float]:
    """
    Computes rolling window features for trend analysis.
    
    Args:
        values: Input time-series
        window: Window size in samples
        
    Returns:
        Dictionary of rolling features
    """
    if len(values) < window:
        return {}
    
    # Moving average
    ma = pd.Series(values).rolling(window=window, center=True, min_periods=1).mean()
    
    # Rate of change
    roc = pd.Series(values).pct_change().rolling(window=window, min_periods=1).mean()
    
    # Acceleration of change (second derivative)
    accel = pd.Series(values).diff().diff().rolling(window=window, min_periods=1).mean()
    
    return {
        'rolling_mean_trend': float(np.mean(ma)),
        'rolling_mean_slope': float(np.polyfit(range(len(ma)), ma, 1)[0]) if len(ma) > 1 else 0.0,
        'rate_of_change_mean': float(np.nanmean(roc)),
        'rate_of_change_volatility': float(np.nanstd(roc)),
        'acceleration_mean': float(np.nanmean(accel)),
    }


def extract_lag_features(swing_metrics: pd.DataFrame, golfer_id: int, n_lags: int = 3) -> Dict[str, float]:
    """
    Extracts lag features from previous swings by the same golfer.
    
    Captures:
    - Consistency (variance across recent swings)
    - Trend (improving/declining)
    - Fatigue effects (degradation over session)
    
    Args:
        swing_metrics: DataFrame with swing-level metrics
        golfer_id: Golfer to analyze
        n_lags: Number of previous swings to consider
        
    Returns:
        Dictionary of lag features
    """
    golfer_swings = swing_metrics[swing_metrics['golfer_id'] == golfer_id].sort_values('swing_id')
    
    if len(golfer_swings) < 2:
        return {}
    
    features = {}
    
    # Get recent swings
    recent = golfer_swings.tail(n_lags)
    
    # Consistency features
    if 'kinematic_sequence_score' in recent.columns:
        features['seq_consistency'] = float(1.0 - recent['kinematic_sequence_score'].std())  # Higher = more consistent
        features['seq_trend'] = float(np.polyfit(range(len(recent)), recent['kinematic_sequence_score'], 1)[0])
    
    if 'ball_speed_mph' in recent.columns:
        features['speed_variance_recent'] = float(recent['ball_speed_mph'].var())
        features['speed_trend'] = float(np.polyfit(range(len(recent)), recent['ball_speed_mph'], 1)[0])
    
    # Fatigue indicator (degradation over session)
    if len(golfer_swings) > 5:
        first_half = golfer_swings.head(len(golfer_swings)//2)['ball_speed_mph'].mean()
        second_half = golfer_swings.tail(len(golfer_swings)//2)['ball_speed_mph'].mean()
        features['session_fatigue'] = float(first_half - second_half)  # Positive = fatigue
    
    # Swing count (experience in this session)
    features['swings_in_session'] = int(len(golfer_swings))
    
    return features


def compute_all_time_series_features(frames_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to compute all time-series features for all swings.
    
    Args:
        frames_df: Frame-level time-series data
        metrics_df: Swing-level metrics (to merge with)
        
    Returns:
        DataFrame with enhanced features
    """
    print("Computing time-series features for all swings...")
    
    all_features = []
    
    for swing_id in metrics_df['swing_id'].unique():
        # Extract swing phase features from frame data
        phase_features = extract_swing_phase_features(frames_df, swing_id)
        
        # Get golfer_id for lag features
        golfer_id = metrics_df[metrics_df['swing_id'] == swing_id]['golfer_id'].iloc[0]
        lag_feats = extract_lag_features(metrics_df, golfer_id)
        
        # Combine
        combined = {**phase_features, **lag_feats, 'swing_id': swing_id}
        all_features.append(combined)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Merge with original metrics
    enhanced_df = metrics_df.merge(features_df, on='swing_id', how='left')
    
    print(f"Added {len(features_df.columns)-1} time-series features")
    
    return enhanced_df


# Convenience function for single swing analysis
def analyze_swing_kinematics(frames_df: pd.DataFrame, swing_id: int) -> Dict:
    """
    Complete kinematic analysis of a single swing.
    
    Returns comprehensive dictionary with all time-series insights.
    """
    features = extract_swing_phase_features(frames_df, swing_id)
    
    # Add interpretation
    interpretation = {}
    
    if 'motion_smoothness' in features:
        if features['motion_smoothness'] > 0.7:
            interpretation['smoothness_rating'] = 'Excellent'
        elif features['motion_smoothness'] > 0.5:
            interpretation['smoothness_rating'] = 'Good'
        else:
            interpretation['smoothness_rating'] = 'Needs Work'
    
    if 'club_speed_max' in features and 'club_speed_at_impact' in features:
        efficiency = features['club_speed_at_impact'] / features['club_speed_max'] if features['club_speed_max'] > 0 else 0
        interpretation['speed_efficiency'] = f"{efficiency:.1%}"
        if efficiency > 0.95:
            interpretation['timing'] = 'Perfect timing - peak speed at impact'
        elif efficiency > 0.85:
            interpretation['timing'] = 'Good timing'
        else:
            interpretation['timing'] = 'Early/late peak - speed not at impact'
    
    return {
        'features': features,
        'interpretation': interpretation
    }
