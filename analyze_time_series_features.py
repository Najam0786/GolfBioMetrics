"""
analyze_time_series_features.py
Demonstrates advanced statistical feature extraction from time-series swing data.

Shows sophisticated analysis capabilities:
- Velocity/acceleration profiles
- Jerk analysis (smoothness)
- Rolling statistics
- Frequency domain analysis
- Lag features (consistency/trends)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.time_series_features import (
    extract_swing_phase_features,
    extract_lag_features,
    compute_statistical_moments,
    compute_frequency_features,
    compute_all_time_series_features
)

print("=" * 80)
print("ADVANCED TIME-SERIES STATISTICAL ANALYSIS — GolfBioMetrics")
print("=" * 80)

# Load data
frames_df = pd.read_csv('data/synthetic/golf_swing_frames.csv')
metrics_df = pd.read_csv('data/synthetic/golf_swing_metrics.csv')

print(f"\nData loaded:")
print(f"  Frame-level records: {len(frames_df):,}")
print(f"  Swing-level records: {len(metrics_df)}")
print(f"  Unique swings: {frames_df['swing_id'].nunique()}")

# 1. Extract features for a sample of swings
print("\n" + "=" * 80)
print("[1] SWING PHASE KINEMATICS ANALYSIS")
print("=" * 80)

sample_swings = metrics_df[metrics_df['skill_level'].isin(['elite', 'amateur'])].groupby('skill_level').head(3)['swing_id'].values

for swing_id in sample_swings[:3]:
    swing_info = metrics_df[metrics_df['swing_id'] == swing_id].iloc[0]
    features = extract_swing_phase_features(frames_df, swing_id)
    
    print(f"\nSwing #{swing_id} ({swing_info['skill_level'].upper()}, Golfer #{swing_info['golfer_id']}):")
    print("-" * 60)
    
    if 'club_speed_max' in features:
        print(f"  Club Speed Profile:")
        print(f"    Max speed:         {features['club_speed_max']:.2f} m/s")
        print(f"    Mean speed:        {features['club_speed_mean']:.2f} m/s")
        print(f"    Speed at impact:   {features['club_speed_at_impact']:.2f} m/s")
        print(f"    Timing efficiency: {features['club_speed_at_impact']/features['club_speed_max']:.1%}")
    
    if 'motion_smoothness' in features:
        print(f"\n  Motion Quality:")
        print(f"    Smoothness score:  {features['motion_smoothness']:.3f} (1.0 = perfect)")
        print(f"    Jerk (roughness):  {features['jerk_mean']:.3f} m/s³")
        print(f"    Max jerk:          {features['jerk_max']:.3f} m/s³")
        
        # Rating
        if features['motion_smoothness'] > 0.7:
            rating = "EXCELLENT"
        elif features['motion_smoothness'] > 0.5:
            rating = "GOOD"
        else:
            rating = "NEEDS IMPROVEMENT"
        print(f"    Smoothness rating: {rating}")
    
    if 'downswing_speed_variance' in features:
        print(f"\n  Downswing Statistics:")
        print(f"    Speed variance:    {features['downswing_speed_variance']:.3f}")
        print(f"    Speed skewness:    {features['downswing_speed_skewness']:.3f}")
        print(f"    Speed range:       {features['downswing_speed_range']:.2f} m/s")

# 2. Compare skill levels on time-series metrics
print("\n" + "=" * 80)
print("[2] SKILL LEVEL COMPARISON — Time-Series Metrics")
print("=" * 80)

# Extract features for all swings
all_ts_features = []
for swing_id in metrics_df['swing_id'].unique():
    features = extract_swing_phase_features(frames_df, swing_id)
    if features:
        swing_info = metrics_df[metrics_df['swing_id'] == swing_id].iloc[0]
        features['skill_level'] = swing_info['skill_level']
        features['swing_id'] = swing_id
        all_ts_features.append(features)

ts_df = pd.DataFrame(all_ts_features)

# Group by skill level
skill_comparison = ts_df[ts_df['skill_level'].isin(['elite', 'semi_pro', 'amateur'])].groupby('skill_level').agg({
    'motion_smoothness': ['mean', 'std'],
    'jerk_mean': ['mean', 'std'],
    'club_speed_max': ['mean', 'std'],
    'downswing_speed_variance': ['mean', 'std'],
    'swing_high_freq_noise': ['mean', 'std']
}).round(3)

print("\nTime-Series Metrics by Skill Level:")
print(skill_comparison.to_string())

print("\nKey Findings:")
if 'motion_smoothness' in ts_df.columns:
    elite_smooth = ts_df[ts_df['skill_level'] == 'elite']['motion_smoothness'].mean()
    amateur_smooth = ts_df[ts_df['skill_level'] == 'amateur']['motion_smoothness'].mean()
    print(f"  • Elite golfers are {((elite_smooth/amateur_smooth - 1)*100):.1f}% smoother than amateurs")

if 'jerk_mean' in ts_df.columns:
    elite_jerk = ts_df[ts_df['skill_level'] == 'elite']['jerk_mean'].mean()
    amateur_jerk = ts_df[ts_df['skill_level'] == 'amateur']['jerk_mean'].mean()
    print(f"  • Elite golfers have {((1 - elite_jerk/amateur_jerk)*100):.1f}% less jerk (roughness)")

if 'swing_high_freq_noise' in ts_df.columns:
    elite_noise = ts_df[ts_df['skill_level'] == 'elite']['swing_high_freq_noise'].mean()
    amateur_noise = ts_df[ts_df['skill_level'] == 'amateur']['swing_high_freq_noise'].mean()
    print(f"  • Elite golfers have {((1 - elite_noise/amateur_noise)*100):.1f}% less high-freq noise (jitter)")

# 3. Statistical Moments Analysis
print("\n" + "=" * 80)
print("[3] STATISTICAL MOMENTS — Distribution Shape Analysis")
print("=" * 80)

# Get club speed for a representative swing
sample_swing = metrics_df[metrics_df['skill_level'] == 'elite']['swing_id'].iloc[0]
sample_frames = frames_df[(frames_df['swing_id'] == sample_swing) & 
                          (frames_df['keypoint'] == 'club_head')].sort_values('frame_id')

if len(sample_frames) > 10:
    # Compute velocity
    positions = np.column_stack([sample_frames['x'].values, 
                                  sample_frames['y'].values, 
                                  sample_frames['z'].values])
    dt = np.diff(sample_frames['timestamp_s'].values)
    dt = np.append(dt, dt[-1])
    velocity = np.zeros_like(positions)
    velocity[1:-1] = (positions[2:] - positions[:-2]) / (dt[2:] + dt[:-2])[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    
    moments = compute_statistical_moments(speed)
    
    print(f"\nClub Speed Distribution Statistics (Elite Swing #{sample_swing}):")
    print(f"  Mean:      {moments['mean']:.3f} m/s")
    print(f"  Std Dev:   {moments['std']:.3f} m/s")
    print(f"  Variance:  {moments['variance']:.3f}")
    print(f"  Skewness:  {moments['skewness']:.3f}")
    print(f"  Kurtosis:  {moments['kurtosis']:.3f}")
    print(f"  Range:     {moments['range']:.3f} m/s")
    print(f"  CV:        {moments['cv']:.3f} (coefficient of variation)")
    
    print(f"\nInterpretation:")
    if moments['skewness'] > 0.5:
        print(f"  • Skewness {moments['skewness']:.2f} > 0: Right tail (sudden accelerations)")
    elif moments['skewness'] < -0.5:
        print(f"  • Skewness {moments['skewness']:.2f} < 0: Left tail (smooth deceleration)")
    else:
        print(f"  • Skewness {moments['skewness']:.2f} ≈ 0: Symmetric distribution")
    
    if moments['kurtosis'] > 1:
        print(f"  • Kurtosis {moments['kurtosis']:.2f} > 0: Heavy tails (outliers present)")
    else:
        print(f"  • Kurtosis {moments['kurtosis']:.2f} ≈ 0: Normal distribution")

# 4. Frequency Domain Analysis
print("\n" + "=" * 80)
print("[4] FREQUENCY DOMAIN ANALYSIS — FFT Spectral Features")
print("=" * 80)

if len(sample_frames) > 16:
    freq_features = compute_frequency_features(speed, sample_rate=60.0)
    
    print(f"\nFrequency Features for Elite Swing #{sample_swing}:")
    print(f"  Dominant frequency:     {freq_features['dominant_freq']:.2f} Hz")
    print(f"  Spectral entropy:       {freq_features['spectral_entropy']:.2f} bits")
    print(f"  High-freq noise ratio: {freq_features['high_freq_ratio']:.3f}")
    print(f"  Spectral centroid:      {freq_features['spectral_centroid']:.2f} Hz")
    
    print(f"\nInterpretation:")
    print(f"  • Dominant freq {freq_features['dominant_freq']:.2f} Hz = swing tempo pattern")
    print(f"  • Entropy {freq_features['spectral_entropy']:.2f} = ", end="")
    if freq_features['spectral_entropy'] > 2:
        print("random/jittery motion")
    else:
        print("rhythmic/smooth motion")
    print(f"  • High-freq noise {freq_features['high_freq_ratio']:.1%} = ", end="")
    if freq_features['high_freq_ratio'] > 0.3:
        print("significant jitter (compensations)")
    else:
        print("clean motion")

# 5. Lag Features (Consistency Analysis)
print("\n" + "=" * 80)
print("[5] LAG FEATURES — Consistency & Trend Analysis")
print("=" * 80)

# Analyze golfers with multiple swings
golfers_with_multiple = metrics_df['golfer_id'].value_counts()
multi_golfers = golfers_with_multiple[golfers_with_multiple > 3].index[:3]

for golfer_id in multi_golfers:
    lag_feats = extract_lag_features(metrics_df, golfer_id, n_lags=5)
    
    print(f"\nGolfer #{golfer_id} Consistency Analysis:")
    print("-" * 60)
    
    if 'seq_consistency' in lag_feats:
        print(f"  Sequence consistency:   {lag_feats['seq_consistency']:.3f} (1.0 = perfectly consistent)")
    
    if 'seq_trend' in lag_feats:
        trend = lag_feats['seq_trend']
        print(f"  Sequence trend:         {trend:+.4f} per swing", end="")
        if trend > 0.01:
            print(" ← IMPROVING")
        elif trend < -0.01:
            print(" ← DECLINING")
        else:
            print(" ← STABLE")
    
    if 'speed_variance_recent' in lag_feats:
        print(f"  Speed variance (last 5):  {lag_feats['speed_variance_recent']:.2f}")
    
    if 'speed_trend' in lag_feats:
        print(f"  Speed trend:              {lag_feats['speed_trend']:+.3f} mph per swing")
    
    if 'session_fatigue' in lag_feats:
        fatigue = lag_feats['session_fatigue']
        print(f"  Session fatigue:          {fatigue:+.2f} mph degradation", end="")
        if fatigue > 2:
            print(" ⚠️  SIGNIFICANT")
        elif fatigue > 1:
            print(" ⚠️  MODERATE")
        else:
            print(" ✓ MINIMAL")
    
    if 'swings_in_session' in lag_feats:
        print(f"  Swings analyzed:          {lag_feats['swings_in_session']}")

# 6. Advanced Visualization
print("\n" + "=" * 80)
print("[6] Generating Advanced Statistical Visualizations...")
print("=" * 80)

sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Motion Smoothness by Skill Level
ax1 = axes[0, 0]
if 'motion_smoothness' in ts_df.columns and 'skill_level' in ts_df.columns:
    skill_order = ['elite', 'semi_pro', 'amateur', 'amateur_edge', 'elite_edge', 'semi_pro_edge']
    available_skills = []
    smooth_data = []
    for s in skill_order:
        if s in ts_df['skill_level'].values:
            data = ts_df[ts_df['skill_level'] == s]['motion_smoothness'].dropna().values
            if len(data) > 0:
                available_skills.append(s)
                smooth_data.append(data)
    if smooth_data:
        bp = ax1.boxplot(smooth_data, tick_labels=[s.replace('_', ' ').title() for s in available_skills])
        ax1.set_ylabel('Motion Smoothness Score')
        ax1.set_title('Motion Smoothness by Skill Level\n(Higher = Smoother)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

# Plot 2: Jerk Analysis
ax2 = axes[0, 1]
if 'jerk_mean' in ts_df.columns and 'skill_level' in ts_df.columns:
    for skill in ['elite', 'amateur']:
        if skill in ts_df['skill_level'].values:
            data = ts_df[ts_df['skill_level'] == skill]['jerk_mean'].dropna()
            if len(data) > 0 and not np.all(np.isnan(data)):
                ax2.hist(data, bins=20, alpha=0.6, label=skill.title())
    ax2.set_xlabel('Mean Jerk (m/s³)')
    ax2.set_ylabel('Count')
    ax2.set_title('Jerk Distribution: Elite vs Amateur\n(Lower = Smoother)', fontweight='bold')
    ax2.legend()

# Plot 3: Speed Profile Example (Elite vs Amateur)
ax3 = axes[1, 0]
# Get speed profiles for comparison
elite_swing = metrics_df[metrics_df['skill_level'] == 'elite']['swing_id'].iloc[0]
amateur_swing = metrics_df[metrics_df['skill_level'] == 'amateur']['swing_id'].iloc[0]

for swing_id, label, color in [(elite_swing, 'Elite', '#1565C0'), (amateur_swing, 'Amateur', '#E65100')]:
    frames = frames_df[(frames_df['swing_id'] == swing_id) & 
                       (frames_df['keypoint'] == 'club_head')].sort_values('frame_id')
    if len(frames) > 10:
        positions = np.column_stack([frames['x'].values, frames['y'].values, frames['z'].values])
        dt = np.diff(frames['timestamp_s'].values)
        dt = np.append(dt, dt[-1])
        velocity = np.zeros_like(positions)
        velocity[1:-1] = (positions[2:] - positions[:-2]) / (dt[2:] + dt[:-2])[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        
        # Normalize to percentage of swing
        pct = np.linspace(0, 100, len(speed))
        ax3.plot(pct, speed, label=label, color=color, linewidth=2)

ax3.set_xlabel('Swing Progress (%)')
ax3.set_ylabel('Club Speed (m/s)')
ax3.set_title('Speed Profile Comparison\nElite vs Amateur', fontweight='bold')
ax3.legend()
ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Impact')

# Plot 4: Spectral Entropy by Skill
ax4 = axes[1, 1]
if 'swing_spectral_entropy' in ts_df.columns and 'skill_level' in ts_df.columns:
    entropy_data = []
    labels = []
    for skill in ['elite', 'semi_pro', 'amateur']:
        if skill in ts_df['skill_level'].values:
            data = ts_df[ts_df['skill_level'] == skill]['swing_spectral_entropy'].dropna()
            if len(data) > 0:
                entropy_data.append(data)
                labels.append(skill.title())
    
    entropy_data = [d for d in entropy_data if len(d) > 0]
    labels = [labels[i] for i in range(len(entropy_data))]
    if entropy_data and len(entropy_data) == len(labels):
        bp = ax4.boxplot(entropy_data, tick_labels=labels)
        ax4.set_ylabel('Spectral Entropy')
        ax4.set_title('Motion Randomness by Skill\n(Lower = More Rhythmic)', fontweight='bold')

plt.suptitle('Advanced Time-Series Statistical Analysis', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/time_series_statistical_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/figures/time_series_statistical_analysis.png")

# Summary Statistics
print("\n" + "=" * 80)
print("[7] SUMMARY: New Time-Series Features")
print("=" * 80)

feature_counts = {
    'Velocity/Acceleration Profile': 6,
    'Jerk/Smoothness Metrics': 4,
    'Phase-Specific Statistics': 5,
    'Frequency Domain Features': 4,
    'Coordination Timing': 3,
    'Lag/Consistency Features': 5
}

print("\nFeature Categories Added:")
total = 0
for category, count in feature_counts.items():
    print(f"  • {category:<30}: {count} features")
    total += count
print(f"\n  TOTAL TIME-SERIES FEATURES: {total}")

print(f"\nCombined with previous features:")
print(f"  • Original biomechanics:        20 features")
print(f"  • Demographics:                 12 features")
print(f"  • Environmental:                13 features")
print(f"  • Time-series (NEW):            {total} features")
print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  • GRAND TOTAL:                  {20+12+13+total} features")

print("\n" + "=" * 80)
print("KEY STATISTICAL INSIGHTS:")
print("=" * 80)
print("  • Jerk minimization = professional motion quality")
print("  • Spectral entropy < 2 = rhythmic, efficient swings")
print("  • High-freq noise < 30% = clean technique")
print("  • Lag features reveal consistency trends")
print("  • Frequency analysis detects compensatory jitter")
print("=" * 80)
