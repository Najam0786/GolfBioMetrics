"""
validate_and_clean_data.py
Comprehensive data quality validation and cleaning for GolfBioMetrics.

Ensures:
- No null/NaN values in critical columns
- No infinity values
- No duplicate records
- Proper data types
- Consistent ranges
- Clean feature matrix for ML

This script should be run before any ML training or analysis.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from features.engineering import build_complete_feature_matrix

print("=" * 80)
print("GolfBioMetrics — Comprehensive Data Validation & Cleaning")
print("=" * 80)

issues_found = []
warnings_found = []

# 1. Validate Frame-Level Data
print("\n[1] Validating Frame-Level Data (golf_swing_frames.csv)...")
print("-" * 80)

try:
    frames_df = pd.read_csv('data/synthetic/golf_swing_frames.csv')
    print(f"  Total records: {len(frames_df):,}")
    
    # Check required columns
    required_cols = ['swing_id', 'frame_id', 'timestamp_s', 'golfer_id', 'keypoint', 'x', 'y', 'z', 'confidence']
    missing_cols = [c for c in required_cols if c not in frames_df.columns]
    if missing_cols:
        issues_found.append(f"Frame data missing columns: {missing_cols}")
    else:
        print(f"  ✓ All required columns present")
    
    # Check for nulls
    null_counts = frames_df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        issues_found.append(f"Frame data has nulls: {null_cols.to_dict()}")
    else:
        print(f"  ✓ No null values")
    
    # Check for NaN in numerical columns
    num_cols = ['x', 'y', 'z', 'confidence', 'timestamp_s']
    nan_counts = frames_df[num_cols].isna().sum()
    if nan_counts.sum() > 0:
        issues_found.append(f"Frame data NaN in numerical: {nan_counts[nan_counts > 0].to_dict()}")
    else:
        print(f"  ✓ No NaN in numerical columns")
    
    # Check infinity
    inf_counts = np.isinf(frames_df[num_cols]).sum()
    if inf_counts.sum() > 0:
        issues_found.append(f"Frame data has infinity: {inf_counts[inf_counts > 0].to_dict()}")
    else:
        print(f"  ✓ No infinity values")
    
    # Check duplicates
    dups = frames_df.duplicated().sum()
    if dups > 0:
        issues_found.append(f"Frame data has {dups} duplicate rows")
    else:
        print(f"  ✓ No duplicate rows")
    
    # Check swing_id consistency
    unique_swings = frames_df['swing_id'].nunique()
    print(f"  ✓ Unique swings: {unique_swings}")
    
    # Check keypoint distribution
    keypoint_counts = frames_df['keypoint'].value_counts()
    expected_keypoints = 18  # 18 keypoints per frame
    frames_per_swing = len(frames_df) / unique_swings / expected_keypoints
    print(f"  ✓ Average frames per swing: {frames_per_swing:.1f}")
    
    # Check confidence range
    conf_min = frames_df['confidence'].min()
    conf_max = frames_df['confidence'].max()
    if conf_min < 0 or conf_max > 1:
        warnings_found.append(f"Confidence values out of [0,1] range: min={conf_min}, max={conf_max}")
    else:
        print(f"  ✓ Confidence in valid range [0,1]")
    
    # Check timestamp consistency
    ts_issues = frames_df.groupby('swing_id')['timestamp_s'].apply(lambda x: not x.is_monotonic_increasing).sum()
    if ts_issues > 0:
        warnings_found.append(f"{ts_issues} swings have non-monotonic timestamps")
    else:
        print(f"  ✓ All timestamps monotonically increasing")
    
except Exception as e:
    issues_found.append(f"Error reading frame data: {str(e)}")

# 2. Validate Swing-Level Metrics
print("\n[2] Validating Swing-Level Data (golf_swing_metrics.csv)...")
print("-" * 80)

try:
    metrics_df = pd.read_csv('data/synthetic/golf_swing_metrics.csv')
    print(f"  Total swings: {len(metrics_df)}")
    
    # Check required columns
    required_metrics = ['swing_id', 'golfer_id', 'skill_level', 'kinematic_sequence_score', 
                       'ball_speed_mph', 'injury_risk_score']
    missing_metrics = [c for c in required_metrics if c not in metrics_df.columns]
    if missing_metrics:
        issues_found.append(f"Metrics missing columns: {missing_metrics}")
    else:
        print(f"  ✓ All required columns present")
    
    # Check for nulls
    null_counts = metrics_df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        issues_found.append(f"Metrics has nulls: {null_cols.to_dict()}")
    else:
        print(f"  ✓ No null values")
    
    # Check duplicates
    dups = metrics_df.duplicated(subset=['swing_id']).sum()
    if dups > 0:
        issues_found.append(f"Metrics has {dups} duplicate swing_ids")
    else:
        print(f"  ✓ No duplicate swing_ids")
    
    # Check swing_id uniqueness
    if metrics_df['swing_id'].nunique() != len(metrics_df):
        issues_found.append("Swing IDs not unique in metrics")
    else:
        print(f"  ✓ All swing_ids unique")
    
    # Check golfer_id validity
    unique_golfers = metrics_df['golfer_id'].nunique()
    print(f"  ✓ Unique golfers: {unique_golfers}")
    
    # Check skill_level distribution
    skill_dist = metrics_df['skill_level'].value_counts()
    print(f"  ✓ Skill distribution: {skill_dist.to_dict()}")
    
    # Check numerical ranges
    numerical_checks = {
        'kinematic_sequence_score': (0, 1),
        'ball_speed_mph': (40, 120),
        'injury_risk_score': (0, 1),
        'xfactor_degrees': (10, 60),
        'lag_angle_mid_downswing': (20, 100)
    }
    
    for col, (min_val, max_val) in numerical_checks.items():
        if col in metrics_df.columns:
            actual_min = metrics_df[col].min()
            actual_max = metrics_df[col].max()
            if actual_min < min_val or actual_max > max_val:
                warnings_found.append(f"{col} out of expected range: [{actual_min:.2f}, {actual_max:.2f}] vs expected [{min_val}, {max_val}]")
            else:
                print(f"  ✓ {col} in range [{min_val}, {max_val}]")
    
    # Check environmental features (new)
    env_cols = ['temperature_c', 'wind_speed_mph', 'elevation_m']
    for col in env_cols:
        if col in metrics_df.columns:
            nan_count = metrics_df[col].isna().sum()
            if nan_count > 0:
                issues_found.append(f"Environmental column {col} has {nan_count} NaN values")
            else:
                print(f"  ✓ Environmental feature '{col}' has no NaN")
    
except Exception as e:
    issues_found.append(f"Error reading metrics: {str(e)}")

# 3. Validate Complete Feature Matrix
print("\n[3] Validating Complete Feature Matrix (with time-series)...")
print("-" * 80)

try:
    print("  Building complete feature matrix...")
    feature_df = build_complete_feature_matrix(frames_df, metrics_df, include_labels=True)
    
    print(f"  Matrix shape: {feature_df.shape}")
    
    # Check for NaN
    nan_counts = feature_df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) > 0:
        print(f"  ⚠ Found {len(nan_cols)} columns with NaN values:")
        for col, count in nan_cols.items():
            print(f"    - {col}: {count} NaN ({count/len(feature_df)*100:.1f}%)")
        
        # Clean NaN values
        print(f"  Cleaning NaN values...")
        for col in nan_cols.index:
            if feature_df[col].dtype in ['float64', 'int64']:
                median_val = feature_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                feature_df[col] = feature_df[col].fillna(median_val)
            else:
                feature_df[col] = feature_df[col].fillna('unknown')
        
        # Verify cleaning
        remaining_nan = feature_df.isnull().sum().sum()
        if remaining_nan == 0:
            print(f"  ✓ All NaN values cleaned")
        else:
            issues_found.append(f"Could not clean all NaN: {remaining_nan} remaining")
    else:
        print(f"  ✓ No NaN values in feature matrix")
    
    # Check for infinity
    num_cols = feature_df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(feature_df[num_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    
    if len(inf_cols) > 0:
        print(f"  ⚠ Found {len(inf_cols)} columns with infinity:")
        for col, count in inf_cols.items():
            print(f"    - {col}: {count} inf values")
        
        # Clean infinity
        print(f"  Cleaning infinity values...")
        for col in inf_cols.index:
            finite_max = feature_df[col][np.isfinite(feature_df[col])].max()
            finite_min = feature_df[col][np.isfinite(feature_df[col])].min()
            if pd.isna(finite_max):
                finite_max, finite_min = 1e6, -1e6
            feature_df[col] = feature_df[col].replace([np.inf, -np.inf], [finite_max, finite_min])
        
        remaining_inf = np.isinf(feature_df[num_cols].values).sum()
        if remaining_inf == 0:
            print(f"  ✓ All infinity values cleaned")
        else:
            issues_found.append(f"Could not clean all infinity: {remaining_inf} remaining")
    else:
        print(f"  ✓ No infinity values in feature matrix")
    
    # Check feature column counts
    meta_cols = ['swing_id', 'golfer_id', 'skill_level', 'club_type']
    label_cols = ['ball_speed_mph', 'carry_distance_yards', 'offline_yards',
                  'injury_risk_score', 'clubhead_speed_mph']
    
    feature_cols = [c for c in feature_df.columns 
                    if c not in meta_cols + label_cols
                    and feature_df[c].dtype in ['float64', 'int64']]
    
    print(f"  ✓ Feature columns: {len(feature_cols)}")
    print(f"  ✓ Meta columns: {len(meta_cols)}")
    print(f"  ✓ Label columns: {len([c for c in label_cols if c in feature_df.columns])}")
    
    # Check for constant columns (no variance)
    constant_cols = []
    for col in feature_cols:
        if feature_df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        warnings_found.append(f"Constant columns (no variance): {constant_cols}")
    else:
        print(f"  ✓ All feature columns have variance")
    
    # Check correlation between features
    print(f"  Checking feature correlations...")
    corr_matrix = feature_df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(col, row, upper.loc[row, col]) 
                 for col in upper.columns 
                 for row in upper.index 
                 if upper.loc[row, col] > 0.95]
    
    if high_corr:
        warnings_found.append(f"{len(high_corr)} highly correlated feature pairs (>0.95)")
        for feat1, feat2, corr in high_corr[:3]:
            print(f"    ⚠ High correlation: {feat1} vs {feat2}: {corr:.3f}")
    else:
        print(f"  ✓ No problematic correlations (>0.95)")
    
    # Save cleaned feature matrix
    print(f"\n  Saving cleaned feature matrix...")
    feature_df.to_csv('data/synthetic/feature_matrix_cleaned.csv', index=False)
    print(f"  ✓ Saved: data/synthetic/feature_matrix_cleaned.csv")
    
except Exception as e:
    issues_found.append(f"Error building feature matrix: {str(e)}")
    import traceback
    traceback.print_exc()

# 4. Validate ML Model Artifacts
print("\n[4] Validating ML Model Artifacts...")
print("-" * 80)

if os.path.exists('outputs/model_artifacts'):
    artifacts = os.listdir('outputs/model_artifacts')
    print(f"  Found {len(artifacts)} artifacts:")
    for artifact in artifacts:
        print(f"    - {artifact}")
else:
    warnings_found.append("No model artifacts directory found")

# 5. Summary Report
print("\n" + "=" * 80)
print("[5] DATA QUALITY SUMMARY REPORT")
print("=" * 80)

if issues_found:
    print(f"\n❌ CRITICAL ISSUES FOUND: {len(issues_found)}")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print(f"\n✅ NO CRITICAL ISSUES FOUND")

if warnings_found:
    print(f"\n⚠️  WARNINGS: {len(warnings_found)}")
    for i, warning in enumerate(warnings_found, 1):
        print(f"  {i}. {warning}")
else:
    print(f"\n✅ NO WARNINGS")

print(f"\n" + "=" * 80)
if not issues_found:
    print("✅ DATA VALIDATION PASSED — All systems ready for ML training!")
    print("=" * 80)
    print(f"\nCleaned data available at:")
    print(f"  • data/synthetic/feature_matrix_cleaned.csv")
    print(f"  • data/synthetic/golf_swing_frames.csv")
    print(f"  • data/synthetic/golf_swing_metrics.csv")
    exit(0)
else:
    print("❌ DATA VALIDATION FAILED — Please fix issues before training")
    print("=" * 80)
    exit(1)
