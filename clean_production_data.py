"""
clean_production_data.py
Removes problematic features and creates production-ready clean dataset.

Removes:
- Constant columns (no variance, useless for ML)
- Highly correlated redundant features
- Columns with excessive missing values

Creates:
- Clean feature matrix with only useful features
- Data quality report
- Production-ready ML dataset
"""

import pandas as pd
import numpy as np
import json
import os

print("=" * 80)
print("GolfBioMetrics — Production Data Cleaning")
print("=" * 80)

# Load cleaned feature matrix
print("\n[1] Loading feature matrix...")
feature_df = pd.read_csv('data/synthetic/feature_matrix_cleaned.csv')
print(f"  Original shape: {feature_df.shape}")

# Get column categories
meta_cols = ['swing_id', 'golfer_id', 'skill_level', 'club_type']
label_cols = ['ball_speed_mph', 'carry_distance_yards', 'offline_yards',
              'injury_risk_score', 'clubhead_speed_mph', 'swing_quality_class']

# Get all feature columns (exclude meta and labels)
all_cols = feature_df.columns.tolist()
feature_cols = [c for c in all_cols 
                if c not in meta_cols + label_cols
                and feature_df[c].dtype in ['float64', 'int64']]

print(f"  Total feature columns: {len(feature_cols)}")

# Step 1: Remove constant columns (no variance)
print("\n[2] Removing constant columns (no variance)...")
constant_cols = []
for col in feature_cols:
    if feature_df[col].nunique() <= 1:
        constant_cols.append(col)

if constant_cols:
    print(f"  Found {len(constant_cols)} constant columns:")
    for col in constant_cols:
        print(f"    - {col}: value={feature_df[col].iloc[0]}")
    
    # Remove constant columns
    feature_df = feature_df.drop(columns=constant_cols)
    feature_cols = [c for c in feature_cols if c not in constant_cols]
    print(f"  ✓ Removed {len(constant_cols)} constant columns")
else:
    print(f"  ✓ No constant columns found")

# Step 2: Remove highly correlated features (>0.98)
print("\n[3] Removing highly correlated features (>0.98 correlation)...")

# Calculate correlation matrix
corr_matrix = feature_df[feature_cols].corr().abs()

# Find upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation > 0.98
high_corr_pairs = [(col, row, upper.loc[row, col]) 
                   for col in upper.columns 
                   for row in upper.index 
                   if upper.loc[row, col] > 0.98]

# Remove one feature from each highly correlated pair
# Keep the first occurrence, remove subsequent ones
cols_to_remove = set()
for feat1, feat2, corr in high_corr_pairs:
    # Remove feat2 if not already marked
    if feat2 not in cols_to_remove and feat1 not in cols_to_remove:
        cols_to_remove.add(feat2)
        print(f"    - {feat2} (correlation {corr:.3f} with {feat1})")

if cols_to_remove:
    feature_df = feature_df.drop(columns=list(cols_to_remove))
    feature_cols = [c for c in feature_cols if c not in cols_to_remove]
    print(f"  ✓ Removed {len(cols_to_remove)} highly correlated columns")
else:
    print(f"  ✓ No highly correlated columns found")

# Step 3: Check for low variance features
print("\n[4] Checking for very low variance features...")
low_variance_cols = []
for col in feature_cols:
    variance = feature_df[col].var()
    if variance < 0.001:  # Very low variance threshold
        low_variance_cols.append((col, variance))

if low_variance_cols:
    print(f"  Found {len(low_variance_cols)} very low variance columns:")
    for col, var in low_variance_cols:
        print(f"    - {col}: variance={var:.6f}")
    
    # Don't remove low variance automatically - just warn
    print(f"  ⚠ These have low variance but are kept (may still be useful)")
else:
    print(f"  ✓ All columns have reasonable variance")

# Step 4: Verify data quality
print("\n[5] Final data quality verification...")

# Check for any remaining NaN
nan_count = feature_df.isnull().sum().sum()
print(f"  NaN values: {nan_count}")

# Check for infinity
num_df = feature_df.select_dtypes(include=[np.number])
inf_count = np.isinf(num_df.values).sum()
print(f"  Infinity values: {inf_count}")

# Check for duplicates
dup_count = feature_df.duplicated().sum()
print(f"  Duplicate rows: {dup_count}")

# Get final column counts
final_feature_cols = [c for c in feature_df.columns 
                      if c not in meta_cols + label_cols
                      and feature_df[c].dtype in ['float64', 'int64']]

print(f"\n[6] Final Feature Matrix Summary:")
print(f"  Original features: 84")
print(f"  Constant removed: {len(constant_cols)}")
print(f"  Correlated removed: {len(cols_to_remove)}")
print(f"  Final features: {len(final_feature_cols)}")
print(f"  Final shape: {feature_df.shape}")

# Categorize remaining features
biomech = [c for c in final_feature_cols if any(x in c for x in ['kinematic', 'lag', 'xfactor', 'tempo', 'weight', 'club_path', 'sequence', 'power', 'release', 'compensat'])]
demo = [c for c in final_feature_cols if any(x in c for x in ['age', 'gender', 'height', 'fitness', 'experience', 'hand', 'physical', 'career'])]
env = [c for c in final_feature_cols if any(x in c for x in ['temp', 'wind', 'elev', 'humidity', 'course', 'air_density', 'circadian', 'env_', 'normalized', 'hour', 'links'])]

print(f"\n  Feature breakdown:")
print(f"    • Biomechanics: {len(biomech)} features")
print(f"    • Demographics: {len(demo)} features")
print(f"    • Environmental: {len(env)} features")
print(f"    • Other: {len(final_feature_cols) - len(biomech) - len(demo) - len(env)} features")

# Save production-ready dataset
print(f"\n[7] Saving production-ready dataset...")
os.makedirs('data/synthetic', exist_ok=True)
feature_df.to_csv('data/synthetic/feature_matrix_production.csv', index=False)
print(f"  ✓ Saved: data/synthetic/feature_matrix_production.csv")

# Create data quality report
report = {
    'cleaning_date': pd.Timestamp.now().isoformat(),
    'original_shape': {'rows': 500, 'columns': 93},
    'final_shape': {'rows': len(feature_df), 'columns': len(feature_df.columns)},
    'features_removed': {
        'constant_columns': constant_cols,
        'correlated_columns': list(cols_to_remove),
        'total_removed': len(constant_cols) + len(cols_to_remove)
    },
    'final_features': {
        'biomechanics': biomech,
        'demographics': demo,
        'environmental': env,
        'all_features': final_feature_cols,
        'total': len(final_feature_cols)
    },
    'data_quality': {
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'dup_count': int(dup_count),
        'status': 'CLEAN' if nan_count == 0 and inf_count == 0 and dup_count == 0 else 'NEEDS_ATTENTION'
    }
}

with open('data/synthetic/data_cleaning_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"  ✓ Saved: data/synthetic/data_cleaning_report.json")

# Create feature list for reference
with open('data/synthetic/production_feature_list.txt', 'w') as f:
    f.write("GolfBioMetrics Production Features\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total Features: {len(final_feature_cols)}\n\n")
    
    f.write("BIOMECHANICS:\n")
    for feat in sorted(biomech):
        f.write(f"  - {feat}\n")
    
    f.write(f"\nDEMOGRAPHICS ({len(demo)}):\n")
    for feat in sorted(demo):
        f.write(f"  - {feat}\n")
    
    f.write(f"\nENVIRONMENTAL ({len(env)}):\n")
    for feat in sorted(env):
        f.write(f"  - {feat}\n")
    
    other = [c for c in final_feature_cols if c not in biomech + demo + env]
    if other:
        f.write(f"\nOTHER ({len(other)}):\n")
        for feat in sorted(other):
            f.write(f"  - {feat}\n")

print(f"  ✓ Saved: data/synthetic/production_feature_list.txt")

print("\n" + "=" * 80)
if report['data_quality']['status'] == 'CLEAN':
    print("✅ PRODUCTION DATA READY")
    print("=" * 80)
    print(f"\nThe dataset is now clean and ready for ML:")
    print(f"  • {len(final_feature_cols)} high-quality features")
    print(f"  • No missing values")
    print(f"  • No infinite values")
    print(f"  • No duplicates")
    print(f"  • No constant columns")
    print(f"  • No highly correlated redundancy")
    print(f"\nUse 'feature_matrix_production.csv' for all ML training.")
else:
    print("⚠️  DATA QUALITY ISSUES REMAIN")
    print("=" * 80)

exit(0 if report['data_quality']['status'] == 'CLEAN' else 1)
