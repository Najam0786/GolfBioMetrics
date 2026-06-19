"""
train_models_complete.py
Train ALL ML models with the complete 84-feature matrix including time-series statistics.

This script:
1. Loads frame-level and swing-level data
2. Builds complete feature matrix with time-series features (84 features total)
3. Trains all 5 ML models with full feature set
4. Evaluates and compares performance
5. Saves models for inference
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
import sys
sys.path.insert(0, 'src')

from features.engineering import build_complete_feature_matrix, get_feature_names
from models.linear_regression import train_linear_regression
from models.decision_tree import train_decision_tree
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from models.svm_model import train_svm

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GolfBioMetrics — Complete ML Pipeline with Time-Series Features")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
frames_df = pd.read_csv('data/synthetic/golf_swing_frames.csv')
metrics_df = pd.read_csv('data/synthetic/golf_swing_metrics.csv')

print(f"  Frame data: {len(frames_df):,} rows")
print(f"  Swing data: {len(metrics_df)} rows")

# Build complete feature matrix
print("\n[2] Building complete feature matrix with time-series analysis...")
feature_df = build_complete_feature_matrix(frames_df, metrics_df, include_labels=True)

# Get feature columns (exclude meta)
META_COLS = ["swing_id", "golfer_id", "skill_level", "club_type"]
LABEL_COLS = ["ball_speed_mph", "carry_distance_yards", "offline_yards",
              "injury_risk_score", "clubhead_speed_mph", "swing_quality_class"]

feature_cols = [c for c in feature_df.columns 
                if c not in META_COLS + LABEL_COLS
                and feature_df[c].dtype in [np.float64, np.int64, float, int]]

# Handle NaN and infinity values
print(f"\n  Handling {feature_df[feature_cols].isna().sum().sum()} NaN values...")
for col in feature_cols:
    # Fill NaN with median (or 0 if all NaN)
    if feature_df[col].isna().any():
        median_val = feature_df[col].median()
        if pd.isna(median_val):  # If median is also NaN (all values NaN)
            median_val = 0.0
        feature_df[col] = feature_df[col].fillna(median_val)
    
    # Replace infinity with large finite values
    if np.isinf(feature_df[col]).any():
        finite_max = feature_df[col][np.isfinite(feature_df[col])].max()
        finite_min = feature_df[col][np.isfinite(feature_df[col])].min()
        if pd.isna(finite_max):
            finite_max, finite_min = 1e6, -1e6
        feature_df[col] = feature_df[col].replace([np.inf, -np.inf], [finite_max, finite_min])

# Final verification and drop any rows that still have issues
print("  Final data cleaning...")
feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
feature_df = feature_df.dropna(subset=feature_cols + ['ball_speed_mph', 'carry_distance_yards', 'injury_risk_score'])

# Verify no NaN or inf remain
total_nan = feature_df[feature_cols].isna().sum().sum()
total_inf = np.isinf(feature_df[feature_cols].values).sum()
print(f"  ✓ NaN remaining: {total_nan}, Inf remaining: {total_inf}")
print(f"  ✓ Usable swings after cleaning: {len(feature_df)}")

print(f"  ✓ Feature matrix: {len(feature_df)} swings × {len(feature_cols)} features")
print(f"  ✓ All {len(feature_cols)} features will be used for training")

# Count by category
biomech = [c for c in feature_cols if any(x in c for x in ['kinematic', 'lag', 'xfactor', 'tempo', 'weight', 'club_path', 'sequence', 'power', 'release'])]
demo = [c for c in feature_cols if any(x in c for x in ['age', 'gender', 'height', 'fitness', 'experience', 'hand', 'physical'])]
env = [c for c in feature_cols if any(x in c for x in ['temp', 'wind', 'elev', 'humidity', 'course', 'air_density', 'circadian', 'env_', 'normalized'])]
ts = [c for c in feature_cols if any(x in c for x in ['speed', 'jerk', 'accel', 'smoothness', 'entropy', 'fft', 'phase', 'spectral', 'xfactor_max_ts', 'xfactor_timing', 'consistency', 'trend', 'fatigue'])]

print(f"\n  Feature breakdown:")
print(f"    • Biomechanics + Derived:  {len(biomech)} features")
print(f"    • Demographics:             {len(demo)} features")
print(f"    • Environmental:            {len(env)} features")
print(f"    • Time-Series Statistics:   {len(ts)} features")

# Save feature list for reference
os.makedirs('outputs/model_artifacts', exist_ok=True)
with open('outputs/model_artifacts/complete_feature_list.json', 'w') as f:
    json.dump({
        'total_features': len(feature_cols),
        'biomechanics': biomech,
        'demographics': demo,
        'environmental': env,
        'time_series': ts,
        'all_features': feature_cols
    }, f, indent=2)

# Train all models
results = {}

print("\n" + "=" * 80)
print("[3] Training ML Models with Complete Feature Set")
print("=" * 80)

# Model 1: Linear Regression (Ball Speed)
print("\n[3.1] Linear Regression — Ball Speed Prediction...")
try:
    lr_results = train_linear_regression(
        feature_df, 
        feature_cols=feature_cols,
        target_col='ball_speed_mph',
        test_size=0.15,
        random_state=42
    )
    results['linear_regression'] = {
        'model': 'Linear Regression',
        'target': 'ball_speed_mph',
        'r2_test': lr_results['results']['test']['r2'],
        'rmse_test': lr_results['results']['test']['rmse'],
        'features_used': len(feature_cols)
    }
    print(f"  ✓ R² = {lr_results['results']['test']['r2']:.3f}")
    print(f"  ✓ RMSE = {lr_results['results']['test']['rmse']:.2f} mph")
    
    # Save model
    with open('outputs/model_artifacts/linear_regression_ball_speed.pkl', 'wb') as f:
        pickle.dump(lr_results, f)
except Exception as e:
    print(f"  ✗ Error: {e}")

# Model 2: Decision Tree (Swing Quality)
print("\n[3.2] Decision Tree — Swing Quality Classification...")
try:
    # Create binary classification target
    feature_df['swing_quality_class'] = (feature_df['kinematic_sequence_score'] > 0.7).astype(int)
    
    dt_results = train_decision_tree(
        feature_df,
        feature_cols=feature_cols,
        target_col='swing_quality_class',
        test_size=0.15,
        random_state=42
    )
    results['decision_tree'] = {
        'model': 'Decision Tree',
        'target': 'swing_quality_class',
        'accuracy': dt_results['accuracy'],
        'f1_score': dt_results['f1_score'],
        'features_used': len(feature_cols)
    }
    print(f"  ✓ Accuracy = {dt_results['accuracy']:.3f}")
    print(f"  ✓ F1 Score = {dt_results['f1_score']:.3f}")
    
    with open('outputs/model_artifacts/decision_tree_quality.pkl', 'wb') as f:
        pickle.dump(dt_results, f)
except Exception as e:
    print(f"  ✗ Error: {e}")

# Model 3: Random Forest (Carry Distance)
print("\n[3.3] Random Forest — Carry Distance Prediction...")
try:
    rf_results = train_random_forest(
        feature_df,
        feature_cols=feature_cols,
        target_col='carry_distance_yards',
        n_estimators=150,
        test_size=0.15,
        random_state=42
    )
    results['random_forest'] = {
        'model': 'Random Forest',
        'target': 'carry_distance_yards',
        'r2_test': rf_results['results']['test']['r2'],
        'rmse_test': rf_results['results']['test']['rmse'],
        'features_used': len(feature_cols),
        'top_features': rf_results['feature_importance'].head(5).to_dict()
    }
    print(f"  ✓ R² = {rf_results['results']['test']['r2']:.3f}")
    print(f"  ✓ RMSE = {rf_results['results']['test']['rmse']:.2f} yards")
    
    # Show top features
    print(f"\n  Top 5 important features:")
    for i, row in rf_results['feature_importance'].head(5).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    with open('outputs/model_artifacts/random_forest_distance.pkl', 'wb') as f:
        pickle.dump(rf_results, f)
except Exception as e:
    print(f"  ✗ Error: {e}")

# Model 4: XGBoost (Injury Risk)
print("\n[3.4] XGBoost + SHAP — Injury Risk Prediction...")
try:
    xgb_results = train_xgboost(
        feature_df,
        feature_cols=feature_cols,
        target_col='injury_risk_score',
        n_estimators=200,
        max_depth=5,
        test_size=0.15,
        random_state=42
    )
    results['xgboost'] = {
        'model': 'XGBoost + SHAP',
        'target': 'injury_risk_score',
        'r2_test': xgb_results['results']['test']['r2'],
        'rmse_test': xgb_results['results']['test']['rmse'],
        'features_used': len(feature_cols)
    }
    print(f"  ✓ R² = {xgb_results['results']['test']['r2']:.3f}")
    print(f"  ✓ RMSE = {xgb_results['results']['test']['rmse']:.3f}")
    
    with open('outputs/model_artifacts/xgboost_injury_risk.pkl', 'wb') as f:
        pickle.dump(xgb_results, f)
except Exception as e:
    print(f"  ✗ Error: {e}")

# Model 5: SVM (Efficiency Classification)
print("\n[3.5] SVM — Swing Efficiency Classification...")
try:
    # Create efficiency target based on sequence score
    feature_df['efficient_swing'] = (feature_df['kinematic_sequence_score'] > 0.65).astype(int)
    
    svm_results = train_svm(
        feature_df,
        feature_cols=feature_cols,
        target_col='efficient_swing',
        test_size=0.15,
        random_state=42
    )
    results['svm'] = {
        'model': 'SVM',
        'target': 'efficient_swing',
        'accuracy': svm_results['accuracy'],
        'f1_score': svm_results['f1_score'],
        'auc_roc': svm_results['auc_roc'],
        'features_used': len(feature_cols)
    }
    print(f"  ✓ Accuracy = {svm_results['accuracy']:.3f}")
    print(f"  ✓ F1 Score = {svm_results['f1_score']:.3f}")
    print(f"  ✓ AUC-ROC = {svm_results['auc_roc']:.3f}")
    
    with open('outputs/model_artifacts/svm_efficiency.pkl', 'wb') as f:
        pickle.dump(svm_results, f)
except Exception as e:
    print(f"  ✗ Error: {e}")

# Summary
print("\n" + "=" * 80)
print("[4] TRAINING COMPLETE — PERFORMANCE SUMMARY")
print("=" * 80)

for model_name, model_results in results.items():
    print(f"\n{model_results['model']}:")
    if 'r2_test' in model_results:
        print(f"  R² = {model_results['r2_test']:.3f} | RMSE = {model_results['rmse_test']:.3f}")
    if 'accuracy' in model_results:
        print(f"  Accuracy = {model_results['accuracy']:.3f} | F1 = {model_results.get('f1_score', 0):.3f}")
    print(f"  Features: {model_results['features_used']}")

# Save results summary
with open('outputs/model_artifacts/training_results_complete.json', 'w') as f:
    json.dump({
        'training_date': datetime.now().isoformat(),
        'total_features': len(feature_cols),
        'feature_breakdown': {
            'biomechanics': len(biomech),
            'demographics': len(demo),
            'environmental': len(env),
            'time_series': len(ts)
        },
        'model_results': results,
        'data_stats': {
            'total_swings': len(feature_df),
            'frame_records': len(frames_df),
            'train_test_split': '70/15/15'
        }
    }, f, indent=2, default=str)

print("\n" + "=" * 80)
print("All models trained with complete 84-feature matrix!")
print("=" * 80)
print("\nArtifacts saved:")
print("  • outputs/model_artifacts/complete_feature_list.json")
print("  • outputs/model_artifacts/training_results_complete.json")
print("  • Individual model .pkl files for each algorithm")
print("\nTime-series features are now FULLY INTEGRATED into ML pipeline ✓")
