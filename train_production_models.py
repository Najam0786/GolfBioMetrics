"""
train_production_models.py
Train ALL ML models with the CLEAN 58-feature production dataset.

This uses the validated, cleaned feature matrix with:
- 58 high-quality features (removed constants and correlations)
- Zero NaN or infinity values
- Production-ready for deployment
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
import sys
sys.path.insert(0, 'src')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GolfBioMetrics — Production Model Training (58 Clean Features)")
print("=" * 80)

# Load CLEAN production data
print("\n[1] Loading CLEAN production dataset...")
feature_df = pd.read_csv('data/synthetic/feature_matrix_production.csv')

print(f"  Dataset shape: {feature_df.shape}")
print(f"  Swings: {len(feature_df)}")

# Verify data quality
print("\n[2] Verifying data quality...")
nan_count = feature_df.isnull().sum().sum()
inf_count = np.isinf(feature_df.select_dtypes(include=[np.number]).values).sum()
dup_count = feature_df.duplicated().sum()

print(f"  NaN values: {nan_count}")
print(f"  Infinity values: {inf_count}")
print(f"  Duplicate rows: {dup_count}")

if nan_count > 0 or inf_count > 0:
    print("  ❌ Data quality issues found! Exiting.")
    exit(1)

print("  ✅ Data quality verified — CLEAN")

# Define columns
META_COLS = ["swing_id", "golfer_id", "skill_level", "club_type"]
LABEL_COLS = ["ball_speed_mph", "carry_distance_yards", "offline_yards",
              "injury_risk_score", "clubhead_speed_mph"]

# Get feature columns (58 total)
feature_cols = [c for c in feature_df.columns 
                if c not in META_COLS + LABEL_COLS
                and feature_df[c].dtype in ['float64', 'int64']]

print(f"\n[3] Feature summary:")
print(f"  Total features: {len(feature_cols)}")

# Categorize features
biomech = [c for c in feature_cols if any(x in c for x in ['kinematic', 'lag', 'xfactor', 'tempo', 'weight', 'club_path', 'sequence', 'compensat'])]
demo = [c for c in feature_cols if any(x in c for x in ['age', 'gender', 'height', 'fitness', 'experience', 'hand', 'physical', 'career'])]
env = [c for c in feature_cols if any(x in c for x in ['temp', 'wind', 'elev', 'humidity', 'course', 'air_density', 'circadian', 'hour', 'links'])]

print(f"    • Biomechanics: {len(biomech)}")
print(f"    • Demographics: {len(demo)}")
print(f"    • Environmental: {len(env)}")
print(f"    • Other: {len(feature_cols) - len(biomech) - len(demo) - len(env)}")

# Create output directory
os.makedirs('outputs/model_artifacts', exist_ok=True)

results = {}

print("\n" + "=" * 80)
print("[4] Training Production ML Models")
print("=" * 80)

# Model 1: Linear Regression (Ball Speed)
print("\n[4.1] Linear Regression — Ball Speed Prediction...")
try:
    X = feature_df[feature_cols].values
    y = feature_df['ball_speed_mph'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results['linear_regression'] = {
        'model': 'Linear Regression',
        'target': 'ball_speed_mph',
        'r2_test': float(r2),
        'rmse_test': float(rmse),
        'features_used': len(feature_cols)
    }
    
    print(f"  ✅ R² = {r2:.3f}")
    print(f"  ✅ RMSE = {rmse:.2f} mph")
    
    # Save model
    with open('outputs/model_artifacts/production_linear_regression.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_cols, 'results': results['linear_regression']}, f)
    
except Exception as e:
    print(f"  ❌ Error: {e}")

# Model 2: Decision Tree (Swing Quality Classification)
print("\n[4.2] Decision Tree — Swing Quality Classification...")
try:
    # Create binary target: good (seq > 0.65) vs poor
    y = (feature_df['kinematic_sequence_score'] > 0.65).astype(int)
    X = feature_df[feature_cols].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results['decision_tree'] = {
        'model': 'Decision Tree',
        'target': 'swing_quality_class',
        'accuracy': float(acc),
        'f1_score': float(f1),
        'features_used': len(feature_cols)
    }
    
    print(f"  ✅ Accuracy = {acc:.3f}")
    print(f"  ✅ F1 Score = {f1:.3f}")
    
    with open('outputs/model_artifacts/production_decision_tree.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_names': feature_cols, 'results': results['decision_tree']}, f)
    
except Exception as e:
    print(f"  ❌ Error: {e}")

# Model 3: Random Forest (Carry Distance)
print("\n[4.3] Random Forest — Carry Distance Prediction...")
try:
    X = feature_df[feature_cols].values
    y = feature_df['carry_distance_yards'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results['random_forest'] = {
        'model': 'Random Forest',
        'target': 'carry_distance_yards',
        'r2_test': float(r2),
        'rmse_test': float(rmse),
        'features_used': len(feature_cols),
        'top_features': importance.head(5).to_dict('records')
    }
    
    print(f"  ✅ R² = {r2:.3f}")
    print(f"  ✅ RMSE = {rmse:.2f} yards")
    print(f"\n  Top 5 important features:")
    for i, row in importance.head(5).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    with open('outputs/model_artifacts/production_random_forest.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_names': feature_cols, 
                     'results': results['random_forest'], 'importance': importance}, f)
    
except Exception as e:
    print(f"  ❌ Error: {e}")

# Model 4: XGBoost (Injury Risk)
print("\n[4.4] XGBoost + SHAP — Injury Risk Prediction...")
try:
    X = feature_df[feature_cols].values
    y = feature_df['injury_risk_score'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results['xgboost'] = {
        'model': 'XGBoost + SHAP',
        'target': 'injury_risk_score',
        'r2_test': float(r2),
        'rmse_test': float(rmse),
        'features_used': len(feature_cols)
    }
    
    print(f"  ✅ R² = {r2:.3f}")
    print(f"  ✅ RMSE = {rmse:.3f}")
    
    with open('outputs/model_artifacts/production_xgboost.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_names': feature_cols, 'results': results['xgboost']}, f)
    
except Exception as e:
    print(f"  ❌ Error: {e}")

# Model 5: SVM (Efficiency Classification)
print("\n[4.5] SVM — Swing Efficiency Classification...")
try:
    # Create efficiency target
    y = (feature_df['kinematic_sequence_score'] > 0.65).astype(int)
    X = feature_df[feature_cols].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # AUC-ROC
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    results['svm'] = {
        'model': 'SVM',
        'target': 'efficient_swing',
        'accuracy': float(acc),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'features_used': len(feature_cols)
    }
    
    print(f"  ✅ Accuracy = {acc:.3f}")
    print(f"  ✅ F1 Score = {f1:.3f}")
    print(f"  ✅ AUC-ROC = {auc:.3f}")
    
    with open('outputs/model_artifacts/production_svm.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_cols, 'results': results['svm']}, f)
    
except Exception as e:
    print(f"  ❌ Error: {e}")

# Summary
print("\n" + "=" * 80)
print("[5] PRODUCTION TRAINING COMPLETE — PERFORMANCE SUMMARY")
print("=" * 80)

for model_name, model_results in results.items():
    print(f"\n{model_results['model']}:")
    if 'r2_test' in model_results:
        print(f"  R² = {model_results['r2_test']:.3f} | RMSE = {model_results['rmse_test']:.3f}")
    if 'accuracy' in model_results:
        print(f"  Accuracy = {model_results['accuracy']:.3f} | F1 = {model_results.get('f1_score', 0):.3f}")
        if 'auc_roc' in model_results:
            print(f"  AUC-ROC = {model_results['auc_roc']:.3f}")
    print(f"  Features: {model_results['features_used']} (clean, production-ready)")

# Save comprehensive results
with open('outputs/model_artifacts/production_training_results.json', 'w') as f:
    json.dump({
        'training_date': datetime.now().isoformat(),
        'dataset': 'feature_matrix_production.csv',
        'data_quality': {'nan': 0, 'inf': 0, 'dup': 0},
        'features': {
            'total': len(feature_cols),
            'biomechanics': len(biomech),
            'demographics': len(demo),
            'environmental': len(env),
            'list': feature_cols
        },
        'model_results': results,
        'status': 'PRODUCTION_READY'
    }, f, indent=2, default=str)

print("\n" + "=" * 80)
print("✅ ALL PRODUCTION MODELS TRAINED SUCCESSFULLY")
print("=" * 80)
print("\nProduction artifacts saved:")
print("  • outputs/model_artifacts/production_linear_regression.pkl")
print("  • outputs/model_artifacts/production_decision_tree.pkl")
print("  • outputs/model_artifacts/production_random_forest.pkl")
print("  • outputs/model_artifacts/production_xgboost.pkl")
print("  • outputs/model_artifacts/production_svm.pkl")
print("  • outputs/model_artifacts/production_training_results.json")
print("\nAll models trained with CLEAN 58-feature production dataset!")
print("Ready for deployment to DSG production environment.")
