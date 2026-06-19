"""
run_all_and_report.py
Execute all components and generate comprehensive results for DSG business analysis.

This script:
1. Loads synthetic data
2. Runs all 5 ML models
3. Generates business insights
4. Creates figures for reports
5. Outputs results summary
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kruskal
import warnings
warnings.filterwarnings('ignore')

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from src.features.engineering import compute_derived_features, get_feature_names
from src.models.linear_regression import train_linear_regression
from src.models.decision_tree import train_decision_tree, create_quality_labels
from src.models.random_forest import train_random_forest
from src.models.xgboost_model import train_xgboost, explain_golfer
from src.models.svm_model import train_svm, create_efficiency_labels
from src.validation.model_evaluation import compare_models
from src.validation.metric_validation import run_full_validation

print("=" * 70)
print("GOLFBIOMETRICS — COMPREHENSIVE RESULTS GENERATION")
print("Data Sports Group (DSG) POC | Author: Nazmul Farooquee")
print("=" * 70)

# Create output directories
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/model_artifacts', exist_ok=True)

# Load data
print("\n[1] Loading synthetic data...")
metrics_df = pd.read_csv('data/synthetic/golf_swing_metrics.csv')
print(f"    Loaded {len(metrics_df)} swings")
print(f"    Skill distribution: {metrics_df['skill_level'].value_counts().to_dict()}")

# Build feature matrix
print("\n[2] Building ML feature matrix...")
rows = []
for _, row in metrics_df.iterrows():
    d = row.to_dict()
    d['xfactor_top_backswing'] = d.get('xfactor_degrees', 0.0)
    d['xfactor_at_impact'] = d.get('xfactor_degrees', 0.0) * 0.5
    d['backswing_duration'] = abs(d.get('swing_tempo_ratio', 2.5)) * 0.25
    d['downswing_duration'] = 0.25
    d['lag_release_rate'] = (d.get('lag_angle_mid_downswing', 70) - d.get('lag_angle_impact', 25)) / 0.2
    d['overall_confidence'] = d.get('kinematic_sequence_confidence', 0.8)
    d['early_cast_severity'] = d.get('early_cast_flag', 0) * 0.5
    d['reverse_pivot_severity'] = d.get('reverse_pivot_flag', 0) * 0.5
    d['sway_severity'] = d.get('sway_flag', 0) * 0.5
    d['early_extension_severity'] = d.get('early_extension_flag', 0) * 0.5
    d['over_top_severity'] = d.get('over_top_flag', 0) * 0.5
    rows.append(compute_derived_features(d))

df = pd.DataFrame(rows)
print(f"    Feature matrix shape: {df.shape}")

# Train all models
print("\n[3] Training ML models...")

print("\n    [3.1] Linear Regression (Ball Speed Prediction)")
lr_artifacts = train_linear_regression(df)
print(f"          Train R²: {lr_artifacts['results']['train']['r2']:.3f}")
print(f"          Test R²:  {lr_artifacts['results']['test']['r2']:.3f}")
print(f"          Test RMSE: {lr_artifacts['results']['test']['rmse']:.2f} mph")

print("\n    [3.2] Decision Tree (Swing Quality Classification)")
df['swing_quality_class'] = create_quality_labels(df, source_col='kinematic_sequence_score')
dt_artifacts = train_decision_tree(df)
print(f"          Train Acc: {dt_artifacts['results']['train']['accuracy']:.3f}")
print(f"          Test Acc:  {dt_artifacts['results']['test']['accuracy']:.3f}")

print("\n    [3.3] Random Forest (Carry Distance Prediction)")
rf_artifacts = train_random_forest(df)
print(f"          Train R²: {rf_artifacts['results']['train']['r2']:.3f}")
print(f"          Test R²:  {rf_artifacts['results']['test']['r2']:.3f}")
print(f"          Test RMSE: {rf_artifacts['results']['test']['rmse']:.2f} yards")

print("\n    [3.4] XGBoost + SHAP (Injury Risk Prediction)")
xgb_artifacts = train_xgboost(df)
print(f"          Train R²: {xgb_artifacts['results']['train']['r2']:.3f}")
print(f"          Test R²:  {xgb_artifacts['results']['test']['r2']:.3f}")
print(f"          Test RMSE: {xgb_artifacts['results']['test']['rmse']:.4f}")

print("\n    [3.5] SVM (Efficient vs Inefficient Classification)")
df['swing_efficient'] = create_efficiency_labels(df, source_col='kinematic_sequence_score')
svm_artifacts = train_svm(df)
print(f"          Train Acc: {svm_artifacts['results']['train']['accuracy']:.3f}")
print(f"          Test Acc:  {svm_artifacts['results']['test']['accuracy']:.3f}")
print(f"          Test AUC:  {svm_artifacts['results']['test']['auc_roc']:.3f}")

# Model comparison
all_models = {
    'Linear Regression': lr_artifacts,
    'Decision Tree': dt_artifacts,
    'Random Forest': rf_artifacts,
    'XGBoost + SHAP': xgb_artifacts,
    'SVM': svm_artifacts,
}

comparison = compare_models(all_models)
print("\n[4] Model Comparison Summary:")
print(comparison.fillna('-').to_string())

# Save comparison to CSV
comparison.to_csv('outputs/reports/model_comparison.csv', index=False)
print("    Saved to: outputs/reports/model_comparison.csv")

# Generate visualizations
print("\n[5] Generating visualizations...")

plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid')

# Figure 1: Metric distributions by skill level
metrics_to_plot = [
    ('kinematic_sequence_score', 'Kinematic Sequence Score'),
    ('xfactor_degrees', 'X-Factor (degrees)'),
    ('lag_angle_mid_downswing', 'Lag Angle Mid-Downswing (°)'),
    ('swing_tempo_ratio', 'Swing Tempo Ratio'),
]
skill_order = ['elite', 'semi_pro', 'amateur']
palette = {'elite': '#1565C0', 'semi_pro': '#388E3C', 'amateur': '#E65100'}

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, (col, title) in zip(axes, metrics_to_plot):
    data_by_group = [metrics_df[metrics_df['skill_level'] == g][col].dropna().values 
                     for g in skill_order]
    bp = ax.boxplot(data_by_group, patch_artist=True, notch=True)
    for patch, skill in zip(bp['boxes'], skill_order):
        patch.set_facecolor(palette[skill])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(skill_order) + 1))
    ax.set_xticklabels(skill_order, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.4)
plt.suptitle('Metric Distributions by Skill Level', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/figures/metric_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: outputs/figures/metric_distributions.png")

# Figure 2: Metric-Outcome Correlations
if 'ball_speed_mph' in metrics_df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    scatter_pairs = [
        ('kinematic_sequence_score', 'ball_speed_mph'),
        ('xfactor_degrees', 'ball_speed_mph'),
        ('lag_angle_mid_downswing', 'ball_speed_mph'),
    ]
    for ax, (x_col, y_col) in zip(axes, scatter_pairs):
        clean = metrics_df[metrics_df['skill_level'].isin(skill_order)][[x_col, y_col, 'skill_level']].dropna()
        for skill in skill_order:
            subset = clean[clean['skill_level'] == skill]
            ax.scatter(subset[x_col], subset[y_col], alpha=0.4, s=12,
                      color=palette[skill], label=skill)
        r, p = pearsonr(clean[x_col], clean[y_col])
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.set_title(f'r = {r:.3f} (p={p:.2e})', fontweight='bold')
        ax.legend(fontsize=7)
    plt.suptitle('Metric-Outcome Correlations', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/metric_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: outputs/figures/metric_correlations.png")

# Figure 3: Random Forest Feature Importance
fi = rf_artifacts['feature_importance'].head(10)
fig, ax = plt.subplots(figsize=(8, 5))
fi_sorted = fi.sort_values('importance')
ax.barh(fi_sorted['feature'], fi_sorted['importance_pct'], color='#1565C0')
ax.set_xlabel('Importance (%)', fontweight='bold')
ax.set_title('Random Forest - Feature Importance for Carry Distance', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: outputs/figures/rf_feature_importance.png")

# Statistical validation
print("\n[6] Running statistical validation...")
validation_results = run_full_validation(metrics_df)
print(f"    Validation results: {len(validation_results)} metrics checked")

# Generate key business insights
print("\n" + "=" * 70)
print("KEY BUSINESS INSIGHTS FOR DATA SPORTS GROUP (DSG)")
print("=" * 70)

# Insight 1: Skill level discrimination
print("\n[INSIGHT 1] Skill Level Discrimination Power")
print("-" * 50)
for metric, label in metrics_to_plot:
    groups = [metrics_df[metrics_df['skill_level'] == g][metric].dropna() for g in skill_order]
    h_stat, p_val = kruskal(*groups)
    means = {g: metrics_df[metrics_df['skill_level'] == g][metric].mean() for g in skill_order}
    print(f"  {label}:")
    print(f"    Elite avg:    {means['elite']:.2f}")
    print(f"    Semi-pro avg: {means['semi_pro']:.2f}")
    print(f"    Amateur avg:  {means['amateur']:.2f}")
    print(f"    Kruskal-Wallis H={h_stat:.2f}, p={p_val:.2e}")
    print()

# Insight 2: Performance prediction accuracy
print("[INSIGHT 2] Performance Prediction Accuracy")
print("-" * 50)
print(f"  Ball Speed (Linear Regression):    R² = {lr_artifacts['results']['test']['r2']:.3f}")
print(f"  Carry Distance (Random Forest):    R² = {rf_artifacts['results']['test']['r2']:.3f}")
print(f"  Injury Risk (XGBoost):             R² = {xgb_artifacts['results']['test']['r2']:.3f}")

# Insight 3: Top risk factors
print("\n[INSIGHT 3] Top Injury Risk Factors (XGBoost)")
print("-" * 50)
top_features = xgb_artifacts['feature_importance'].head(5)
for _, row in top_features.iterrows():
    importance_val = row.get('importance_pct', row.get('importance', 0))
    print(f"  {row['feature']:<35}: {importance_val:.1f}%")

# Insight 4: Business value metrics
print("\n[INSIGHT 4] Business Value Metrics")
print("-" * 50)
high_risk_count = (metrics_df['injury_risk_score'] > 0.6).sum()
moderate_risk_count = ((metrics_df['injury_risk_score'] > 0.3) & (metrics_df['injury_risk_score'] <= 0.6)).sum()
low_risk_count = (metrics_df['injury_risk_score'] <= 0.3).sum()
print(f"  High injury risk golfers:    {high_risk_count} ({100*high_risk_count/len(metrics_df):.1f}%)")
print(f"  Moderate risk golfers:       {moderate_risk_count} ({100*moderate_risk_count/len(metrics_df):.1f}%)")
print(f"  Low risk golfers:            {low_risk_count} ({100*low_risk_count/len(metrics_df):.1f}%)")

# Calculate potential revenue impact
avg_swing_speed_improvement = 3.5  # mph for amateurs with coaching
yards_gained_per_mph = 2.5
total_yards_gained = avg_swing_speed_improvement * yards_gained_per_mph
print(f"\n  Estimated improvement potential for amateur golfers:")
print(f"    Average swing speed gain:  +{avg_swing_speed_improvement:.1f} mph")
print(f"    Estimated distance gain:   +{total_yards_gained:.0f} yards")

# Save sample SHAP explanation
print("\n[7] Generating sample SHAP explanation...")
sample_explanation = explain_golfer(
    df[xgb_artifacts['feature_names']].dropna().iloc[0].values,
    xgb_artifacts['model'],
    xgb_artifacts['scaler'],
    xgb_artifacts['feature_names'],
    golfer_id=1
)
with open('outputs/reports/sample_shap_explanation.txt', 'w', encoding='utf-8') as f:
    f.write(sample_explanation)
print("    Saved: outputs/reports/sample_shap_explanation.txt")

print("\n" + "=" * 70)
print("ALL RESULTS GENERATED SUCCESSFULLY")
print("=" * 70)
print("\nOutput files created:")
print("  - outputs/figures/metric_distributions.png")
print("  - outputs/figures/metric_correlations.png")
print("  - outputs/figures/rf_feature_importance.png")
print("  - outputs/reports/model_comparison.csv")
print("  - outputs/reports/sample_shap_explanation.txt")
