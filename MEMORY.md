# GolfBioMetrics — Project Memory & Context File

**Purpose:** Complete project state capture for continuity across sessions  
**Instruction:** Read this file at the start of every new session  
**Last Updated:** June 18, 2026

---

## Project Status: PRODUCTION READY

| Milestone | Status | Detail |
|-----------|--------|--------|
| Synthetic dataset | Done | 500 swings, 626,025 frames, 0 NaN |
| Biomechanics metrics | Done | 7 core metrics, all research-validated |
| Feature engineering | Done | 58 production features across 4 domains |
| ML models — trained | Done | 5 models, R² 0.85–0.96 |
| Data validation | Done | 0 NaN, 0 infinity, 0 duplicates |
| Notebook narrative | Done | All 6 notebooks have markdown after every code cell |
| SHAP explainability | Done | 3-profile comparison in notebook 04 |
| DT/SVM leakage fix | Done | Source column excluded, realistic accuracy |
| Master presentation | Done | 18-slide PPT + full MD — navy/gold professional design, ML results + Translation Layer |
| GitHub push | Done | June 18, 2026 |
| **Translation Layer** | **Done** | **ML → Human reports for Golfers, Coaches, Fitters** |

---

## Project Identity

- **Project:** GolfBioMetrics
- **Client:** Data Sports Group (DSG) via Ascendium
- **Goal:** POC → production-ready golf swing analysis system
- **Author:** Nazmul Farooquee (nazmulfarooquee@gmail.com)
- **Phase:** Deployment Ready — Awaiting DSG Approval

---

## ML Model Performance (Corrected — June 18, 2026)

| Model | Target | Metric | Value | Notes |
|-------|--------|--------|-------|-------|
| Linear Regression | Ball speed | R² | 0.85 | 58 production features |
| Decision Tree | Swing quality | CV-10 Acc | 82% | Source col excluded |
| Random Forest | Carry distance | R² | 0.83 | RMSE = 9.2 yds |
| XGBoost + SHAP | Injury risk | R² | 0.96 | RMSE = 0.052 |
| SVM | Efficient/Inefficient | Accuracy | 92% | AUC-ROC = 0.979 |

### Critical Fix Applied (June 18, 2026)

The Decision Tree and SVM previously reported 100% accuracy because `kinematic_sequence_score` — the column used to derive their labels — was included in the feature set. This was circular and not predictive.

**Fix:** `label_source_col` parameter added to both `train_decision_tree()` and `train_svm()`. The source column is now excluded automatically. Decision Tree CV-10 = 82%, SVM test = 92% reflect genuine generalisation.

Files changed:
- `src/models/decision_tree.py` — added `label_source_col`, `max_depth=8`, `min_samples_leaf=10`
- `src/models/svm_model.py` — added `label_source_col`

---

## Feature Architecture (58 Production Features)

| Domain | Count | Key Features |
|--------|-------|-------------|
| Biomechanics | 25 | kinematic_sequence_score, xfactor_degrees, lag_angle_mid_downswing, swing_tempo_ratio, compensatory flags, confidence scores |
| Demographics | 10 | age_capability_factor, experience_engrainment, career_stage, physical_profile_score |
| Environmental | 15 | wind_headwind_component, air_density_factor, temperature_efficiency, circadian_factor, env_difficulty_index |
| Time-Series | 8 | swing_dominant_frequency, seq_consistency, session_fatigue, speed_trend |

---

## Notebook State (June 18, 2026)

All 6 notebooks have been enriched with markdown narrative cells between every code block:

| Notebook | Cells (md/code) | Key addition |
|----------|----------------|-------------|
| 01_data_generation | 7 md / 6 code | Narrative explaining each output and what it means |
| 02_metric_computation | 5 md / 6 code | Already had good narrative — unchanged |
| 03_feature_engineering | 6 md / 5 code | Explains derived features and correlation structure |
| 04_model_training | 9 md + 3 SHAP / 9 code | DT/SVM fix explained; SHAP 3-profile section added |
| 05_validation | 6 md / 5 code | Explains Kruskal-Wallis and Pearson interpretation |
| 06_stakeholder_reports | 5 md / 6 code | Already had good narrative — unchanged |

Notebook 04 now has a dedicated SHAP section (cells 13–15) showing three golfer profiles (low/moderate/high risk) with clinical action scripts.

---

## Presentation Assets (June 18, 2026 — v3 redesign)

Primary deliverable:
- `presentation/GolfBioMetrics_DSG_Master_Presentation.pptx` — 16-slide rebuilt deck
- `presentation/GolfBioMetrics_DSG_Master_Presentation.md` — 35 KB full narrative

Design: white background, deep navy header (#0F2447), amber left stripe + underline (#CA8A04), golf-green for positive indicators. No text overflows header (0–0.62") or footer (7.22"–7.5") zones.

Slide structure (no business pricing or revenue content):
1. Title — stats: 58 features, 5 models, 500 swings, R² 0.96
2. The Problem — competitor comparison table (consumer vs lab solutions)
3. 3-Layer Architecture — pose → biomechanics → ML
4. Dataset — cohort stats + data quality certification
5. 7 Core Biomechanics Metrics — table with elite benchmarks + research source
6. Feature Engineering — 4 domain cards (25/10/15/8 features)
7. ML Models — why interpretable, 5-model selection rationale
8. Model Results — performance summary + 4 key insights
9. SHAP Explainability — 3 golfer profiles (low/moderate/high risk) with action plans
10. Statistical Validation — Kruskal-Wallis + Pearson + benchmark comparison
11. How It Helps: Golfers — plain-English output + example drill
12. How It Helps: Coaches — session summary + lesson planning
13. How It Helps: Sports Medicine — injury risk + SHAP clinical report
14. Competitive Landscape — 8-capability comparison table
15. Next Steps — what exists today + 3 deployment phases
16. Technical Appendix — all 58 feature names + hyperparameters + references

---

## Translation Layer — ML to Human Communication (NEW)

**Status:** ✅ Implemented June 18, 2026

**Location:** `src/translation/`

### What It Does
Converts technical ML outputs (R² = 0.849, feature values) into **actionable golf intelligence** that golfers, coaches, and equipment fitters can understand and use.

### Three Dashboards

#### 1. Golfer Report (`golfer_report.py`)
**For:** Weekend golfers, aspiring players
**Output:** Plain English problem descriptions + specific drill recommendations

**Key Features:**
- Feature → Diagnosis → Drill database (5 core metrics mapped)
- Yards-lost calculation ("You're losing 15 yards due to early release")
- 7-day personalized practice plans
- Expected outcome predictions ("+6° lag angle in 2 weeks")

**Example Output:**
```
⚠️ WHAT WE FOUND:
You're losing 22 yards because of early club release.
(Your lag angle: 16.5° at impact | Tour average: 26°)

📋 YOUR 7-DAY FIX:
Drill: "The Pump Practice"
• 20 slow swings daily, pause at top, feel wrist hinge
• Track progress: Upload swing video on Day 1, 4, 7
• Expected: +5° lag angle = +12 yards distance
```

#### 2. Coach Dashboard (`coach_dashboard.py`)
**For:** PGA professionals, teaching pros
**Output:** Progress tracking, data-driven insights, lesson planning

**Key Features:**
- Multi-session progress charts (X-Factor, sequence, lag angle over time)
- Tour average comparisons ("Student at 38th percentile vs Tour")
- Data-driven coaching insights ("X-Factor improving — add speed work")
- This-session focus recommendations

**Example Output:**
```
💡 INSIGHT: Limited Hip Mobility Identified
X-Factor of 34° suggests hip flexibility limitation. Tour pros average 48°.

📋 LESSON PLAN:
Combine technique work with flexibility training. Chair drill daily + 
recommend yoga/Pilates.

📊 vs TOUR: Student 34.0° | Tour Avg 48° | Gap: 14.0°
```

#### 3. Equipment Fitter (`equipment_fitter.py`)
**For:** Club fitters, equipment manufacturers
**Output:** Shaft specs, loft adjustments, environmental optimizations

**Key Features:**
- Shaft flex algorithm (considers tempo + release + sequence)
- Loft adjustments for elevation/temperature
- Environmental context (Denver altitude: +1° loft)
- Expected dispersion/distance improvements

**Example Output:**
```
⛳ SHAFT RECOMMENDATION:
Current: S flex, ~60g
Recommended: X flex, 65g

Why: Your aggressive tempo (2.3) and late release (26.5°) suggests 
you need more tip stability.

Expected: -15% dispersion, +3 yards
```

### Files Created
```
src/translation/
├── __init__.py                    ← Package marker
├── golfer_report.py               ← Golfer-facing reports
├── coach_dashboard.py             ← Coach progress tracking
├── equipment_fitter.py            ← Equipment recommendations
└── master_translator.py           ← Unified interface
```

### Usage
```python
from src.translation.master_translator import MasterTranslator

translator = MasterTranslator()

# Generate all three reports
golfer_report = translator.for_golfer(swing_data, golfer_id="john_001")
coach_report = translator.for_coach(session_history, student_id="john_001")
equipment_report = translator.for_fitter(swing_data, env_data, golfer_id="john_001")
```

### Demo
Run: `python src/translation/master_translator.py`

---

## Data Assets

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| golf_swing_frames.csv | 626,025 | 18+ | Frame-level keypoints — excluded from git |
| golf_swing_metrics.csv | 500 | 40+ | Swing-level raw metrics |
| feature_matrix_production.csv | 500 | 67 | 58 features + id/target columns |
| data_cleaning_report.json | — | — | Validation audit |

Skill distribution: Elite 150, Semi-Pro 150, Amateur 150, Edge cases 50

Key statistics by cohort:

| Cohort | Ball Speed | Carry | Kin. Seq. | X-Factor | Injury Risk |
|--------|-----------|-------|-----------|----------|-------------|
| Elite | 94.9 mph | 159 yds | 0.907 | 47.5° | 0.019 |
| Semi-Pro | 82.1 mph | 137 yds | 0.751 | 37.5° | 0.131 |
| Amateur | 66.7 mph | 111 yds | 0.530 | 22.5° | 0.311 |
| Edge | 60–82 mph | 100–133 yds | < 0.53 | 10°–25° | 0.637–0.715 |

---

## Critical Coding Principles

```python
# 1. Clip dot products before arccos
dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
angle = np.degrees(np.arccos(dot))

# 2. Use atan2 for signed angles
angle_signed = np.degrees(np.arctan2(cross, dot))

# 3. Every metric returns (value, confidence) tuple
def metric_function(...) -> tuple[float, float]:
    return value, confidence

# 4. Add epsilon to denominators
ratio = numerator / (denominator + 1e-8)

# 5. Separate computation from IO
def compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    pass  # pure computation, no file reading

# 6. Exclude label-source column from classifier features
feature_cols = [c for c in feature_cols if c != label_source_col]
```

---

## Session Continuity

**To resume any session:**
1. Read this MEMORY.md
2. Run: `python -c "import pandas as pd; df = pd.read_csv('data/synthetic/feature_matrix_production.csv'); print(f'Features: {len(df.columns)}, Swings: {len(df)}, NaN: {df.isnull().sum().sum()}')"`
3. Expected output: `Features: 67, Swings: 500, NaN: 0`
4. Check `outputs/model_artifacts/` has 5 `.pkl` files

**Next steps if continuing:**
- Real data collection: 50 swings per beta coach
- API wrapper around inference pipeline
- Coach dashboard MVP (web)
- Equipment manufacturer A/B test demo

---

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| No deep learning | DSG mandate: no black-box models |
| Synthetic data for POC | No real data at POC stage — validated against published ranges |
| 5 interpretable models | One per use case: coach / triage / distance / injury / screen |
| SHAP for explainability | Only method that gives per-feature, per-golfer contributions |
| Exclude label-source from classifiers | Prevents trivial circular accuracy |
| 58 features (not 84) | 26 columns removed: 19 constant, 7 leakage-risk |

---

*End of Memory File — June 18, 2026*
