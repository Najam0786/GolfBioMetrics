# GolfBioMetrics — Master Plan

**Project Name:** GolfBioMetrics  
**Version:** 1.0  
**Objective:** Design, implement, and validate biomechanically meaningful metrics from golf swing motion data  
**Status:** Proof of Concept (POC)  
**Client:** Data Sports Group (DSG) via Ascendium  
**Author:** Nazmul Farooquee  
**Last Updated:** June 15, 2026  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Architecture](#3-solution-architecture)
4. [Synthetic Data Design](#4-synthetic-data-design)
5. [Biomechanics Metrics](#5-biomechanics-metrics)
6. [Feature Engineering](#6-feature-engineering)
7. [ML Model Strategy](#7-ml-model-strategy)
8. [Stakeholder Segments](#8-stakeholder-segments)
9. [Validation & Success Criteria](#9-validation--success-criteria)
10. [Code Architecture](#10-code-architecture)
11. [Presentation Structure](#11-presentation-structure-6-pages)
12. [Memory & Decisions Log](#12-memory--decisions-log)

---

## 1. Executive Summary

### What GolfBioMetrics Does

Golf analytics today measure **outcomes** (ball speed, launch angle, carry distance) but not **causes** (swing biomechanics). Coaches can see that a golfer hit the ball 240 yards but not *why* — was it poor kinetic sequencing? Early lag release? Limited hip rotation?

GolfBioMetrics bridges this gap by computing **biomechanically valid, geometrically correct, and interpretable metrics** from human motion data — video or sensor input — that:

- Explain swing mechanics with scientific rigor
- Correlate with measurable performance outcomes
- Are actionable for coaches, athletes, and analysts
- Support injury risk identification
- Enable equipment manufacturers to validate new designs

### What This POC Demonstrates

1. A complete synthetic data pipeline for golf swing biomechanics
2. Seven core biomechanics metrics computed using pure geometry and physics
3. Five interpretable ML models correlating metrics with performance outcomes
4. Statistical validation showing metrics discriminate between skill levels
5. Confidence scoring for unreliable or noisy measurements
6. A clear business case across four stakeholder segments

### Key Design Principle

> **"Scientific correctness first, ML second."**  
> Metrics are computed deterministically from geometry — not learned from data.  
> ML models explain and predict — they do not replace biomechanical reasoning.  
> This directly addresses DSG's requirement: *"not black-box modeling."*

---

## 2. Problem Statement

### 2.1 What DSG Explicitly Needs (From JD)

> *"Design, implement, and own biomechanically valid metrics from human motion and sensor data. This role focuses on scientific correctness, geometric reasoning, and interpretable metrics, not black-box modeling. This also involves developing mathematical models including Geometric, Bayesian, Causality, Personalization and other extending to machine learning models."*

### 2.2 The Five Core Problems to Solve

#### Problem 1: Biomechanical Interpretability Gap
Current technology tells you *what* happened (ball speed = 145 mph) but not *why* (poor kinetic sequencing, weak lag angle, compensatory trunk rotation). Coaches need causal explanations, not outcome scores.

#### Problem 2: Metric Validation Gap
No standardized commercial biomechanics metrics exist for golf at scale. Research labs have them, but they are not operationalized, portable, or coach-friendly.

#### Problem 3: Sensor Noise & Reliability
Markerless pose estimation introduces geometric noise and confidence uncertainty. Every metric must include confidence scoring and handle missing or unreliable keypoints gracefully.

#### Problem 4: Skill-Level Discrimination
Metrics must meaningfully discriminate between amateur and professional golfers. A metric that scores identically across skill levels has no diagnostic value.

#### Problem 5: Actionability for End Users
A metric must translate into specific, clear coaching cues:
- **Not useful:** "Your kinematic sequence score is 0.73."
- **Useful:** "Your hips and shoulders peaked simultaneously. Initiate the downswing with the lower body 40ms earlier to build lag."

### 2.3 Why This Matters for the Golf Industry

| Existing Tool | What It Measures | What It Misses |
|--------------|-----------------|----------------|
| TrackMan / Foresight Launch Monitor | Ball speed, launch angle, spin rate, carry distance | Why those numbers were produced |
| Video analysis (V1 Golf, CoachNow) | Swing visuals for coach review | Quantified, reproducible biomechanics metrics |
| Swing sensors (HackMotion wrist sensor) | Wrist angle only | Full-body kinetic chain |
| Research-grade motion capture (Vicon) | Gold-standard full-body kinematics | Too expensive, lab-only, non-portable |

**GolfBioMetrics fills the gap:** portable, affordable, scientifically valid, coach-facing metrics from video or wearable input.

---

## 3. Solution Architecture

### 3.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────┐
│         LAYER 1: MOTION CAPTURE INPUT           │
│   Video (markerless pose) or IMU wearables      │
│   → Output: 3D skeleton keypoints per frame     │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│       LAYER 2: BIOMECHANICS COMPUTATION         │
│   Pure geometry, physics, signal processing     │
│   NO machine learning — fully deterministic     │
│   → Output: 7 interpretable metrics + scores   │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│         LAYER 3: ML OUTCOME PREDICTION          │
│   5 interpretable traditional ML models         │
│   Correlate metrics → performance outcomes      │
│   → Output: Ball speed, accuracy, injury risk   │
└─────────────────────────────────────────────────┘
```

### 3.2 Why This Architecture?

- **Layer 1** is a solved problem: MediaPipe, OpenPose, and similar models are production-ready pre-trained pose estimators. We do not train our own.
- **Layer 2** is deterministic and explainable: required by DSG's mandate against black-box modeling. Every calculation can be audited and explained geometrically.
- **Layer 3** bridges metrics to real performance outcomes: this validates that our metrics actually matter for predicting distance, accuracy, and injury risk.

### 3.3 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Pose estimation | MediaPipe (pre-trained) | Free, fast, 33-keypoint skeleton, runs on standard video |
| Numerical computation | NumPy, SciPy | Fast array operations for 3D geometry |
| Signal processing | SciPy (filtfilt, signal) | Zero-phase filtering, event detection |
| ML models | Scikit-learn, XGBoost | Interpretable models with feature importance |
| Visualization | Matplotlib, Plotly, Seaborn | Static and interactive metric visualizations |
| Data management | Pandas | Swing-level and frame-level data structures |
| Statistical testing | SciPy stats, Pingouin | Metric validation, group comparison |
| Environment | Python 3.10+, Jupyter Notebook + .py scripts | Notebook for exploration, scripts for production |

---

## 4. Synthetic Data Design

### 4.1 Why Synthetic Data?

For this POC we cannot rely on real golf swing datasets because they are:
- Small in size (most research studies have <50 golfers)
- Proprietary and not publicly available
- Inconsistent in format and sensor type
- Missing ground-truth biomechanics labels

Synthetic data allows us to:
- Control ground truth perfectly (we know the exact angles and timings)
- Generate realistic variation across skill levels
- Inject known edge cases (compensatory patterns, occlusion, noise)
- Validate that our metric computation code returns correct values
- Focus compute and time on metric design, not data collection

### 4.2 Skeleton Definition

We use the **MediaPipe 33-keypoint skeleton**, simplified to 23 keypoints relevant for golf:

```
Torso:    nose, left_shoulder, right_shoulder, left_hip, right_hip
Arms:     left_elbow, right_elbow, left_wrist, right_wrist
Legs:     left_knee, right_knee, left_ankle, right_ankle
Hands:    left_index, right_index (fingertip approximation for grip)
Club:     club_midpoint, club_head (synthetic additional keypoints)
```

Each keypoint is represented as a 3D coordinate: (x, y, z) in metres, plus a confidence score (0–1).

### 4.3 Swing Phase Definition

| Phase | Start Event | End Event | Duration (Elite) | Duration (Amateur) |
|-------|------------|-----------|-----------------|-------------------|
| Address | Static stance | First clubhead movement | 0.0–0.1s | 0.0–0.2s |
| Backswing | First movement | Club reaches top | 0.1–0.6s | 0.1–0.8s |
| Transition | Top of backswing | First downswing movement | 0.6–0.7s | 0.8–1.0s |
| Downswing | First downswing movement | Ball impact | 0.7–1.0s | 1.0–1.4s |
| Impact | Ball contact | ~5ms window | 1.0s | 1.4s |
| Follow-through | Post-impact | Finish position | 1.0–1.8s | 1.4–2.2s |

### 4.4 Synthetic Motion Generation

We model each joint angle over time using **parametric kinematic equations**:

```python
# Example: pelvis rotation during downswing
def pelvis_rotation(t, t_peak, peak_angle, swing_speed_factor):
    """
    Models pelvis rotation as a smooth bell-shaped acceleration/deceleration curve.
    t: current time (seconds)
    t_peak: time of peak angular velocity
    peak_angle: maximum rotation angle (degrees)
    swing_speed_factor: 1.0 for elite, 0.75 for amateur
    """
    sigma = 0.08 * swing_speed_factor  # width of bell curve
    return peak_angle * np.exp(-((t - t_peak) ** 2) / (2 * sigma ** 2))

# Kinetic chain: pelvis leads thorax leads arm leads club
# Each segment peaks ~20-40ms after its proximal neighbour (elite)
# Elite: t_pelvis < t_thorax < t_arm < t_club (each ~30ms apart)
# Amateur: segments peak simultaneously or in wrong order
```

### 4.5 Golfer Population Parameters

```
Total synthetic swings:     500
├── Elite golfers:          150 swings (n=15 golfers × 10 swings each)
│   ├── Swing speed:        105–120 mph clubhead speed
│   ├── X-Factor:           40–55° at top of backswing
│   ├── Sequence quality:   0.85–0.98
│   └── Compensations:      None or minimal
│
├── Semi-pro golfers:       150 swings (n=15 golfers × 10 swings each)
│   ├── Swing speed:        90–105 mph clubhead speed
│   ├── X-Factor:           30–45° at top of backswing
│   ├── Sequence quality:   0.65–0.85
│   └── Compensations:      Occasional (1–2 per swing)
│
├── Amateur golfers:        150 swings (n=15 golfers × 10 swings each)
│   ├── Swing speed:        70–90 mph clubhead speed
│   ├── X-Factor:           15–30° at top of backswing
│   ├── Sequence quality:   0.40–0.65
│   └── Compensations:      Frequent (2–4 per swing)
│
└── Edge cases:             50 swings
    ├── Extreme noise (simulating poor camera conditions)
    ├── Missing keypoints (occlusion during fast downswing)
    ├── Unusual body types (tall, short, left-handed mirrored)
    └── Severe compensation patterns (injury-like movement)
```

### 4.6 Synthetic Performance Outcomes (Labels)

For each swing we compute synthetic ground-truth outcome labels:

```python
# Ball speed prediction from biomechanics
ball_speed_mph = (
    0.82 * clubhead_speed_mph        # Smash factor component
    + 0.15 * lag_angle_score * 10    # Lag contribution
    + 0.03 * xfactor_degrees         # X-Factor contribution
    - 0.05 * compensation_count * 5  # Compensation penalty
    + np.random.normal(0, 2)         # Realistic noise
)

# Carry distance from ball speed
carry_distance_yards = ball_speed_mph * 1.67  # Approximate ratio for driver

# Accuracy: offline distance (lower = more accurate)
offline_yards = (
    2.0 + (1.0 - club_path_consistency) * 15
    + compensation_count * 3
    + np.random.normal(0, 1.5)
)

# Injury risk score (0-1)
injury_risk = (
    0.3 * reverse_pivot_severity
    + 0.25 * excessive_sway_severity
    + 0.25 * early_extension_severity
    + 0.20 * trunk_tilt_change
)
```

### 4.7 Output Data Schema

```
Frame-level CSV (golf_swing_frames.csv):
[swing_id, frame_id, timestamp_s, golfer_id, skill_level,
 keypoint_name, x, y, z, confidence]

Swing-level CSV (golf_swing_metrics.csv):
[swing_id, golfer_id, skill_level, club_type,
 kinematic_sequence_score, kinematic_sequence_confidence,
 lag_angle_mid_downswing, lag_angle_impact, lag_confidence,
 xfactor_degrees, xfactor_confidence,
 weight_transfer_timing_ms, weight_transfer_confidence,
 club_path_consistency, club_path_confidence,
 swing_tempo_ratio, tempo_confidence,
 early_cast_flag, reverse_pivot_flag, sway_flag, trunk_tilt_flag,
 compensation_count, compensation_severity,
 ball_speed_mph, carry_distance_yards, offline_yards, injury_risk_score]
```

---

## 5. Biomechanics Metrics

All metrics are computed using **pure geometry and physics** from skeleton keypoints.  
No machine learning in this layer.

### Metric 1: Kinematic Sequence Quality Score

**What It Measures:**  
How well the golfer follows the proximal-to-distal energy transfer pattern during the downswing: Pelvis → Thorax → Lead Arm → Club.

**Scientific Basis:**  
Research (Sheffield Hallam University, TPI Biomechanics, ASMI) consistently shows that elite golfers peak each segment's angular velocity in strict proximal-to-distal order, each ~20–50ms apart. This pattern multiplies energy at the clubhead like a whip crack.

**Computation:**
```python
def kinematic_sequence_score(angular_velocities: dict, timestamps: np.ndarray) -> tuple:
    """
    angular_velocities: dict with keys ['pelvis', 'thorax', 'arm', 'club']
                        each value is a 1D array of angular velocity over time
    timestamps: array of time values (seconds)
    Returns: (score: float 0-1, confidence: float 0-1, timing_gaps_ms: dict)
    """
    segments = ['pelvis', 'thorax', 'arm', 'club']
    peak_times = {}

    for seg in segments:
        # Find peak angular velocity during downswing phase only
        peak_idx = np.argmax(angular_velocities[seg])
        peak_times[seg] = timestamps[peak_idx]

    # Score: +0.25 for each correct ordering pair
    correct_order = 0
    pairs = [('pelvis','thorax'), ('thorax','arm'), ('arm','club')]
    for a, b in pairs:
        if peak_times[a] < peak_times[b]:
            correct_order += 1

    # Bonus: timing gaps should be 20-50ms each (elite characteristic)
    timing_gaps = {f"{a}_to_{b}": (peak_times[b] - peak_times[a]) * 1000
                   for a, b in pairs}
    gap_score = np.mean([1.0 if 15 <= gap <= 60 else 0.5
                         for gap in timing_gaps.values()])

    score = (correct_order / 3) * 0.7 + gap_score * 0.3
    return score, timing_gaps
```

**Validation Range:**  
- Amateur: 0.40–0.65  
- Semi-pro: 0.65–0.85  
- Elite: 0.85–0.98  

**Coaching Cue:**  
"Your thorax peaked before your pelvis. Start the downswing with lower body rotation. Target: pelvis leads by 30ms."

---

### Metric 2: Lag Angle

**What It Measures:**  
The angle between the forearm and club shaft during downswing. Measures stored energy (lag maintenance) and energy release (lag release at impact).

**Scientific Basis:**  
The wrist cock (lag angle) stores potential energy during the downswing. Releasing it too early (casting) loses 10–25 mph ball speed. Elite golfers maintain 70–90° lag until the hands reach hip height, then release to ~20–30° at impact.

**Computation:**
```python
def lag_angle(wrist_pos: np.ndarray, elbow_pos: np.ndarray,
              club_head_pos: np.ndarray) -> float:
    """
    All positions are 3D vectors (x, y, z).
    Returns lag angle in degrees.
    """
    # Forearm vector: wrist to elbow
    v_forearm = elbow_pos - wrist_pos
    v_forearm = v_forearm / np.linalg.norm(v_forearm)

    # Club vector: wrist to club head
    v_club = club_head_pos - wrist_pos
    v_club = v_club / np.linalg.norm(v_club)

    # Angle between vectors (always positive 0-180°)
    dot = np.clip(np.dot(v_forearm, v_club), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))
```

**Note:** The `np.clip` is critical — floating point errors can push dot product outside [-1, 1], causing arccos to return NaN.

**Validation Range:**  
- Mid-downswing: 70–90° (elite), 40–65° (amateur with early cast)  
- At impact: 20–35° (elite), 5–15° (amateur — over-released)  

**Coaching Cue:**  
"You're releasing the lag at hip height instead of impact. Hold the wrist angle for 40ms longer."

---

### Metric 3: Hip-Shoulder Separation (X-Factor)

**What It Measures:**  
The rotational angular difference between pelvis and thorax in the transverse (horizontal) plane. Also called the X-Factor. Represents the coil tension available for power generation.

**Scientific Basis:**  
Studies (Cheetham et al., McTeigue et al.) show X-Factor at the top of the backswing correlates strongly with clubhead speed. Elite golfers average 40–50°; amateurs average 15–30°.

**Computation:**
```python
def xfactor(left_hip: np.ndarray, right_hip: np.ndarray,
            left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> float:
    """
    Computes hip-shoulder separation in the transverse plane.
    All positions are 3D vectors (x, y, z).
    Returns X-Factor angle in degrees.
    """
    # Pelvis axis vector (projected onto horizontal plane)
    pelvis_axis = right_hip - left_hip
    pelvis_axis[1] = 0  # zero out vertical component → project to horizontal
    pelvis_axis = pelvis_axis / np.linalg.norm(pelvis_axis)

    # Thorax axis vector (projected onto horizontal plane)
    thorax_axis = right_shoulder - left_shoulder
    thorax_axis[1] = 0
    thorax_axis = thorax_axis / np.linalg.norm(thorax_axis)

    # Signed angle between axes using atan2
    cross = pelvis_axis[0] * thorax_axis[2] - pelvis_axis[2] * thorax_axis[0]
    dot = np.dot(pelvis_axis, thorax_axis)
    return float(np.degrees(np.arctan2(cross, dot)))
```

**Note:** We use `atan2(cross, dot)` instead of `arccos(dot)` because:
- `atan2` gives a signed angle (distinguishes direction)
- `atan2` is numerically stable near 0° and 180°

**Validation Range:**  
- Top of backswing: 40–55° (elite), 15–30° (amateur)  
- At impact: 20–40° (elite — hips have rotated through, shoulders catching up)  

**Coaching Cue:**  
"Your X-Factor is 22° — 18° below elite average. You need more shoulder turn or less hip turn during backswing."

---

### Metric 4: Weight Transfer Timing

**What It Measures:**  
When the golfer's centre of mass shifts from back foot to front foot, relative to the start of the downswing. Timing of weight transfer directly affects kinetic chain efficiency.

**Scientific Basis:**  
Weight should begin transferring to the front foot at or slightly before downswing initiation. Transferring too early (sliding) or too late (hanging back) disrupts sequencing.

**Computation:**
```python
def weight_transfer_timing(
    keypoints_over_time: np.ndarray,  # shape: (frames, 23, 3)
    timestamps: np.ndarray,
    left_ankle_idx: int,
    right_ankle_idx: int,
    downswing_start_frame: int
) -> dict:
    """
    Returns timing of weight transfer relative to downswing initiation.
    Positive = after downswing start (late), Negative = before (early).
    Optimal: -80 to -120ms (weight starts shifting just before downswing).
    """
    # Approximate COM as mean of all keypoints (simplified)
    com_x = np.mean(keypoints_over_time[:, :, 0], axis=1)

    # Back foot and front foot x positions (for right-handed golfer)
    back_foot_x = keypoints_over_time[:, right_ankle_idx, 0]
    front_foot_x = keypoints_over_time[:, left_ankle_idx, 0]

    # Normalized COM position between feet (0=back foot, 1=front foot)
    com_normalized = (com_x - back_foot_x) / (front_foot_x - back_foot_x + 1e-8)

    # Find frame where COM crosses 0.5 (equal weight distribution = transfer point)
    transfer_frame = np.argmax(com_normalized > 0.5)
    transfer_time_ms = (timestamps[transfer_frame] -
                        timestamps[downswing_start_frame]) * 1000

    return {
        'transfer_time_ms': transfer_time_ms,
        'is_early': transfer_time_ms > -50,     # Before weight transfer should happen
        'is_late': transfer_time_ms < -200,      # Too late, hanging back
        'is_optimal': -150 <= transfer_time_ms <= -50
    }
```

**Validation Range:**  
- Optimal: -80 to -120ms (before downswing initiation)  
- Amateur sway: transfer occurs during backswing (>+200ms = wrong direction)  
- Amateur hang-back: transfer occurs after impact (<-200ms)  

---

### Metric 5: Club Path Consistency

**What It Measures:**  
How consistently the club follows a single swing plane throughout the downswing. Deviation from the plane predicts shot direction inconsistency.

**Computation:**
```python
def club_path_consistency(
    club_positions: np.ndarray  # shape: (frames, 3) - x,y,z during downswing
) -> dict:
    """
    Fits a plane to club positions using SVD.
    Returns consistency score and swing plane normal vector.
    """
    # Centre the data
    centroid = np.mean(club_positions, axis=0)
    centred = club_positions - centroid

    # SVD to find best-fit plane (normal = third singular vector)
    _, _, vh = np.linalg.svd(centred)
    plane_normal = vh[-1]  # Normal to best-fit plane

    # Compute distances from each point to the plane
    distances = np.abs(centred @ plane_normal)
    rms_deviation_cm = np.sqrt(np.mean(distances ** 2)) * 100  # Convert to cm

    # Consistency score: 1.0 = perfect plane, decreasing with deviation
    consistency = max(0.0, 1.0 - (rms_deviation_cm / 10.0))

    return {
        'consistency_score': consistency,
        'rms_deviation_cm': rms_deviation_cm,
        'swing_plane_normal': plane_normal
    }
```

**Validation Range:**  
- Elite: 0.92–0.99 (≤1cm deviation)  
- Semi-pro: 0.80–0.92 (1–2cm deviation)  
- Amateur: 0.60–0.80 (2–5cm deviation)  

---

### Metric 6: Compensatory Pattern Detection

**What It Measures:**  
Presence and severity of biomechanically inefficient movement patterns that reduce performance or increase injury risk.

| Pattern | Definition | Injury Risk | Performance Impact |
|---------|-----------|------------|-------------------|
| Early arm cast | Elbow extends before hip peak velocity | Elbow / wrist | -10 to -25 mph ball speed |
| Reverse pivot | Weight on front foot at top of backswing | Lumbar spine | Loss of power, inconsistency |
| Lateral sway | COM moves >20cm laterally during backswing | Lower back, hip | Sequence disruption |
| Early extension | Hips thrust toward ball during downswing | Lower back | Inconsistent contact |
| Over-the-top | Club path crosses inside-to-outside | Shoulder | Slices, pulls |

**Computation (Example: Early Cast Detection):**
```python
def detect_early_cast(
    lag_angles: np.ndarray,  # lag angle at each downswing frame
    hip_angular_velocity: np.ndarray,
    timestamps: np.ndarray
) -> dict:
    """
    Early cast: lag angle decreases BEFORE pelvis reaches peak velocity.
    """
    hip_peak_time = timestamps[np.argmax(hip_angular_velocity)]
    lag_release_start = timestamps[np.argmax(np.diff(lag_angles) < -2)]

    is_early_cast = lag_release_start < hip_peak_time
    severity = max(0.0, (hip_peak_time - lag_release_start) / 0.1)  # 0-1

    return {
        'early_cast_detected': is_early_cast,
        'severity': min(1.0, severity),
        'timing_error_ms': (hip_peak_time - lag_release_start) * 1000
    }
```

---

### Metric 7: Swing Tempo & Rhythm

**What It Measures:**  
The ratio of backswing duration to downswing duration. Tempo ratio is a fundamental measure of swing rhythm and repeatability.

**Computation:**
```python
def swing_tempo(
    t_address: float,
    t_transition: float,
    t_impact: float
) -> dict:
    """
    Computes backswing-to-downswing time ratio.
    Optimal ratio: 2.0–3.0 for most golfers.
    """
    backswing_time = t_transition - t_address
    downswing_time = t_impact - t_transition
    tempo_ratio = backswing_time / (downswing_time + 1e-8)

    return {
        'backswing_time_s': backswing_time,
        'downswing_time_s': downswing_time,
        'tempo_ratio': tempo_ratio,
        'is_rushing': tempo_ratio < 1.8,  # Downswing too fast relative to backswing
        'is_optimal': 2.0 <= tempo_ratio <= 3.0
    }
```

**Validation Range:**  
- Elite average: 2.3–2.8  
- Amateur (rushing): 1.2–1.8  
- Amateur (slow): 3.5–5.0 (overthinking, tension)

---

## 6. Feature Engineering

### 6.1 Frame-Level to Swing-Level Aggregation

Each swing has 30–60 frames. We aggregate to a single swing-level feature vector:

```
For each metric computed frame-by-frame, extract:
├── Peak value (e.g., max X-Factor during downswing)
├── Value at specific event (e.g., lag angle AT impact frame)
├── Rate of change (e.g., how fast X-Factor changes at transition)
├── Phase-specific mean (e.g., mean club path deviation during downswing)
└── Timing of peak relative to impact (e.g., when pelvis peaks in ms before impact)
```

### 6.2 Derived Features (Engineered from Core Metrics)

| Feature | Derivation | Interpretation |
|---------|-----------|---------------|
| Sequence efficiency index | kinematic_score × (1 - compensation_severity) | Combined quality index |
| Power potential score | xfactor × lag_angle_mid × weight_transfer_ok | Available stored energy |
| Release efficiency | lag_at_mid / lag_at_impact | How well lag is maintained and released |
| Tempo-sequence alignment | tempo_ratio × kinematic_sequence_score | Rhythm and sequencing combined |
| Reliability score | Mean of all confidence scores | How much to trust this swing's metrics |

### 6.3 Feature Matrix for ML Models

```
Input feature matrix X (one row per swing, 500 rows × 20 features):

Core Metrics (7):
- kinematic_sequence_score
- lag_angle_mid_downswing
- lag_angle_impact
- xfactor_top_backswing
- weight_transfer_timing_ms
- club_path_consistency
- swing_tempo_ratio

Compensation Flags (5):
- early_cast_severity
- reverse_pivot_severity
- sway_severity
- early_extension_severity
- over_top_severity

Engineered Features (8):
- sequence_efficiency_index
- power_potential_score
- release_efficiency
- tempo_sequence_alignment
- backswing_duration
- downswing_duration
- xfactor_at_impact
- lag_release_rate

Target Variables y (4 options depending on model):
- ball_speed_mph           → Regression
- carry_distance_yards     → Regression
- swing_quality_class      → Classification (Poor / Average / Good / Elite)
- injury_risk_score        → Regression
```

---

## 7. ML Model Strategy

### 7.1 Why Traditional ML (Not Deep Learning)

DSG explicitly requires: *"not black-box modeling."*

| Approach | Interpretability | Why For/Against |
|---------|-----------------|-----------------|
| CNN / Deep Learning | ❌ Black-box | Violates DSG mandate. Cannot explain WHY a swing is rated highly. |
| Linear Regression | ✅ Fully transparent | Coefficients directly show feature contribution |
| Decision Tree | ✅ Rule-based | Coaches can follow the logic: "IF X-Factor > 40 AND lag > 70 THEN..." |
| Random Forest | ✅ Feature importance | Ensemble handles non-linearity while staying interpretable |
| XGBoost | ✅ SHAP values | State-of-the-art accuracy + explainability via SHAP |
| SVM | ✅ Margin-based | Good for outlier robustness in classification tasks |

**Note on Pose Estimation (Layer 1):** A pre-trained CNN (MediaPipe) is used only to extract keypoints from video. This is a solved engineering problem — not where DSG needs interpretability. The scientific ownership begins at Layer 2 (metric computation).

---

### Model 1: Linear Regression

**Objective:** Predict ball speed (mph) from biomechanics metrics.

**Why:** Baseline model. Coefficients show direct relationships with physical meaning.

**Expected Output:**
```
Ball Speed (mph) = 
  85.2                          # baseline (average amateur)
+ 12.4 × kinematic_seq_score   # sequencing quality
+ 0.18 × xfactor_degrees       # hip-shoulder coil
+ 0.22 × lag_mid_downswing     # energy storage
- 8.3 × early_cast_severity    # penalty for casting
- 5.1 × reverse_pivot_severity # penalty for reverse pivot
+ ε                             # residual
```

**Metrics:** R², RMSE, MAE, coefficient p-values  
**Expected Performance:** R² ≈ 0.75–0.85

---

### Model 2: Decision Tree Classifier

**Objective:** Classify swing quality into four categories: Poor / Average / Good / Elite.

**Why:** Decision rules are directly interpretable by coaches. Can be printed as a flowchart.

**Expected Rules:**
```
IF kinematic_sequence_score >= 0.85:
    IF xfactor_degrees >= 40:
        → ELITE (predicted ball speed >110 mph)
    ELSE:
        → GOOD (predicted ball speed 95–110 mph)
ELIF kinematic_sequence_score >= 0.65:
    → AVERAGE (predicted ball speed 80–95 mph)
ELSE:
    → POOR (predicted ball speed <80 mph)
```

**Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix  
**Expected Performance:** Accuracy ≈ 0.82–0.88

---

### Model 3: Random Forest Regressor

**Objective:** Predict carry distance (yards) from full feature set.

**Why:** Captures non-linear interactions between features. Feature importance shows which metrics matter most for distance.

**Key Output: Feature Importance Ranking**
```
Expected feature importance order:
1. kinematic_sequence_score    (~28%)
2. lag_angle_mid_downswing     (~22%)
3. xfactor_top_backswing       (~18%)
4. early_cast_severity         (~12%)
5. weight_transfer_timing      (~8%)
6. club_path_consistency       (~6%)
7. swing_tempo_ratio           (~4%)
8. Other features              (~2%)
```

**Metrics:** R², RMSE, Feature Importance Plot  
**Expected Performance:** R² ≈ 0.88–0.94

---

### Model 4: XGBoost + SHAP Explainability

**Objective:** Predict injury risk score (0–1) with per-swing explanations.

**Why:** Best predictive performance + SHAP values provide individual-level explanations:
- "This golfer's injury risk is 0.72 — driven primarily by reverse pivot severity (0.31 contribution) and early extension (0.24 contribution)."

**SHAP Output Example:**
```
Golfer ID: 042 | Injury Risk: 0.74 (HIGH)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reverse pivot severity:    +0.31 ↑ (increases risk)
Early extension severity:  +0.24 ↑ (increases risk)
Excessive sway:            +0.18 ↑ (increases risk)
Kinematic sequence score:  -0.08 ↓ (reduces risk — protective)
X-Factor:                  -0.04 ↓ (reduces risk — protective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recommendation: Address reverse pivot pattern immediately.
Risk of lumbar injury is elevated if training volume increases.
```

**Metrics:** AUC-ROC, Brier Score, SHAP Summary Plot, SHAP Waterfall per golfer  
**Expected Performance:** AUC ≈ 0.90–0.95

---

### Model 5: Support Vector Machine (SVM)

**Objective:** Binary classification — Efficient vs Inefficient kinetic sequence.

**Why:** SVM is robust to outliers and works well with smaller, high-dimensional feature sets. Provides a clean boundary between efficient and inefficient swings.

**Use Case:** Fast, binary screening — "Is this swing worth detailed analysis or does it need fundamental correction first?"

**Metrics:** Accuracy, Precision, Recall, F1, Decision Boundary Visualization  
**Expected Performance:** Accuracy ≈ 0.85–0.90

---

### 7.2 Model Comparison Summary

| Model | Target | Type | Key Strength | Key Output |
|-------|--------|------|-------------|-----------|
| Linear Regression | Ball speed (mph) | Regression | Interpretable coefficients | Feature weights |
| Decision Tree | Swing quality class | Classification | Coach-readable rules | Decision flowchart |
| Random Forest | Carry distance (yards) | Regression | Feature importance | Ranking of metric value |
| XGBoost + SHAP | Injury risk score | Regression | Per-golfer explanations | Individual SHAP values |
| SVM | Efficient / Inefficient | Binary classification | Outlier robustness | Quick screening |

---

## 8. Stakeholder Segments

### Segment 1: Golf Coaches & Instructors

**Problem:** Coaches observe swings visually but lack quantified, reproducible metrics to track improvement or diagnose issues precisely.

**What GolfBioMetrics Delivers:**
- Kinematic sequence report per session
- Trend charts across multiple sessions (is X-Factor improving?)
- Automatic flagging of compensatory patterns
- Specific coaching cue generation from metric values

**Example Dashboard Output:**
```
Student: John Smith | Session: 12 June 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kinematic Sequence:  0.71 (+0.08 vs last session) ✅
X-Factor:            31°  (+3° vs last session)    ⚠️ Below elite (40°)
Lag Angle (Impact):  28°  (within optimal range)   ✅
Early Cast:          Mild (severity: 0.32)          ⚠️
Injury Risk:         Low (score: 0.21)              ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Priority for next session:
1. Increase shoulder turn during backswing (+9° target)
2. Monitor early cast pattern — wrist is releasing at hip height
```

---

### Segment 2: Individual Golfers (Amateur & Competitive)

**Problem:** Golfers take lessons but cannot quantify their own improvement between sessions. They rely on feel, which is subjective and unreliable.

**What GolfBioMetrics Delivers:**
- Personal biomechanics baseline on first session
- Progress tracking across sessions
- Personalised recommendations based on their specific swing profile
- Comparison against age/gender/skill-level peer benchmarks (not just pro averages)

**Personalisation Example:**
```
Your Swing Profile: 47-year-old male amateur, 12 handicap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your X-Factor:  27°
Peer average:   24° (you are above average for your group)
Elite target:   40° (ambitious — focus on flexibility first)
Realistic 6-month target: 33° (+6°)

Estimated ball speed gain if target achieved: +4 mph
Estimated distance gain: +8 yards
```

---

### Segment 3: Golf Equipment Manufacturers

**Problem:** Manufacturers (Callaway, TaylorMade, Ping, Titleist) need to validate whether new club designs improve biomechanical efficiency for specific golfer profiles — not just average ball speed.

**What GolfBioMetrics Delivers:**
- Controlled A/B testing: Club A vs Club B across matched golfer profiles
- Biomechanics breakdown: does the new design change kinematic sequence, lag angle, or only outcome metrics?
- Segment-specific analysis: does the new shaft flex benefit low-swing-speed amateurs specifically?

**Example Business Report Output:**
```
Equipment Test: Driver Model X vs Model X-Pro
Test Group: Mid-handicap amateurs (n=30, swing speed 85–95 mph)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric               Model X    Model X-Pro   Δ       p-value
Ball speed (mph)     92.3       95.1          +2.8    p=0.003 ✅
Lag angle (impact)   24.1°      26.8°         +2.7°   p=0.018 ✅
Kinematic seq score  0.71       0.73          +0.02   p=0.210 ❌
X-Factor             29.3°      29.1°         -0.2°   p=0.850 ❌
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conclusion: Model X-Pro improves ball speed and lag maintenance
but does not change fundamental swing mechanics.
Benefit is equipment-driven, not technique-driven.
```

---

### Segment 4: Sports Medicine & Injury Prevention

**Problem:** Golf injuries (particularly lower back, elbow, and wrist) are common and costly. Clinicians lack quantified biomechanics data from actual on-course conditions to identify risk before injury occurs.

**What GolfBioMetrics Delivers:**
- Individual injury risk scores per swing session
- Identification of specific compensatory patterns linked to injury risk
- Workload monitoring: how does biomechanics quality change as golfer fatigues across a round?
- Return-to-play benchmarking: has the golfer's movement pattern returned to pre-injury baseline?

---

## 9. Validation & Success Criteria

### 9.1 Metric Validation (Layer 2)

**Synthetic Ground Truth Tests:**
```
For each metric, build unit tests using synthetic data with known ground truth:

TEST 1: Perfect elite swing → kinematic_sequence_score should be ≥ 0.90
TEST 2: Reversed sequence → kinematic_sequence_score should be ≤ 0.40
TEST 3: 90° known angle between vectors → lag_angle() should return 90.0 ± 0.001°
TEST 4: Known X-Factor of 45° → xfactor() should return 45.0 ± 0.5°
TEST 5: NaN keypoints → all metrics should return confidence = 0, not crash
TEST 6: Single frame input → metrics should raise ValueError with clear message
```

**Discriminant Validity:**
```
Do metrics distinguish skill levels? (ANOVA / Kruskal-Wallis test)
Expected: p < 0.001 for all primary metrics across skill groups
```

**Correlation with Outcomes:**
```
Pearson correlation matrix: metrics vs ball_speed, carry_distance, offline_yards
Expected:
  kinematic_sequence_score ↔ ball_speed:    r ≥ 0.75
  xfactor_degrees          ↔ ball_speed:    r ≥ 0.65
  lag_angle_mid            ↔ ball_speed:    r ≥ 0.60
  early_cast_severity      ↔ ball_speed:    r ≤ -0.55
```

### 9.2 ML Model Validation (Layer 3)

```
Train/validation/test split: 70% / 15% / 15%
Cross-validation: 5-fold stratified by skill level

Minimum acceptance thresholds:
  Linear Regression:   R² ≥ 0.75 on test set
  Decision Tree:       Accuracy ≥ 0.80 on test set
  Random Forest:       R² ≥ 0.85 on test set
  XGBoost:            AUC-ROC ≥ 0.88 on test set
  SVM:                Accuracy ≥ 0.82 on test set
```

### 9.3 Business Validation

- Can a coach understand the metric without a statistics background?
- Does the system gracefully handle missing or low-confidence data?
- Are coaching cues specific and actionable (not generic)?
- Does the system flag when it cannot reliably compute a metric?

---

## 10. Code Architecture

### 10.1 Project Folder Structure

```
GolfBioMetrics/
│
├── README.md                    # Quick start guide
├── MASTER_PLAN.md              # This document — read first
├── MEMORY.md                   # Decisions log and context
├── SKILLS.md                   # Biomechanics reference library
│
├── data/
│   ├── synthetic/
│   │   ├── golf_swing_frames.csv       # Frame-level keypoints
│   │   └── golf_swing_metrics.csv      # Swing-level aggregated metrics
│   └── real/                           # Placeholder for real DSG data
│
├── src/
│   ├── __init__.py
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── synthetic_swing_generator.py  # Generates synthetic swing data
│   │   └── noise_injection.py            # Adds realistic sensor noise
│   │
│   ├── biomechanics/
│   │   ├── __init__.py
│   │   ├── geometry_utils.py    # 3D vector math, angle calculations
│   │   ├── signal_processing.py # Filtering, event detection, smoothing
│   │   ├── metrics.py           # All 7 core metric computations
│   │   ├── confidence.py        # Confidence scoring for each metric
│   │   └── compensations.py     # Compensatory pattern detection
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── aggregation.py       # Frame-level to swing-level aggregation
│   │   └── engineering.py       # Derived feature computation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linear_regression.py  # Model 1
│   │   ├── decision_tree.py      # Model 2
│   │   ├── random_forest.py      # Model 3
│   │   ├── xgboost_model.py      # Model 4 + SHAP
│   │   └── svm_model.py          # Model 5
│   │
│   └── validation/
│       ├── __init__.py
│       ├── metric_validation.py  # Unit tests for biomechanics metrics
│       └── model_evaluation.py   # ML model evaluation and comparison
│
├── notebooks/
│   ├── 01_data_generation.ipynb     # Generate and explore synthetic data
│   ├── 02_metric_computation.ipynb  # Compute and visualize metrics
│   ├── 03_feature_engineering.ipynb # Build feature matrix
│   ├── 04_model_training.ipynb      # Train and compare all 5 models
│   ├── 05_validation.ipynb          # Statistical validation
│   └── 06_stakeholder_reports.ipynb # Generate stakeholder-specific outputs
│
├── tests/
│   ├── test_geometry_utils.py
│   ├── test_metrics.py
│   ├── test_signal_processing.py
│   └── test_models.py
│
├── outputs/
│   ├── figures/                # All charts and visualizations
│   ├── reports/                # Stakeholder reports
│   └── model_artifacts/        # Saved model files
│
└── requirements.txt
```

### 10.2 Key Coding Principles (From DSG JD)

```
1. SEPARATION OF CONCERNS (mandatory):
   ├── Computation (src/biomechanics/): pure math, no IO, fully testable
   ├── Aggregation (src/features/): applies computation across time series
   └── Orchestration (notebooks/): loads data, calls aggregation, saves results

2. DETERMINISTIC CODE:
   - No random seeds without explicit documentation
   - Same input ALWAYS produces same output
   - No side effects in computation functions

3. TESTABILITY:
   - Every metric function has unit tests with synthetic known-good inputs
   - Edge cases explicitly tested: NaN, zero vectors, empty arrays, single frames

4. NUMERICAL STABILITY:
   - Always clip dot products before arccos: np.clip(dot, -1.0, 1.0)
   - Always use atan2 for signed angles instead of arccos
   - Add epsilon (1e-8) to denominators to prevent division by zero

5. CONFIDENCE PROPAGATION:
   - Every metric returns (value, confidence) tuple
   - Low confidence metrics are flagged, not silently returned
```

### 10.3 Requirements

```
# requirements.txt
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
mediapipe>=0.10.0    # For real video pose estimation
jupyter>=1.0.0
pytest>=7.3.0
pingouin>=0.5.3      # Statistical testing
```

---

## 11. Presentation Structure (6 Pages)

### Page 1: Title & Problem Statement
**Title:** GolfBioMetrics: Biomechanically Valid Metrics for Golf Swing Analytics  
**Subtitle:** A Proof of Concept for Data Sports Group (DSG)

**Content:**
- The gap: current golf analytics measure outcomes, not causes
- The opportunity: bridge swing mechanics → performance outcomes
- The approach: geometric reasoning + interpretable ML, not black-box AI

---

### Page 2: Solution Architecture
**Title:** Three-Layer Scientific Architecture

**Content:**
- Layer 1: Motion capture (pre-trained pose estimation — solved problem)
- Layer 2: Biomechanics computation (pure geometry, deterministic, auditable)
- Layer 3: ML outcome prediction (5 interpretable models, no black-box)
- Visual: architecture diagram showing data flow

---

### Page 3: Core Biomechanics Metrics
**Title:** Seven Scientifically Valid, Coach-Interpretable Metrics

**Content:**
- Table showing all 7 metrics: name, what it measures, coaching cue
- Example: kinematic sequence visualization (pelvis → thorax → arm → club timing chart)
- Confidence scoring system explained
- Validation ranges (amateur vs semi-pro vs elite benchmarks)

---

### Page 4: ML Model Results
**Title:** Five Interpretable Models — Predicting Performance Outcomes

**Content:**
- Model comparison table (accuracy, R², AUC)
- Feature importance chart from Random Forest
- SHAP waterfall plot for one example golfer (injury risk)
- Key finding: kinematic sequence quality is the strongest predictor of ball speed

---

### Page 5: Stakeholder Impact
**Title:** Who Benefits and How

**Content:**
- 2×2 grid: Coaches / Golfers / Manufacturers / Sports Medicine
- One specific example output for each segment
- Revenue model suggestion: SaaS per-seat (coaches), consumer app (golfers), enterprise contracts (manufacturers, clinics)

---

### Page 6: Next Steps & Roadmap
**Title:** From POC to Production

**Content:**
- Phase 1 (POC — done): Synthetic data, metric validation, model training
- Phase 2 (Pilot): Real golfer data capture, metric calibration against launch monitor ground truth
- Phase 3 (Product): API delivery, coach-facing dashboard, mobile app integration
- Open question for DSG: Which customer segment and use case should we prioritize first?

---

## 12. Memory & Decisions Log

This section tracks key decisions made during the project to avoid re-explaining context in future sessions.

### Decisions Made

| Decision | Choice | Rationale |
|---------|--------|-----------|
| Use synthetic data for POC | Yes | No access to real DSG data; synthetic allows ground-truth validation |
| Use deep learning / CNN | No | DSG mandates no black-box modeling; MediaPipe used only for pose extraction |
| Include agentic AI | No | Not required by JD; overkill for this stage |
| Number of ML models | 5 | Covers regression, classification, ensemble, boosting, SVM approaches |
| Project name | GolfBioMetrics | Clear, professional, signals both sport and scientific approach |
| Primary sport focus | Golf (driver swing) | Aligned with DSG JD and role description |
| Pose estimation tool | MediaPipe | Free, fast, production-ready, 33 keypoints |
| Target stakeholders | Coaches, Golfers, Manufacturers, Sports Medicine | Covers all major revenue streams for DSG |
| Presentation length | 6 pages | Concise executive summary suitable for Ascendium/DSG pitch |

### Context From Planning Sessions

- DSG is being served via Ascendium (staffing/consulting intermediary)
- Key insight from comparable companies: Catapult (rugby), Rapsodo (baseball), Gears (football) all follow the same 3-layer architecture: raw sensor → interpretable metrics → actionable insights
- Golf biomechanics research confirms: kinematic sequence (proximal-to-distal) is the most important predictor of both ball speed and consistency
- The Freiburg cheese project (earlier Ascendium assignment) established Naz's ability to design and solve problems with synthetic data — same approach applied here
- Interview is a deep technical panel — both POC code and presentation need to be ready

### Open Questions (Ask Ascendium Before Interview)

1. What sensors/cameras does DSG currently use to capture motion data?
2. Which customer segment is DSG's primary revenue focus?
3. What is their biggest current bottleneck: data capture, metric design, or user interface?
4. Are they building for real-time feedback or post-session analysis?
5. What does a successful hire look like in the first 90 days?

---

*End of MASTER_PLAN.md*  
*Next files to create: MEMORY.md, SKILLS.md, CODE_STRUCTURE.md*
