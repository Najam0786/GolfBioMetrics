# GolfBioMetrics — Skills Reference Library

**Purpose:** Biomechanics, geometry, and signal processing reference for coding sessions.  
**Use:** Read this when writing metric computation code to ensure formulas are correct.  
**Last Updated:** June 18, 2026

---

## 1. Coordinate System Convention

```
World coordinate system (right-handed):
  X axis: mediolateral (left-right)
  Y axis: vertical (up-down)
  Z axis: anterior-posterior (forward-back)

Golfer faces: negative Z direction (toward ball)
Right-handed golfer:
  Lead side = left side of body
  Trail side = right side of body
  Back foot = right foot
  Front foot = left foot
```

---

## 2. Core 3D Geometry Formulas

### Unsigned Angle Between Two Vectors (0° to 180°)
```python
import numpy as np

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns unsigned angle in degrees. Always 0-180."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)  # CLIP IS MANDATORY
    return float(np.degrees(np.arccos(dot)))
```

### Signed Angle Between Two 3D Vectors (with Reference Normal)
```python
def signed_angle_3d(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> float:
    """
    Returns signed angle from v1 to v2 in degrees.
    normal: reference axis that defines positive rotation direction.
    Positive = counter-clockwise when viewed from normal direction.
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    cross = np.cross(v1_norm, v2_norm)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.degrees(np.arctan2(np.linalg.norm(cross), dot))
    sign = np.sign(np.dot(cross, normal))
    return float(sign * angle)
```

### Signed Angle Between Two 2D Vectors
```python
def signed_angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Returns signed angle from v1 to v2 in degrees.
    Positive = counter-clockwise.
    Use for transverse plane (horizontal) angles.
    """
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = np.dot(v1, v2)
    return float(np.degrees(np.arctan2(cross, dot)))
```

### Project Vector Onto Plane
```python
def project_onto_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Projects vector v onto the plane defined by plane_normal.
    Used for computing transverse/frontal plane angles.
    """
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
    return v - np.dot(v, n) * n
```

### Best-Fit Plane Through 3D Data Points (SVD)
```python
def fit_plane(points: np.ndarray) -> tuple:
    """
    Fits a plane to a set of 3D points using SVD.
    Returns: (normal_vector, centroid)
    Use for: club path consistency metric.
    """
    centroid = np.mean(points, axis=0)
    centred = points - centroid
    _, _, vh = np.linalg.svd(centred)
    normal = vh[-1]  # Last row = normal to best-fit plane
    return normal, centroid
```

### Mirror Across Sagittal Plane (for Left-Handed Golfers)
```python
def mirror_sagittal(keypoints: np.ndarray) -> np.ndarray:
    """
    Mirrors keypoints across the sagittal (YZ) plane.
    Converts right-handed to left-handed golfer representation.
    keypoints: shape (N, 3) — N keypoints, each with (x, y, z)
    """
    mirrored = keypoints.copy()
    mirrored[:, 0] = -mirrored[:, 0]  # Negate X component only
    # NOTE: After mirroring, left/right keypoint labels must also be swapped
    return mirrored
```

---

## 3. Signal Processing Reference

### Zero-Phase Filtering (Butterworth)
```python
from scipy.signal import butter, filtfilt

def smooth_signal(signal: np.ndarray, fps: int = 30,
                  cutoff_hz: float = 6.0, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth filter. No phase distortion.
    cutoff_hz: 6Hz for gross body motion, 10Hz for fast limb motion
    IMPORTANT: requires signal length > 3 × filter order
    """
    if len(signal) < 3 * order:
        return signal  # Too short to filter reliably — return raw
    nyquist = fps / 2.0
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)
```

### Peak Detection (for Kinematic Sequence)
```python
from scipy.signal import find_peaks

def find_peak_in_window(signal: np.ndarray, start_idx: int,
                        end_idx: int) -> tuple:
    """
    Finds the peak angular velocity within a time window.
    Returns: (peak_value, peak_index)
    """
    window = signal[start_idx:end_idx]
    if len(window) == 0:
        return None, None
    peak_idx = np.argmax(window)
    return window[peak_idx], start_idx + peak_idx
```

### Interpolate Missing Frames
```python
def interpolate_missing(signal: np.ndarray,
                        confidence: np.ndarray,
                        threshold: float = 0.3) -> tuple:
    """
    Linearly interpolates frames where confidence < threshold.
    Returns: (interpolated_signal, was_interpolated_mask)
    IMPORTANT: Only interpolate short gaps (< 5 consecutive frames).
    Longer gaps → set confidence = 0, do not interpolate.
    """
    interpolated = signal.copy()
    was_interpolated = np.zeros(len(signal), dtype=bool)
    low_conf = confidence < threshold

    # Find contiguous low-confidence regions
    i = 0
    while i < len(signal):
        if low_conf[i]:
            start = i
            while i < len(signal) and low_conf[i]:
                i += 1
            end = i
            gap_length = end - start
            if gap_length <= 5 and start > 0 and end < len(signal):
                # Short gap: linear interpolation
                interpolated[start:end] = np.linspace(
                    signal[start - 1], signal[end], gap_length + 2)[1:-1]
                was_interpolated[start:end] = True
            else:
                # Long gap: do not interpolate, mark as unreliable
                was_interpolated[start:end] = True
        else:
            i += 1
    return interpolated, was_interpolated
```

### Circular Statistics for Angle Data
```python
from scipy.stats import circmean, circstd
import numpy as np

def circular_mean_degrees(angles_deg: np.ndarray) -> float:
    """
    Correct mean for angular/circular data.
    Use when angles might wrap around 0°/360° boundary.
    Example: mean of [350°, 10°] = 0°, NOT 180°.
    """
    angles_rad = np.radians(angles_deg)
    return float(np.degrees(circmean(angles_rad, high=np.pi, low=-np.pi)))

# NOTE: For angles bounded within physiological ranges (0-160° for knee flexion),
# regular np.mean() is fine — no circular statistics needed.
# Only use circular stats for orientations that can wrap around 360°.
```

---

## 4. Golf Biomechanics Reference Values

### Elite Golfer Benchmarks (Driver Swing)

| Metric | Amateur | Semi-Pro | Elite (PGA Tour) |
|--------|---------|----------|-----------------|
| X-Factor (top of backswing) | 15–30° | 30–45° | 40–55° |
| Lag angle (mid-downswing) | 40–65° | 65–80° | 75–90° |
| Lag angle (at impact) | 5–15° | 15–25° | 20–35° |
| Kinematic sequence score | 0.40–0.65 | 0.65–0.85 | 0.85–0.98 |
| Swing tempo ratio | 1.2–2.0 | 2.0–2.5 | 2.3–3.0 |
| Clubhead speed | 70–85 mph | 90–100 mph | 110–130 mph |
| Ball speed | 95–115 mph | 120–140 mph | 155–190 mph |
| Carry distance | 160–210 yds | 220–260 yds | 270–320 yds |
| Smash factor | 1.35–1.42 | 1.43–1.47 | 1.48–1.52 |

### Kinematic Sequence Timing (Elite Driver)

```
Downswing phase duration: ~0.2–0.3 seconds

Segment peak angular velocity order:
  t=0.000s  Downswing initiated (hips begin rotating)
  t=0.060s  Pelvis peak angular velocity (~600°/s)
  t=0.090s  Thorax peak angular velocity (~650°/s)
  t=0.130s  Lead arm peak angular velocity (~900°/s)
  t=0.200s  Club peak velocity (at impact) (~3000°/s)

Inter-segment timing gap: 30–50ms (elite), 0–20ms or reversed (amateur)
```

### Swing Phases — Event Detection Thresholds

```
Address:         Static — club velocity < 0.5 m/s for > 0.5s
Backswing start: Club velocity > 0.5 m/s moving away from ball
Top of swing:    Club velocity direction reverses (zero crossing of club_z_velocity)
Downswing start: Same as top of swing end
Impact:          Peak club velocity moment (for synthetic data)
                 Or: frame where club-ball distance < 5cm (real data)
Follow-through:  Post-impact deceleration phase
```

### Compensation Patterns — Detection Thresholds

```
Early arm cast:     Lag angle decreases by >10° BEFORE pelvis reaches peak velocity
Reverse pivot:      COM_x on front-foot side (>0.6 normalized) at top of backswing
Lateral sway:       COM_x moves >0.15m laterally during backswing
Early extension:    Hip-to-ball distance decreases during downswing (hips thrust forward)
Over-the-top:       Club path angle crosses outside-in at mid-downswing
Excessive trunk tilt: Spine angle changes >15° between address and impact
```

---

## 5. MediaPipe Keypoint Index Reference

```
Index  Keypoint
0      nose
11     left_shoulder
12     right_shoulder
13     left_elbow
14     right_elbow
15     left_wrist
16     right_wrist
17     left_pinky
18     right_pinky
19     left_index
20     right_index
23     left_hip
24     right_hip
25     left_knee
26     right_knee
27     left_ankle
28     right_ankle

Synthetic additions (custom, not MediaPipe):
100    club_grip_point   (midpoint of hands)
101    club_head         (end of club shaft)
```

---

## 6. Confidence Scoring Formula

```python
def compute_confidence(
    keypoint_confidences: np.ndarray,  # Confidence of each keypoint used
    missing_frame_fraction: float,      # 0.0 = no missing, 1.0 = all missing
    value: float,                       # Computed metric value
    valid_range: tuple,                 # (min, max) physiologically valid range
    temporal_jitter: float              # Std dev of metric across adjacent frames
) -> float:
    """
    Composite confidence score (0.0 to 1.0) for any metric.
    """
    # Sub-scores (each 0-1)
    kp_score   = float(np.mean(keypoint_confidences))
    complete   = 1.0 - missing_frame_fraction
    range_ok   = 1.0 if valid_range[0] <= value <= valid_range[1] else 0.0
    stability  = max(0.0, 1.0 - (temporal_jitter / (valid_range[1] - valid_range[0])))

    # Weighted composite
    confidence = (
        0.35 * kp_score +
        0.30 * complete +
        0.20 * range_ok +
        0.15 * stability
    )

    # Hard rule: if range is invalid, cap confidence at 0.3
    if range_ok == 0.0:
        confidence = min(confidence, 0.3)

    return float(np.clip(confidence, 0.0, 1.0))
```

**Threshold for suppressing metric from user output:**
- confidence < 0.6 → flag as unreliable, do not display to user
- confidence 0.6–0.8 → display with warning indicator
- confidence > 0.8 → display with confidence (reliable)

---

## 7. Unit Test Patterns

```python
import pytest
import numpy as np
from src.biomechanics.geometry_utils import angle_between_vectors, signed_angle_2d

class TestAngleBetweenVectors:

    def test_perpendicular_vectors_return_90(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        assert abs(angle_between_vectors(v1, v2) - 90.0) < 1e-6

    def test_parallel_vectors_return_0(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([3, 0, 0])  # Same direction, different magnitude
        assert abs(angle_between_vectors(v1, v2) - 0.0) < 1e-6

    def test_antiparallel_vectors_return_180(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])
        assert abs(angle_between_vectors(v1, v2) - 180.0) < 1e-6

    def test_nearly_parallel_no_nan(self):
        # Floating point near-identity should not produce NaN
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0 + 1e-10, 0.0, 0.0])
        result = angle_between_vectors(v1, v2)
        assert not np.isnan(result)
        assert 0.0 <= result <= 180.0

    def test_zero_vector_handled(self):
        v1 = np.array([0.0, 0.0, 0.0])  # Zero vector
        v2 = np.array([1.0, 0.0, 0.0])
        # Should not raise, should return 0 or 90 gracefully
        result = angle_between_vectors(v1, v2)
        assert not np.isnan(result)
```

---

## 8. Common Errors & How to Avoid Them

| Error | Cause | Fix |
|-------|-------|-----|
| `arccos` returns NaN | Dot product slightly outside [-1,1] due to float precision | Always `np.clip(dot, -1.0, 1.0)` |
| Angle jumps at 0°/360° boundary | Using regular statistics on circular data | Use `scipy.stats.circmean` |
| Wrong sign for rotation direction | Using `arccos` instead of `atan2` | Use `atan2(cross, dot)` for signed angles |
| filtfilt fails on short signals | Signal length < 3 × filter order | Check length, use lower order, or skip filter |
| Metric is valid but confidence=0 | Range check too strict or wrong units | Verify units (degrees vs radians) before range check |
| Mirroring breaks cross products | Reflection changes handedness | Apply congruence transform to rotation matrices: M @ R @ M |
| Silent NaN propagation | NaN keypoint processed without check | Check for NaN at input, return low confidence early |
| Division by zero in normalization | Zero-length vector divided to get unit vector | Add `1e-8` to denominator: `v / (norm + 1e-8)` |

---

## 9. Translation Layer — Feature to Drill Mapping (NEW)

**Purpose:** Convert biomechanics metrics into actionable golf instruction.
**Location:** `src/translation/golfer_report.py`
**Last Updated:** June 18, 2026

### The Translation Philosophy

ML outputs technical metrics. Golfers need to know:
1. **What's wrong?** (in plain English)
2. **How much is it costing me?** (yards lost)
3. **What drill fixes it?** (specific instructions)
4. **How long until I see improvement?** (expected timeline)

### Feature → Diagnosis → Drill Database

| Feature | Poor Threshold | Diagnosis | Yards Lost Formula | Recommended Drill |
|---------|----------------|-----------|-------------------|-------------------|
| `lag_angle_impact` | < 18° | Early Club Release ("Casting") | `(18 - x) × 1.5` | The Pump Drill |
| `xfactor_degrees` | < 38° | Limited Hip-Shoulder Separation | `(40 - x) × 2` | The Chair Drill |
| `swing_tempo_ratio` | < 2.0 | Rushed Backswing | Fixed 10 yards | Metronome Swings |
| `kinematic_sequence_score` | < 0.75 | Poor Downswing Sequence | `(0.80 - x) × 50` | Step-Through Drill |
| `weight_transfer_timing_ms` | > -50 | Late Weight Transfer | Fixed 12 yards | Heel-Up Drill |

### Drill Template Structure

```python
{
    "name": "The Pump Drill",
    "description": "Pause at top of backswing, pump the club to feel wrist angle",
    "steps": [
        "Take normal backswing",
        "Pause at top for 2 seconds",
        "Pump the club down 6 inches while maintaining wrist hinge",
        "Swing through to finish",
    ],
    "duration_minutes": 10,
    "frequency": "Daily",
    "expected_improvement": "+4-6° lag angle in 2-3 weeks",
    "difficulty": "Beginner",
    "equipment": ["7-iron", "mirror (optional)"],
}
```

### Generating Golfer Reports

```python
from src.translation.golfer_report import generate_golfer_report

# From swing data (58 features)
report = generate_golfer_report(
    swing_data=feature_row,  # pd.Series with 58 features
    golfer_id="john_001",
    handicap=18
)

# Output includes:
# - List of GolfProblem objects (sorted by severity)
# - 7-day practice plan
# - Expected yards gain
# - Handicap reduction estimate
```

### Example Output Format

```
⚠️ PRIORITY FIX #1: Early Club Release ('Casting')
Severity: High | Yards Lost: ~22

What we found:
You're releasing the club too early in the downswing.
This causes weak shots, reduced distance, and often a slice.

Your lag_angle_impact: 16.5° | Target: 22°

💪 RECOMMENDED DRILL: The Pump Drill
Duration: 10 minutes | Frequency: Daily
Expected: +4-6° lag angle in 2-3 weeks

Steps:
1. Take normal backswing
2. Pause at top for 2 seconds
3. Pump the club down 6 inches while maintaining wrist hinge
4. Swing through to finish
```

### Testing the Translation Layer

```bash
# Run the demo
python src/translation/master_translator.py

# This generates all 3 reports:
# 1. Golfer Report (drill recommendations)
# 2. Coach Report (progress tracking)
# 3. Equipment Fitter Report (shaft recommendations)
```
