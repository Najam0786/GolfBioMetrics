# Environmental & Demographic Features — Comprehensive Analysis

**Why Golf is Different from Other Sports: The Environment Matters**

Unlike basketball or tennis (controlled environments), golf is played outdoors across wildly different conditions. A 150-yard shot in Denver (5,000 ft elevation) plays 170 yards. A drive into a 20mph wind goes 20 yards shorter.

**Without environmental context, swing analysis is incomplete.**

---

## 1. Proposed Feature Categories

### 1.1 Demographics & Physical (Easy to Collect)

| Feature | Impact | Data Source |
|---------|--------|-------------|
| **Gender** | Pelvis width, swing mechanics differ | User profile |
| **Height** | Lever arm length → clubhead speed | User profile (+3cm = +1 mph) |
| **Fitness Level (1-10)** | Core strength = stability | Questionnaire |
| **Dominant Hand** | Left/right lead mechanics | User profile |
| **Body Mass Index** | Weight transfer efficiency | Height + weight input |

### 1.2 Environmental (Automatic via APIs)

| Feature | Impact | API Source |
|---------|--------|------------|
| **Elevation (meters)** | Air density → ball flight distance | GPS/Geocoding API |
| **Temperature (°C)** | Muscle elasticity, ball compression | Weather API |
| **Wind Speed (mph)** | Ball flight trajectory | Weather API |
| **Wind Direction** | Push/pull effects | Weather API |
| **Humidity (%)** | Air density subtle effects | Weather API |
| **Time of Day** | Circadian rhythm, fatigue | Timestamp |
| **Season** | Course conditions, clothing restrictions | Calendar |

### 1.3 Course Characteristics (GPS + Course Database)

| Feature | Impact | Detection Method |
|---------|--------|------------------|
| **Course Type** | Links (windy) vs Parkland (sheltered) | Course database lookup |
| **Terrain** | Mountain course = thin air | Elevation + slope analysis |
| **Coastal Proximity** | Links courses, unpredictable wind | GPS distance to coast |
| **Grass Type** | Bermuda vs Bent affects turf interaction | Course database |
| **Green Speed (Stimp)** | Putting stroke mechanics | USGA database or user input |
| **Rough Length** | Recovery swing requirements | Course conditions |

### 1.4 Round Context (Derived from Swing Sequence)

| Feature | Impact | Calculation |
|---------|--------|-------------|
| **Hole Number (1-18)** | Cumulative fatigue | Swing timestamp |
| **Shots Taken This Hole** | Pressure escalation | Score tracking |
| **Score Relative to Par** | Mental state (safe vs aggressive) | Score differential |
| **Previous Shot Quality** | Emotional carryover | Previous swing metrics |
| **Lie Quality** | Swing compensation needs | Visual analysis (future) |
| **Stance Slope** | Balance requirements | IMU data (future) |

---

## 2. Feature Impact Analysis

### 2.1 Environmental Effects on Outcomes

**Elevation (Denver vs Miami):**
```
Denver elevation: 5,280 ft
Air density: 0.98 kg/m³ (vs 1.20 at sea level)
Ball flight: +15% distance
300 yard drive in Miami = 345 yards in Denver

ML Feature: elevation_meters
Effect: +0.5 yards per 100m elevation
```

**Temperature (Winter vs Summer):**
```
Cold (5°C): Ball compression ↓, muscle stiffness ↑
  → -5 to -10 yards vs optimal
Optimal (20-25°C): Standard conditions
Hot (35°C): Ball flies farther, fatigue ↑
  → +3-5 yards but late-round degradation

ML Feature: temperature_celsius
Effect: Non-linear (optimal range exists)
```

**Wind (Headwind vs Tailwind):**
```
10 mph headwind: -8 to -12 yards
10 mph tailwind: +6 to +10 yards
20 mph crosswind: Lateral deviation +20 yards

ML Features: wind_speed_mph, wind_direction_degrees
Effect: Direction-dependent (vector math)
```

### 2.2 Combined Environmental Scoring

```python
# Environmental Difficulty Index (0 = easy, 1 = extreme)
env_difficulty = (
    0.25 * wind_speed_mph / 30 +           # Max 30mph
    0.20 * abs(elevation - 500) / 4000 +    # Normalized around 500m
    0.20 * abs(temperature - 22) / 25 +     # Optimal at 22°C
    0.15 * hole_number / 18 +               # Fatigue factor
    0.20 * (1 if coastal_links else 0)     # Links course difficulty
)

# Use: "Your drive of 285 yards into 20mph wind is equivalent to 
#        310 yards in calm conditions"
```

---

## 3. Architecture: Where Each Feature Fits

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: INPUT ENRICHMENT                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1.1 User Profile (Demographics)                                            │
│     • Gender, Height, Dominant Hand, Fitness Level                        │
│                                                                             │
│ 1.2 Environmental APIs (Automatic)                                         │
│     • GPS Position → Elevation, Coastal Proximity                          │
│     • Weather API → Temperature, Wind Speed/Direction, Humidity             │
│     • Timestamp → Time of Day, Season, Hole Number                          │
│     • Course Database → Course Type, Green Speed, Grass Type              │
│                                                                             │
│ 1.3 Swing Data (Existing)                                                  │
│     • Video/IMU keypoints                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: BIOMECHANICS (UNCHANGED)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ • All 7 metrics remain PURE — environment doesn't change how we measure      │
│ • But environment EXPLAINS variance in outcomes                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: ML PREDICTION (ENRICHED)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ FEATURE CATEGORIES:                                                          │
│                                                                             │
│ Demographic (5):                                                            │
│   age, gender, height_m, fitness_level, dominant_hand                       │
│                                                                             │
│ Environmental (7):                                                          │
│   elevation_m, temperature_c, wind_speed_mph, wind_direction,               │
│   humidity_pct, time_of_day, coastal_flag                                   │
│                                                                             │
│ Course (4):                                                                 │
│   course_type, green_speed_stimp, grass_type, hole_number                   │
│                                                                             │
│ Biomechanics (20):                                                          │
│   [existing 7 core metrics + 13 derived features]                         │
│                                                                             │
│ TOTAL: ~36 features → 5 models                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Business Applications

### 4.1 Realistic Expectations

**Scenario**: 45-year-old male, 5'10", average fitness, playing in Denver, 20mph wind

**Without Environment Context:**
> "Your drive of 285 yards is below the 300-yard elite benchmark."

**With Environment Context:**
> "Your drive of 285 yards in Denver (+15% elevation) into 20mph wind (-10% headwind) is equivalent to **340 yards at sea level in calm conditions**. This is **excellent** for your biomechanical profile."

### 4.2 Coaching Adjustments

**Wind Compensation Coaching:**
```
Detected: 15mph crosswind from left
Biomechanics: Your swing path is 2° outside-in (slice tendency)
Environment: Wind will exaggerate this to 15 yards right of target

Coaching Cue: Aim 10 yards left of target to use wind for draw correction.
```

### 4.3 Equipment Recommendations

**Mountain Golf (5,000+ ft elevation):**
> "Your current driver has 10.5° loft. At this elevation, consider 12° loft to optimize launch angle in thin air. Expect +15 yards carry."

**Windy Links Golf:**
> "Your swing generates high spin (3,200 rpm). In today's 25mph wind, consider low-spin ball model."

### 4.4 Injury Risk (Environmental)

```
Golfer: 62 years old
Environment: Cold morning (8°C), damp course

Additional Risk Factors:
  • temperature: 8°C → Muscle stiffness +15%
  • humidity: 85% → Grip pressure changes
  • wind: 12mph → Compensatory tension

Recommendation: Extended warm-up routine required.
Injury risk elevated 0.12 points vs optimal conditions.
```

---

## 5. Implementation Priority

### Phase A: Demographics (This Week — 2 hours)
- [ ] Gender (M/F/Other/Prefer not to say)
- [ ] Height (cm)
- [ ] Dominant hand (Left/Right)
- [ ] Fitness level (1-10 self-rated)

**Impact**: +5-8% R² improvement, immediate personalization

### Phase B: Weather Integration (Next Week — 4 hours)
- [ ] OpenWeatherMap API integration
- [ ] GPS → weather station matching
- [ ] Temperature, wind speed/direction capture

**Impact**: +10-15% R² improvement, explains 20+ yard variance

### Phase C: Elevation & Course (Week 3 — 6 hours)
- [ ] Google Maps Elevation API
- [ ] Course database integration (10,000+ courses)
- [ ] Links vs Parkland classification

**Impact**: +5-10% R² improvement, mountain golf accuracy

### Phase D: Round Context (Week 4 — 8 hours)
- [ ] Score tracking integration
- [ ] Hole-by-hole fatigue modeling
- [ ] Emotional state inference from patterns

**Impact**: +5% R² improvement, mental game insights

---

## 6. Technical Implementation Details

### 6.1 API Integration Example

```python
import requests

def get_environmental_conditions(lat: float, lon: float, timestamp: datetime):
    """
    Fetches environmental data for swing location.
    
    Args:
        lat, lon: GPS coordinates from user's phone
        timestamp: When swing occurred
    
    Returns:
        dict with temperature, wind, elevation
    """
    # Weather API (OpenWeatherMap)
    weather = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    ).json()
    
    # Elevation API (Google Maps)
    elevation = requests.get(
        f"https://maps.googleapis.com/maps/api/elevation/json"
        f"?locations={lat},{lon}&key={API_KEY}"
    ).json()
    
    return {
        'temperature_c': weather['main']['temp'],
        'wind_speed_mph': weather['wind']['speed'] * 2.237,  # m/s to mph
        'wind_direction': weather['wind']['deg'],
        'humidity_pct': weather['main']['humidity'],
        'elevation_m': elevation['results'][0]['elevation'],
        'pressure_hpa': weather['main']['pressure'],
    }
```

### 6.2 Feature Engineering

```python
def compute_environmental_features(env: dict) -> dict:
    """
    Converts raw environmental data to ML features.
    """
    features = {}
    
    # Elevation effect on distance (air density model)
    sea_level_density = 1.225  # kg/m³ at sea level
    altitude_density = sea_level_density * (1 - 2.25577e-5 * env['elevation_m'])**5.25588
    features['air_density_factor'] = altitude_density / sea_level_density
    
    # Temperature effect (quadratic: optimal around 22°C)
    temp = env['temperature_c']
    features['temperature_efficiency'] = 1.0 - 0.0015 * (temp - 22)**2
    
    # Wind effect (vector decomposition)
    wind_dir_rad = np.radians(env['wind_direction'])
    # Assume golfer faces north (0°), adjust for actual orientation
    features['wind_headwind_component'] = env['wind_speed_mph'] * np.cos(wind_dir_rad)
    features['wind_crosswind_component'] = env['wind_speed_mph'] * np.sin(wind_dir_rad)
    
    # Environmental difficulty index
    features['env_difficulty_index'] = (
        0.3 * min(1.0, env['wind_speed_mph'] / 30) +
        0.2 * abs(env['temperature_c'] - 22) / 30 +
        0.2 * (1 - features['air_density_factor']) +
        0.3 * (1 if env.get('links_course') else 0)
    )
    
    return features
```

---

## 7. Expected Business Outcomes

### Improved Prediction Accuracy

| Model | Target | Current R² | With Environment | Improvement |
|-------|--------|------------|------------------|-------------|
| Linear Regression | Ball Speed | 0.70 | ~0.80 | +14% |
| Random Forest | Carry Distance | 0.72 | ~0.85 | +18% |
| XGBoost | Injury Risk | 0.93 | ~0.96 | +3% |

### New Product Capabilities

1. **"Normalized Distance"**: "That 280-yard drive into 20mph wind = 315 in calm"
2. **Environment-Based Coaching**: "It's 8°C — your X-Factor is typically 3° lower in cold"
3. **Equipment Recommendations**: "High-spin ball not ideal for today's 25mph wind"
4. **Injury Prevention**: "Cold + damp + early morning = elevated warm-up need"

---

## 8. Summary: Yes, Add All of These

**Your instinct is correct** — golf is uniquely environment-dependent. A swing analysis system without environmental context is like a weather app without location.

### Recommended Implementation Order:
1. ✅ **Age/Experience** — DONE (21% importance)
2. 🔄 **Gender/Height/Fitness** — THIS WEEK (easy, +5% accuracy)
3. 🔄 **Weather/Wind** — NEXT WEEK (moderate effort, +10% accuracy)
4. 🔄 **Elevation/Course** — WEEK 3 (API work, +5% accuracy)
5. ⏳ **Round Context** — WEEK 4 (complex, +3% accuracy)

**The full environmental layer transforms GolfBioMetrics from a "swing analyzer" to a "complete golf performance system."**

---

**Next Step**: Shall I implement **gender, height, and fitness level** (Phase A) in the synthetic data generator? This is 30 minutes of work and immediately improves personalization.
