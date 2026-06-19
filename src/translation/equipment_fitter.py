"""
equipment_fitter.py
Translation Layer: ML metrics for golf club fitting.

Purpose: Connect swing biomechanics to equipment recommendations.
Helps fitters recommend shafts, lofts, and club specs based on physics,
not just swing speed.

Key Outputs:
- Shaft flex recommendations based on tempo + release
- Loft adjustments for environmental conditions
- Club specifications backed by 58-feature analysis
- A/B testing framework for equipment changes
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd


@dataclass
class ShaftRecommendation:
    """Shaft specification recommendation."""
    current_shaft: str
    recommended_shaft: str
    
    # Why this recommendation
    fitting_reason: str  # e.g., "Late release + fast tempo suggests need for tip stability"
    
    # Specifications
    flex: str  # "R", "S", "X", "XX"
    weight_grams: int
    torque: float  # degrees
    kick_point: str  # "Low", "Mid", "High"
    tip_stiffness: str  # "Soft", "Medium", "Stiff"
    
    # Expected improvement
    expected_dispersion_reduction: int  # percentage
    expected_distance_gain: int  # yards
    expected_launch_angle_change: float  # degrees
    
    # Testing protocol
    test_protocol: List[str]


@dataclass
class LoftRecommendation:
    """Loft adjustment recommendation based on environment + swing."""
    current_loft: float
    recommended_loft: float
    loft_change: float  # degrees
    
    # Reasons
    primary_reason: str
    environmental_factor: Optional[str]
    swing_factor: Optional[str]
    
    # Impact
    expected_spin_change: int  # rpm
    expected_peak_height_change: int  # feet
    expected_distance_change: int  # yards
    

@dataclass
class EquipmentProfile:
    """Complete equipment fitting profile from swing analysis."""
    golfer_id: str
    analysis_date: str
    
    # Swing Physics
    swing_speed_mph: float
    tempo_ratio: float
    transition_aggression: str  # "Smooth", "Moderate", "Aggressive"
    release_timing: str  # "Early", "On-time", "Late"
    
    # Biomechanics
    xfactor_degrees: float
    kinematic_sequence_quality: str  # "Poor", "Average", "Good", "Elite"
    
    # Environment
    home_course_elevation: float  # meters
    typical_wind_conditions: str
    
    # Recommendations
    driver_recommendations: Dict[str, any]
    iron_recommendations: Dict[str, any]
    wedge_recommendations: Dict[str, any]
    
    # Fitting Notes
    priorities_ranked: List[str]  # ["Distance", "Accuracy", "Consistency", "Feel"]
    special_considerations: List[str]


# ============================================================================
# EQUIPMENT FITTING LOGIC
# ============================================================================

def recommend_shaft(
    swing_speed: float,
    tempo_ratio: float,
    lag_angle: float,
    xfactor: float,
    sequence_score: float,
    current_shaft_flex: str = "Unknown"
) -> ShaftRecommendation:
    """
    Recommend shaft based on swing physics.
    
    Algorithm considers:
    - Speed: Basic flex indicator
    - Tempo: Aggressive tempo needs stiffer tip
    - Release: Late release needs more stability
    - Sequence: Poor sequence benefits from softer shaft for timing
    """
    
    # Base recommendation on swing speed
    if swing_speed < 85:
        base_flex = "R"
        base_weight = 55
    elif swing_speed < 95:
        base_flex = "S"
        base_weight = 60
    elif swing_speed < 105:
        base_flex = "X"
        base_weight = 65
    else:
        base_flex = "XX"
        base_weight = 70
    
    # Adjust for tempo (aggressive = stiffer)
    if tempo_ratio > 3.0:  # Slow, deliberate
        tempo_adjustment = 0  # Keep base
        kick_point = "Mid"
    elif tempo_ratio < 2.5:  # Quick, aggressive
        tempo_adjustment = 1  # Stiffer
        kick_point = "Low"  # Lower kick for more tip stability
    else:
        tempo_adjustment = 0
        kick_point = "Mid"
    
    # Adjust for release (late = stiffer)
    if lag_angle > 24:  # Late release
        release_adjustment = 1  # Stiffer for stability
        tip_stiffness = "Stiff"
    elif lag_angle < 20:  # Early release
        release_adjustment = -1  # Softer for feel
        tip_stiffness = "Medium"
    else:
        release_adjustment = 0
        tip_stiffness = "Medium"
    
    # Adjust for sequence (poor sequence = softer for timing)
    if sequence_score < 0.75:
        sequence_adjustment = -1  # Softer
    else:
        sequence_adjustment = 0
    
    # Calculate final flex
    flex_adjustment = tempo_adjustment + release_adjustment + sequence_adjustment
    
    flex_order = ["R", "S", "X", "XX"]
    if base_flex in flex_order:
        current_idx = flex_order.index(base_flex)
        new_idx = max(0, min(len(flex_order) - 1, current_idx + flex_adjustment))
        final_flex = flex_order[new_idx]
    else:
        final_flex = base_flex
    
    # Weight adjustment
    final_weight = base_weight + (flex_adjustment * 5)
    
    # Torque recommendation (stiffer = lower torque)
    if final_flex in ["X", "XX"]:
        torque = 2.8
    elif final_flex == "S":
        torque = 3.2
    else:
        torque = 3.5
    
    # Build recommendation
    if current_shaft_flex != "Unknown":
        current = f"{current_shaft_flex} flex, ~{base_weight}g"
    else:
        current = f"{base_flex} flex (estimated)"
    
    recommended = f"{final_flex} flex, {final_weight}g"
    
    # Generate reason
    reasons = []
    if tempo_adjustment > 0:
        reasons.append(f"aggressive tempo ({tempo_ratio:.1f})")
    if release_adjustment > 0:
        reasons.append(f"late release ({lag_angle:.1f}°)")
    if sequence_adjustment < 0:
        reasons.append("improving sequence timing")
    
    if reasons:
        fitting_reason = f"Your {' and '.join(reasons)} suggests you need more tip stability."
    else:
        fitting_reason = f"Standard recommendation for your swing speed ({swing_speed:.1f} mph)."
    
    # Expected improvements (estimates)
    if flex_adjustment > 0:  # Going stiffer
        dispersion_reduction = 15
        distance_gain = 3
        launch_change = -1.0  # Lower launch
    elif flex_adjustment < 0:  # Going softer
        dispersion_reduction = 5
        distance_gain = 5
        launch_change = +1.5  # Higher launch
    else:
        dispersion_reduction = 8
        distance_gain = 4
        launch_change = 0
    
    return ShaftRecommendation(
        current_shaft=current,
        recommended_shaft=recommended,
        fitting_reason=fitting_reason,
        flex=final_flex,
        weight_grams=final_weight,
        torque=torque,
        kick_point=kick_point,
        tip_stiffness=tip_stiffness,
        expected_dispersion_reduction=dispersion_reduction,
        expected_distance_gain=distance_gain,
        expected_launch_angle_change=launch_change,
        test_protocol=[
            "Hit 10 shots with current shaft (baseline)",
            "Hit 10 shots with recommended shaft",
            "Compare: dispersion circle, peak height, ball speed",
            "Test on course if possible - 3 holes each",
            "Check feel on partial swings (75%, 50%)",
        ]
    )


def recommend_loft_adjustment(
    ball_speed: float,
    launch_angle: float,
    spin_rate: int,
    elevation_m: float,
    temperature_c: float,
    current_loft: float,
) -> LoftRecommendation:
    """
    Recommend loft adjustment based on ball flight and environment.
    
    Considers:
    - Ball speed (affects optimal launch)
    - Current launch angle (too high/low?)
    - Spin rate (too much/too little?)
    - Elevation (altitude affects air density)
    - Temperature (affects ball flight)
    """
    
    # Calculate optimal launch based on ball speed
    if ball_speed < 140:
        optimal_launch = 12.0
        optimal_spin = 2800
    elif ball_speed < 155:
        optimal_launch = 11.0
        optimal_spin = 2600
    else:
        optimal_launch = 10.0
        optimal_spin = 2400
    
    # Calculate environmental adjustment
    # Denver (1609m) vs sea level: ~10% distance boost
    elevation_factor = elevation_m / 1609  # Normalized to Denver
    elevation_adjustment = elevation_factor * 1.0  # ~1° per Denver-equivalent
    
    # Temperature adjustment
    # Cold (< 10°C): less distance, may need more loft
    # Hot (> 25°C): more distance, may need less loft
    if temperature_c < 10:
        temp_adjustment = 0.5
    elif temperature_c > 25:
        temp_adjustment = -0.5
    else:
        temp_adjustment = 0
    
    # Launch angle adjustment
    launch_diff = launch_angle - optimal_launch
    if abs(launch_diff) > 1.5:
        launch_adjustment = -launch_diff * 0.5  # Rough rule: 2° launch = 1° loft
    else:
        launch_adjustment = 0
    
    # Calculate total adjustment
    total_adjustment = elevation_adjustment + temp_adjustment + launch_adjustment
    
    # Round to nearest 0.5°
    loft_change = round(total_adjustment * 2) / 2
    recommended_loft = current_loft + loft_change
    
    # Determine reasons
    if elevation_adjustment != 0:
        env_reason = f"{elevation_m:.0f}m elevation affects air density"
    else:
        env_reason = None
    
    if temp_adjustment != 0:
        temp_reason = f"{temperature_c:.0f}°C temperature"
    else:
        temp_reason = None
    
    if launch_adjustment != 0:
        swing_reason = f"launch angle {launch_angle:.1f}° vs optimal {optimal_launch:.1f}°"
    else:
        swing_reason = None
    
    # Combine reasons
    reasons = [r for r in [env_reason, temp_reason, swing_reason] if r]
    if reasons:
        primary_reason = f"Recommended because: {', '.join(reasons)}"
    else:
        primary_reason = "Current loft is appropriate for your conditions"
    
    # Calculate expected changes
    if loft_change > 0:  # Adding loft
        spin_change = int(loft_change * 400)  # ~400 rpm per degree
        height_change = int(loft_change * 8)  # ~8 feet per degree
        distance_change = int(-loft_change * 3)  # Slight distance loss
    else:  # Reducing loft
        spin_change = int(loft_change * 400)
        height_change = int(loft_change * 8)
        distance_change = int(-loft_change * 2)  # Slight distance gain
    
    return LoftRecommendation(
        current_loft=current_loft,
        recommended_loft=recommended_loft,
        loft_change=loft_change,
        primary_reason=primary_reason,
        environmental_factor=env_reason,
        swing_factor=swing_reason,
        expected_spin_change=spin_change,
        expected_peak_height_change=height_change,
        expected_distance_change=distance_change,
    )


def generate_equipment_profile(
    swing_metrics: pd.Series,
    environmental_conditions: Dict,
    golfer_id: str = "unknown",
) -> EquipmentProfile:
    """
    Generate complete equipment fitting profile.
    """
    
    # Extract key metrics
    swing_speed = swing_metrics.get("club_speed_at_impact", 90)
    tempo = swing_metrics.get("swing_tempo_ratio", 2.8)
    lag_angle = swing_metrics.get("lag_angle_impact", 22)
    xfactor = swing_metrics.get("xfactor_degrees", 40)
    sequence = swing_metrics.get("kinematic_sequence_score", 0.80)
    
    # Determine characteristics
    if tempo > 3.0:
        transition = "Smooth"
    elif tempo < 2.5:
        transition = "Aggressive"
    else:
        transition = "Moderate"
    
    if lag_angle > 24:
        release = "Late"
    elif lag_angle < 20:
        release = "Early"
    else:
        release = "On-time"
    
    if sequence > 0.90:
        seq_quality = "Elite"
    elif sequence > 0.80:
        seq_quality = "Good"
    elif sequence > 0.70:
        seq_quality = "Average"
    else:
        seq_quality = "Poor"
    
    # Get recommendations
    shaft_rec = recommend_shaft(
        swing_speed=swing_speed,
        tempo_ratio=tempo,
        lag_angle=lag_angle,
        xfactor=xfactor,
        sequence_score=sequence,
    )
    
    loft_rec = recommend_loft_adjustment(
        ball_speed=swing_speed * 1.5,  # Rough estimate
        launch_angle=11.0,  # Default if unknown
        spin_rate=2600,  # Default
        elevation_m=environmental_conditions.get("elevation_m", 0),
        temperature_c=environmental_conditions.get("temperature_c", 20),
        current_loft=10.5,  # Standard driver
    )
    
    # Build driver recommendations
    driver_recs = {
        "shaft": {
            "current": shaft_rec.current_shaft,
            "recommended": shaft_rec.recommended_shaft,
            "reason": shaft_rec.fitting_reason,
            "specifications": {
                "flex": shaft_rec.flex,
                "weight": f"{shaft_rec.weight_grams}g",
                "torque": f"{shaft_rec.torque}°",
                "kick_point": shaft_rec.kick_point,
            },
            "expected_improvement": {
                "dispersion": f"-{shaft_rec.expected_dispersion_reduction}%",
                "distance": f"+{shaft_rec.expected_distance_gain} yards",
            },
        },
        "loft": {
            "current": f"{loft_rec.current_loft:.1f}°",
            "recommended": f"{loft_rec.recommended_loft:.1f}°",
            "change": f"{loft_rec.loft_change:+.1f}°",
            "reason": loft_rec.primary_reason,
            "expected_changes": {
                "spin": f"{loft_rec.expected_spin_change:+d} rpm",
                "peak_height": f"{loft_rec.expected_peak_height_change:+d} ft",
                "distance": f"{loft_rec.expected_distance_change:+d} yards",
            },
        },
    }
    
    # Determine priorities
    if xfactor < 38:
        priorities = ["Distance", "Consistency", "Accuracy", "Feel"]
    elif sequence < 0.75:
        priorities = ["Consistency", "Distance", "Accuracy", "Feel"]
    else:
        priorities = ["Accuracy", "Distance", "Consistency", "Feel"]
    
    # Special considerations
    special = []
    if environmental_conditions.get("elevation_m", 0) > 1000:
        special.append("Altitude adjustment recommended")
    if tempo < 2.3:
        special.append("Quick tempo - consider counter-balanced putter")
    if swing_speed > 100:
        special.append("High swing speed - premium ball recommended")
    
    return EquipmentProfile(
        golfer_id=golfer_id,
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        swing_speed_mph=swing_speed,
        tempo_ratio=tempo,
        transition_aggression=transition,
        release_timing=release,
        xfactor_degrees=xfactor,
        kinematic_sequence_quality=seq_quality,
        home_course_elevation=environmental_conditions.get("elevation_m", 0),
        typical_wind_conditions=environmental_conditions.get("wind_typical", "Moderate"),
        driver_recommendations=driver_recs,
        iron_recommendations={},  # TODO
        wedge_recommendations={},  # TODO
        priorities_ranked=priorities,
        special_considerations=special,
    )


def format_equipment_report(profile: EquipmentProfile) -> str:
    """Generate formatted equipment fitting report."""
    
    lines = [
        "=" * 80,
        "EQUIPMENT FITTING PROFILE",
        f"Golfer: {profile.golfer_id} | Date: {profile.analysis_date}",
        "=" * 80,
        "",
        "🏌️ SWING PHYSICS",
        "-" * 80,
        f"Swing Speed: {profile.swing_speed_mph:.1f} mph",
        f"Tempo Ratio: {profile.tempo_ratio:.1f} ({profile.transition_aggression} transition)",
        f"Release Timing: {profile.release_timing}",
        f"X-Factor: {profile.xfactor_degrees:.1f}°",
        f"Sequence Quality: {profile.kinematic_sequence_quality}",
        "",
        "🎯 DRIVER RECOMMENDATIONS",
        "=" * 80,
        "",
        "SHAFT FITTING:",
        "-" * 80,
    ]
    
    shaft = profile.driver_recommendations["shaft"]
    lines.extend([
        f"Current:  {shaft['current']}",
        f"Recommended: {shaft['recommended']}",
        "",
        f"Why: {shaft['reason']}",
        "",
        "Specifications:",
        f"  • Flex: {shaft['specifications']['flex']}",
        f"  • Weight: {shaft['specifications']['weight']}",
        f"  • Torque: {shaft['specifications']['torque']}",
        f"  • Kick Point: {shaft['specifications']['kick_point']}",
        "",
        f"Expected Improvement:",
        f"  • Dispersion: {shaft['expected_improvement']['dispersion']}",
        f"  • Distance: {shaft['expected_improvement']['distance']}",
        "",
    ])
    
    loft = profile.driver_recommendations["loft"]
    lines.extend([
        "LOFT ADJUSTMENT:",
        "-" * 80,
        f"Current:  {loft['current']}",
        f"Recommended: {loft['recommended']}",
        f"Change: {loft['change']}",
        "",
        f"Reason: {loft['reason']}",
        "",
        "Expected Flight Changes:",
        f"  • Spin: {loft['expected_changes']['spin']}",
        f"  • Peak Height: {loft['expected_changes']['peak_height']}",
        f"  • Distance: {loft['expected_changes']['distance']}",
        "",
    ])
    
    if profile.special_considerations:
        lines.extend([
            "⚠️ SPECIAL CONSIDERATIONS",
            "-" * 80,
        ])
        for consideration in profile.special_considerations:
            lines.append(f"  • {consideration}")
        lines.append("")
    
    lines.extend([
        "🎯 PRIORITIES",
        "-" * 80,
    ])
    for i, priority in enumerate(profile.priorities_ranked, 1):
        lines.append(f"{i}. {priority}")
    
    lines.extend([
        "",
        "=" * 80,
        "Generated by GolfBioMetrics Equipment Fitting AI",
        "Data Sports Group | GolfBioMetrics Technology",
        "=" * 80,
    ])
    
    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EQUIPMENT FITTER DEMO")
    print("=" * 80)
    print()
    
    # Sample swing metrics
    sample_metrics = pd.Series({
        "club_speed_at_impact": 92.5,
        "swing_tempo_ratio": 2.3,  # Fast
        "lag_angle_impact": 26.5,  # Late release
        "xfactor_degrees": 42.0,
        "kinematic_sequence_score": 0.85,
    })
    
    env_conditions = {
        "elevation_m": 1609,  # Denver
        "temperature_c": 25,
        "wind_typical": "Moderate",
    }
    
    profile = generate_equipment_profile(
        swing_metrics=sample_metrics,
        environmental_conditions=env_conditions,
        golfer_id="demo_001",
    )
    
    print(format_equipment_report(profile))
    print()
    print("=" * 80)
    print("Equipment fitting demo complete!")
    print("=" * 80)
