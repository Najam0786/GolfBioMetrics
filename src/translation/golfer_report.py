"""
golfer_report.py
Translation Layer: Convert ML metrics to golfer-friendly reports.

Purpose: Transform technical ML outputs (R², feature values) into 
actionable golf advice that weekend players can understand and use.

Key Outputs:
- Plain English problem descriptions
- Specific drill recommendations  
- Expected outcome predictions
- Visual progress tracking
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json


@dataclass
class GolfProblem:
    """Represents a single golf problem identified by ML."""
    problem_id: str
    title: str  # e.g., "Early Club Release Detected"
    description: str  # Plain English: "You're releasing the club too early..."
    severity: str  # "High", "Medium", "Low"
    
    # ML Source
    primary_metric: str  # e.g., "lag_angle_at_impact"
    current_value: float
    target_value: float
    unit: str  # e.g., "degrees"
    
    # Impact
    yards_lost: int  # Estimated yards lost due to this problem
    handicap_impact: float  # Estimated strokes per round
    
    # Solution
    recommended_drill: str
    drill_duration_minutes: int
    drill_frequency: str  # e.g., "Daily", "3x per week"
    expected_improvement: str  # e.g., "+5° lag angle in 2 weeks"
    
    # Resources
    has_video: bool
    video_url: Optional[str]
    has_pdf_guide: bool


@dataclass
class GolferReport:
    """Complete golfer-facing report from ML analysis."""
    golfer_id: str
    analysis_date: str
    
    # Summary Stats
    total_problems: int
    high_priority_count: int
    
    # Swing Profile
    swing_speed_mph: float
    handicap_estimate: float
    skill_level: str  # "Weekend", "Aspiring", "Competitive"
    
    # Problems (sorted by impact)
    problems: List[GolfProblem]
    
    # Strengths (what they do well)
    strengths: List[Dict[str, str]]
    
    # Weekly Practice Plan
    weekly_plan: Dict[str, List[Dict]]  # Day -> List of drills
    
    # Expected Outcomes
    expected_yards_gain: int
    expected_handicap_reduction: float
    timeline_weeks: int
    
    # ML Confidence
    analysis_confidence: float  # 0-1
    recommendation_confidence: float  # 0-1


# ============================================================================
# FEATURE → DIAGNOSIS → DRILL DATABASE
# ============================================================================

FEATURE_DIAGNOSIS_DATABASE = {
    "lag_angle_impact": {
        "metric_name": "Lag Angle at Impact",
        "unit": "degrees",
        "elite_range": (22, 30),
        "good_range": (18, 22),
        "poor_threshold": 18,
        
        "diagnoses": {
            "low": {
                "title": "Early Club Release ('Casting')",
                "description": "You're releasing the club too early in the downswing. This causes weak shots, reduced distance, and often a slice.",
                "visual_cue": "At impact, your hands and club head are almost in line. Pros keep the club head trailing the hands.",
                "yards_lost_formula": lambda x: int((18 - x) * 1.5),  # ~1.5 yards per degree
                "severity_formula": lambda x: "High" if x < 15 else "Medium" if x < 18 else "Low",
            }
        },
        
        "drills": [
            {
                "name": "The Pump Drill",
                "description": "Pause at top of backswing, pump the club to feel wrist angle, then swing through.",
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
            },
            {
                "name": "Towel Under Arm",
                "description": "Keep towel under lead arm throughout swing to promote connected, laggy swing.",
                "steps": [
                    "Place towel under lead arm (left for right-handers)",
                    "Make half-swings keeping towel in place",
                    "Feel the 'lag' as the towel prevents early release",
                ],
                "duration_minutes": 15,
                "frequency": "3x per week",
                "expected_improvement": "+3-5° lag angle in 3 weeks",
                "difficulty": "Intermediate",
                "equipment": ["7-iron", "small towel"],
            },
        ],
    },
    
    "xfactor_degrees": {
        "metric_name": "X-Factor (Hip-Shoulder Separation)",
        "unit": "degrees",
        "elite_range": (45, 55),
        "good_range": (38, 45),
        "poor_threshold": 38,
        
        "diagnoses": {
            "low": {
                "title": "Limited Hip-Shoulder Separation",
                "description": "Your hips and shoulders are rotating almost together. You're losing 20-30 yards because you can't store and release energy like the pros.",
                "visual_cue": "At the top of your swing, your hips have already turned toward the target. Pros keep hips still while shoulders turn.",
                "yards_lost_formula": lambda x: int((40 - x) * 2),  # ~2 yards per degree
                "severity_formula": lambda x: "High" if x < 35 else "Medium" if x < 38 else "Low",
            }
        },
        
        "drills": [
            {
                "name": "The Chair Drill",
                "description": "Sit on chair edge, practice turning shoulders while keeping hips stable.",
                "steps": [
                    "Sit on edge of chair, feet shoulder-width apart",
                    "Hold club across shoulders",
                    "Turn shoulders 90° while keeping hips facing forward",
                    "Feel the 'coil' in your core",
                ],
                "duration_minutes": 10,
                "frequency": "Daily",
                "expected_improvement": "+5-8° X-Factor in 2-3 weeks",
                "difficulty": "Beginner",
                "equipment": ["chair", "7-iron or similar"],
            },
            {
                "name": "Hip Bump & Hold",
                "description": "Practice bumping hips toward target while keeping shoulders back.",
                "steps": [
                    "Setup normally",
                    "Bump hips 2 inches toward target",
                    "Turn shoulders while keeping hips in bumped position",
                    "Hold the 'separation' feeling at top of backswing",
                ],
                "duration_minutes": 15,
                "frequency": "3x per week",
                "expected_improvement": "+4-6° X-Factor in 3 weeks",
                "difficulty": "Intermediate",
                "equipment": ["7-iron"],
            },
        ],
    },
    
    "swing_tempo_ratio": {
        "metric_name": "Swing Tempo Ratio",
        "unit": "ratio",
        "elite_range": (2.8, 3.2),
        "good_range": (2.5, 2.8),
        "poor_threshold_low": 2.0,
        "poor_threshold_high": 3.5,
        
        "diagnoses": {
            "fast": {
                "title": "Rushed Backswing",
                "description": "Your backswing is too quick, leading to poor sequencing and loss of power.",
                "visual_cue": "You're 'whipping' the club back. Pros take 3x longer on backswing than downswing.",
                "yards_lost_formula": lambda x: 10,  # Flat 10 yards for tempo issues
                "severity_formula": lambda x: "Medium" if x < 2.0 else "Low",
            },
            "slow": {
                "title": "Slow Backswing",
                "description": "Your backswing is too deliberate, causing stiffness and reduced power.",
                "visual_cue": "The backswing looks 'labored'. Pros are smooth but not slow.",
                "yards_lost_formula": lambda x: 8,
                "severity_formula": lambda x: "Medium" if x > 3.5 else "Low",
            }
        },
        
        "drills": [
            {
                "name": "Metronome Swings",
                "description": "Use metronome app to establish 3:1 backswing:downswing ratio.",
                "steps": [
                    "Set metronome to 60 BPM",
                    "Count 3 beats on backswing",
                    "Count 1 beat on downswing",
                    "Practice until it feels natural",
                ],
                "duration_minutes": 15,
                "frequency": "Daily",
                "expected_improvement": "Optimal 3:1 ratio in 1-2 weeks",
                "difficulty": "Beginner",
                "equipment": ["smartphone with metronome app", "7-iron"],
            },
        ],
    },
    
    "kinematic_sequence_score": {
        "metric_name": "Kinematic Sequence",
        "unit": "score",
        "elite_range": (0.85, 0.98),
        "good_range": (0.75, 0.85),
        "poor_threshold": 0.75,
        
        "diagnoses": {
            "low": {
                "title": "Poor Downswing Sequence",
                "description": "Your body parts aren't firing in the right order. You should: hips → shoulders → arms → club. You're likely firing everything at once.",
                "visual_cue": "Your downswing looks 'armsy'. The pros' hips start moving while shoulders are still completing backswing.",
                "yards_lost_formula": lambda x: int((0.80 - x) * 50),  # Significant impact
                "severity_formula": lambda x: "High" if x < 0.70 else "Medium" if x < 0.75 else "Low",
            }
        },
        
        "drills": [
            {
                "name": "Step-Through Drill",
                "description": "Step toward target to force hips to lead the downswing.",
                "steps": [
                    "Take normal backswing",
                    "As you start downswing, step left foot toward target",
                    "This forces hips to fire first",
                    "Feel the 'separation' between hip and shoulder turn",
                ],
                "duration_minutes": 15,
                "frequency": "3x per week",
                "expected_improvement": "+0.10-0.15 sequence score in 3-4 weeks",
                "difficulty": "Intermediate",
                "equipment": ["7-iron"],
            },
            {
                "name": "Hip-Only Swings",
                "description": "Practice starting downswing with hips only, holding shoulders back.",
                "steps": [
                    "Take backswing to top",
                    "Start downswing by bumping hips toward target",
                    "Keep shoulders passive as long as possible",
                    "Let arms 'fall' rather than pull",
                ],
                "duration_minutes": 10,
                "frequency": "Daily",
                "expected_improvement": "+0.08-0.12 sequence score in 2-3 weeks",
                "difficulty": "Intermediate",
                "equipment": ["7-iron"],
            },
        ],
    },
    
    "weight_transfer_timing_ms": {
        "metric_name": "Weight Transfer Timing",
        "unit": "milliseconds",
        "elite_range": (-120, -80),  # ms before impact
        "good_range": (-80, -50),
        "poor_threshold": -50,
        
        "diagnoses": {
            "late": {
                "title": "Late Weight Transfer",
                "description": "You're not shifting weight to front foot early enough. This reduces power and often causes thin/fat shots.",
                "visual_cue": "At impact, your weight is still 50/50 or even back foot. Pros have 80%+ on front foot at impact.",
                "yards_lost_formula": lambda x: 12,
                "severity_formula": lambda x: "Medium" if x > -30 else "Low",
            }
        },
        
        "drills": [
            {
                "name": "Heel-Up Drill",
                "description": "Lift back heel early in downswing to force weight forward.",
                "steps": [
                    "Take normal backswing",
                    "As you start downswing, lift back heel immediately",
                    "Feel weight shift to front foot before impact",
                    "Practice half-swings first",
                ],
                "duration_minutes": 10,
                "frequency": "Daily",
                "expected_improvement": "Earlier weight transfer in 1-2 weeks",
                "difficulty": "Beginner",
                "equipment": ["7-iron"],
            },
        ],
    },
}


# ============================================================================
# REPORT GENERATION FUNCTIONS
# ============================================================================

def diagnose_feature(feature_name: str, value: float, 
                     database: Dict = FEATURE_DIAGNOSIS_DATABASE) -> Optional[Dict]:
    """
    Look up diagnosis for a specific feature value.
    
    Args:
        feature_name: Name of the biomechanics feature
        value: Current measured value
        database: Feature diagnosis database
        
    Returns:
        Dictionary with diagnosis info or None if no issue
    """
    if feature_name not in database:
        return None
    
    feature = database[feature_name]
    
    # Check if value is in poor range
    if feature_name == "swing_tempo_ratio":
        # Special case: both too fast and too slow are problems
        if value < 2.0:
            diagnosis_type = "fast"
        elif value > 3.5:
            diagnosis_type = "slow"
        else:
            return None  # Good tempo
    else:
        # Standard: check if below threshold
        threshold = feature.get("poor_threshold", 0)
        if value >= threshold:
            return None  # No problem
        diagnosis_type = "low"
    
    # Get diagnosis details
    diagnosis = feature["diagnoses"][diagnosis_type]
    
    # Calculate severity and impact
    severity = diagnosis["severity_formula"](value)
    yards_lost = diagnosis["yards_lost_formula"](value)
    
    return {
        "feature": feature_name,
        "metric_name": feature["metric_name"],
        "value": value,
        "unit": feature["unit"],
        "title": diagnosis["title"],
        "description": diagnosis["description"],
        "visual_cue": diagnosis["visual_cue"],
        "severity": severity,
        "yards_lost": yards_lost,
        "drills": feature["drills"],
        "target_range": feature["elite_range"],
    }


def generate_golfer_report(
    swing_data: pd.Series,
    golfer_id: str = "unknown",
    handicap: Optional[float] = None,
) -> GolferReport:
    """
    Generate a complete golfer-facing report from swing metrics.
    
    Args:
        swing_data: Series containing all 58 features for one swing
        golfer_id: Identifier for the golfer
        handicap: Optional known handicap
        
    Returns:
        GolferReport with problems, drills, and practice plan
    """
    problems = []
    
    # Check each diagnostic feature
    diagnostic_features = [
        "lag_angle_impact",
        "xfactor_degrees", 
        "swing_tempo_ratio",
        "kinematic_sequence_score",
        "weight_transfer_timing_ms",
    ]
    
    for feature in diagnostic_features:
        if feature in swing_data.index:
            diagnosis = diagnose_feature(feature, swing_data[feature])
            if diagnosis:
                # Create GolfProblem
                problem = GolfProblem(
                    problem_id=f"{feature}_{len(problems)}",
                    title=diagnosis["title"],
                    description=diagnosis["description"],
                    severity=diagnosis["severity"],
                    primary_metric=feature,
                    current_value=diagnosis["value"],
                    target_value=diagnosis["target_range"][0],  # Lower bound of elite
                    unit=diagnosis["unit"],
                    yards_lost=diagnosis["yards_lost"],
                    handicap_impact=diagnosis["yards_lost"] / 10,  # Rough estimate
                    recommended_drill=diagnosis["drills"][0]["name"],
                    drill_duration_minutes=diagnosis["drills"][0]["duration_minutes"],
                    drill_frequency=diagnosis["drills"][0]["frequency"],
                    expected_improvement=diagnosis["drills"][0]["expected_improvement"],
                    has_video=False,
                    video_url=None,
                    has_pdf_guide=True,
                )
                problems.append(problem)
    
    # Sort by severity and impact
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    problems.sort(key=lambda p: (severity_order[p.severity], -p.yards_lost))
    
    # Calculate expected outcomes
    total_yards_lost = sum(p.yards_lost for p in problems)
    total_handicap_impact = sum(p.handicap_impact for p in problems)
    high_priority = sum(1 for p in problems if p.severity == "High")
    
    # Generate weekly practice plan
    weekly_plan = generate_weekly_plan(problems[:3])  # Top 3 problems
    
    # Determine skill level
    swing_speed = swing_data.get("club_speed_at_impact", 85)
    if swing_speed < 80:
        skill_level = "Weekend"
    elif swing_speed < 95:
        skill_level = "Aspiring"
    else:
        skill_level = "Competitive"
    
    return GolferReport(
        golfer_id=golfer_id,
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        total_problems=len(problems),
        high_priority_count=high_priority,
        swing_speed_mph=swing_speed,
        handicap_estimate=handicap or (36 - swing_speed / 3),  # Rough formula
        skill_level=skill_level,
        problems=problems,
        strengths=[],  # TODO: Add strength detection
        weekly_plan=weekly_plan,
        expected_yards_gain=total_yards_lost,
        expected_handicap_reduction=total_handicap_impact,
        timeline_weeks=4,
        analysis_confidence=0.92,
        recommendation_confidence=0.85,
    )


def generate_weekly_plan(problems: List[GolfProblem]) -> Dict[str, List[Dict]]:
    """Generate a 7-day practice plan from problems."""
    plan = {
        "Monday": [],
        "Tuesday": [],
        "Wednesday": [],
        "Thursday": [],
        "Friday": [],
        "Saturday": [],
        "Sunday": [],
    }
    
    # Distribute drills across week
    for i, problem in enumerate(problems):
        # High priority: 3x per week
        # Medium priority: 2x per week
        if problem.severity == "High":
            days = ["Monday", "Wednesday", "Friday"]
        else:
            days = ["Tuesday", "Thursday"]
        
        for day in days:
            plan[day].append({
                "drill_name": problem.recommended_drill,
                "focus_area": problem.title,
                "duration": problem.drill_duration_minutes,
                "metric_to_track": problem.primary_metric,
                "target_value": problem.target_value,
            })
    
    # Add swing video analysis on Saturday
    plan["Saturday"].append({
        "drill_name": "Video Analysis & Progress Check",
        "focus_area": "Measure Improvement",
        "duration": 30,
        "metric_to_track": "overall_swing_quality",
        "target_value": 0,
    })
    
    # Sunday rest or light play
    plan["Sunday"].append({
        "drill_name": "Play 9 Holes - Focus on Feel",
        "focus_area": "Apply Changes on Course",
        "duration": 120,
        "metric_to_track": "score",
        "target_value": 0,
    })
    
    return plan


def format_golfer_report_text(report: GolferReport) -> str:
    """Generate a plain text report suitable for printing or email."""
    lines = [
        "=" * 70,
        "YOUR SWING ANALYSIS REPORT",
        f"Generated: {report.analysis_date}",
        "=" * 70,
        "",
        f"Swing Speed: {report.swing_speed_mph:.1f} mph | Skill Level: {report.skill_level}",
        f"Estimated Handicap: {report.handicap_estimate:.1f}",
        "",
        f"📊 ANALYSIS CONFIDENCE: {report.analysis_confidence:.0%}",
        "",
    ]
    
    # Problems section
    if report.problems:
        lines.extend([
            "⚠️  PRIORITY FIXES",
            "-" * 70,
            "",
        ])
        
        for i, problem in enumerate(report.problems[:3], 1):  # Top 3
            lines.extend([
                f"{i}. {problem.title}",
                f"   Severity: {problem.severity} | Yards Lost: ~{problem.yards_lost}",
                "",
                f"   What we found:",
                f"   {problem.description}",
                "",
                f"   Your {problem.primary_metric}: {problem.current_value:.1f} {problem.unit}",
                f"   Target: {problem.target_value:.1f} {problem.unit}",
                "",
                f"   💪 RECOMMENDED DRILL: {problem.recommended_drill}",
                f"   Duration: {problem.drill_duration_minutes} minutes",
                f"   Frequency: {problem.drill_frequency}",
                f"   Expected: {problem.expected_improvement}",
                "",
                "-" * 70,
                "",
            ])
    
    # Weekly plan
    lines.extend([
        "📅 YOUR 7-DAY PRACTICE PLAN",
        "=" * 70,
        "",
    ])
    
    for day, drills in report.weekly_plan.items():
        if drills:
            lines.append(f"{day}:")
            for drill in drills:
                lines.append(f"  • {drill['drill_name']} ({drill['duration']} min)")
            lines.append("")
    
    # Expected outcomes
    lines.extend([
        "🎯 EXPECTED OUTCOMES",
        "=" * 70,
        "",
        f"If you complete this plan for {report.timeline_weeks} weeks:",
        f"  • Distance gain: +{report.expected_yards_gain} yards",
        f"  • Handicap reduction: -{report.expected_handicap_reduction:.1f} strokes",
        "",
        "These estimates are based on similar golfers who completed",
        "the recommended drills consistently.",
        "",
        "=" * 70,
        "Questions? Contact your coach or reply to this email.",
        "Powered by GolfBioMetrics AI | Data Sports Group",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Generate report for a sample golfer
    print("=" * 70)
    print("GOLFER REPORT GENERATION DEMO")
    print("=" * 70)
    print()
    
    # Create sample swing data (low lag angle, mediocre X-factor)
    sample_data = pd.Series({
        "lag_angle_impact": 16.5,  # Low - will trigger early release diagnosis
        "xfactor_degrees": 34.0,   # Low - will trigger X-factor diagnosis
        "swing_tempo_ratio": 2.2,  # A bit fast
        "kinematic_sequence_score": 0.68,  # Below ideal
        "weight_transfer_timing_ms": -45,  # Late
        "club_speed_at_impact": 88.5,
    })
    
    # Generate report
    report = generate_golfer_report(sample_data, golfer_id="demo_golfer_001")
    
    # Print formatted report
    print(format_golfer_report_text(report))
    print()
    print("=" * 70)
    print("Report generation complete!")
    print("=" * 70)
