"""
coach_dashboard.py
Translation Layer: ML metrics for golf coaches.

Purpose: Give coaches data-driven insights to validate their intuition,
track student progress, and differentiate their teaching with technology.

Key Outputs:
- Student progress tracking over time
- Before/after swing comparisons
- Biomechanics insights for lesson planning
- Benchmark comparisons to tour averages
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class StudentProfile:
    """Coach-facing student information."""
    student_id: str
    name: str
    age: int
    handicap: float
    primary_goal: str  # "Distance", "Accuracy", "Consistency", "Injury Prevention"
    
    # Session history
    total_sessions: int
    first_session_date: str
    last_session_date: str
    
    # Progress metrics
    baseline_metrics: Dict[str, float]  # First session
    current_metrics: Dict[str, float]    # Latest session
    best_metrics: Dict[str, float]       # Personal bests
    
    # Improvement tracking
    improvement_rate: Dict[str, float]  # Change per session
    

@dataclass
class CoachInsight:
    """Data-driven insight for coaching decisions."""
    insight_id: str
    category: str  # "Technique", "Physical", "Equipment", "Practice"
    
    title: str
    description: str
    supporting_data: str  # ML metric that backs this up
    
    # Coaching action
    recommended_action: str
    lesson_plan_suggestion: str
    
    # Priority
    impact_level: str  # "High", "Medium", "Low"
    effort_required: str  # "Easy", "Moderate", "Challenging"
    
    # Evidence
    trend_chart: Optional[str]  # Reference to visualization
    comparison_to_pros: Dict[str, float]


@dataclass
class CoachReport:
    """Complete coach-facing report for a student."""
    coach_id: str
    student: StudentProfile
    
    # Analysis
    insights: List[CoachInsight]
    
    # Session recommendations
    this_session_focus: str
    suggested_drills: List[Dict]
    suggested_duration: int  # minutes
    
    # Long-term planning
    next_milestone: str
    estimated_sessions_to_goal: int
    
    # ML Confidence
    model_confidence: float


# ============================================================================
# COACH INSIGHT DATABASE
# ============================================================================

COACH_INSIGHTS_DATABASE = {
    # Lag Angle Insights
    "lag_angle_low": {
        "category": "Technique",
        "title": "Early Release Pattern Detected",
        "description": "Student is releasing club early, losing 15-20 yards. Wrist angle at impact is {value}° vs. tour average of 26°.",
        "supporting_metrics": ["lag_angle_impact", "club_speed_at_impact"],
        "lesson_plan": "Focus on lag maintenance drills. Start with pump drill, progress to towel-under-arm for feel.",
        "physical_limitation": False,
        "equipment_factor": False,
    },
    
    # X-Factor Insights
    "xfactor_low": {
        "category": "Physical",
        "title": "Limited Hip Mobility Identified",
        "description": "X-Factor of {value}° suggests hip flexibility limitation. Tour pros average 48°.",
        "supporting_metrics": ["xfactor_degrees", "xfactor_timing"],
        "lesson_plan": "Combine technique work with flexibility training. Chair drill daily + recommend yoga/Pilates.",
        "physical_limitation": True,
        "equipment_factor": False,
    },
    
    "xfactor_improving": {
        "category": "Technique",
        "title": "X-Factor Improving - Add Speed Work",
        "description": "X-Factor improved from {baseline}° to {current}°. Ready for speed training.",
        "supporting_metrics": ["xfactor_degrees", "swing_speed_trend"],
        "lesson_plan": "Graduate from flexibility drills to speed training. Add overspeed sticks.",
        "physical_limitation": False,
        "equipment_factor": False,
    },
    
    # Sequence Insights
    "sequence_poor": {
        "category": "Technique",
        "title": "Downswing Sequence Needs Reordering",
        "description": "Kinematic sequence score of {value} indicates arms/club firing before hips. Classic reverse sequence.",
        "supporting_metrics": ["kinematic_sequence_score", "weight_transfer_timing_ms"],
        "lesson_plan": "Step-through drill and hip-only downswing practice. Use ground force plate if available.",
        "physical_limitation": False,
        "equipment_factor": False,
    },
    
    # Tempo Insights
    "tempo_fast": {
        "category": "Technique",
        "title": "Rushed Transition - Tempo Training Needed",
        "description": "Tempo ratio of {value} is too quick. Backswing should be 3x longer than downswing.",
        "supporting_metrics": ["swing_tempo_ratio", "kinematic_sequence_score"],
        "lesson_plan": "Metronome training. Start at 50% speed, build to full. Focus on 'pause' at top.",
        "physical_limitation": False,
        "equipment_factor": False,
    },
    
    # Equipment Insights
    "shaft_mismatch": {
        "category": "Equipment",
        "title": "Shaft Flex May Be Mismatched",
        "description": "Tempo of {tempo} with late release suggests current shaft may be too soft. Student overloading shaft.",
        "supporting_metrics": ["swing_tempo_ratio", "lag_angle_impact", "club_speed_at_impact"],
        "lesson_plan": "Club fitting session. Test X-stiff shafts with lower torque. Measure dispersion improvement.",
        "physical_limitation": False,
        "equipment_factor": True,
    },
    
    "loft_adjustment": {
        "category": "Equipment",
        "title": "Consider Loft Adjustment for Environment",
        "description": "Student plays at {elevation}m elevation. Current driver loft may be suboptimal.",
        "supporting_metrics": ["elevation_m", "air_density_factor", "ball_speed_mph"],
        "lesson_plan": "TrackMan fitting with altitude adjustment. Consider +1° loft for Denver/altitude play.",
        "physical_limitation": False,
        "equipment_factor": True,
    },
    
    # Injury Prevention
    "injury_risk_high": {
        "category": "Physical",
        "title": "⚠️ High Injury Risk Detected",
        "description": "Compensatory patterns and reverse pivot suggest elevated lower back risk (risk score: {value}).",
        "supporting_metrics": ["injury_risk_score", "compensation_severity", "reverse_pivot_flag"],
        "lesson_plan": "URGENT: Focus on mechanics before distance. Reduce practice volume 50%. Add core strengthening.",
        "physical_limitation": True,
        "equipment_factor": False,
    },
    
    # Strengths
    "tempo_excellent": {
        "category": "Technique",
        "title": "Strength: Excellent Tempo",
        "description": "Tempo ratio of {value} is tour-level. Build on this strength with consistency training.",
        "supporting_metrics": ["swing_tempo_ratio", "tempo_consistency"],
        "lesson_plan": "Use tempo as foundation. Student has natural rhythm - focus on other priorities.",
        "physical_limitation": False,
        "equipment_factor": False,
        "is_strength": True,
    },
}


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_student_progress_chart(
    session_history: List[Dict[str, float]],
    metrics: List[str] = None
) -> Dict[str, List]:
    """
    Generate time-series data for student progress visualization.
    
    Args:
        session_history: List of session data over time
        metrics: List of metrics to track
        
    Returns:
        Dictionary with dates and values for each metric
    """
    if metrics is None:
        metrics = [
            "kinematic_sequence_score",
            "xfactor_degrees",
            "lag_angle_impact",
            "club_speed_at_impact",
        ]
    
    chart_data = {
        "dates": [],
    }
    for metric in metrics:
        chart_data[metric] = []
    
    for session in session_history:
        chart_data["dates"].append(session.get("date", "Unknown"))
        for metric in metrics:
            chart_data[metric].append(session.get(metric, None))
    
    return chart_data


def compare_to_tour_averages(
    student_metrics: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Compare student metrics to tour averages.
    
    Returns:
        Dictionary with comparison for each metric
    """
    TOUR_AVERAGES = {
        "kinematic_sequence_score": {"mean": 0.92, "std": 0.04, "unit": "score"},
        "xfactor_degrees": {"mean": 48, "std": 4, "unit": "degrees"},
        "lag_angle_impact": {"mean": 26, "std": 3, "unit": "degrees"},
        "swing_tempo_ratio": {"mean": 3.0, "std": 0.2, "unit": "ratio"},
        "club_speed_at_impact": {"mean": 105, "std": 8, "unit": "mph"},
        "weight_transfer_timing_ms": {"mean": -100, "std": 15, "unit": "ms"},
    }
    
    comparisons = {}
    
    for metric, tour_stats in TOUR_AVERAGES.items():
        if metric in student_metrics:
            student_value = student_metrics[metric]
            tour_mean = tour_stats["mean"]
            tour_std = tour_stats["std"]
            
            # Calculate percentile (rough normal distribution assumption)
            z_score = (student_value - tour_mean) / tour_std
            percentile = 50 + (z_score * 34)  # Rough approximation
            percentile = max(1, min(99, percentile))  # Clamp to 1-99
            
            comparisons[metric] = {
                "student_value": student_value,
                "tour_average": tour_mean,
                "difference": student_value - tour_mean,
                "percentile_vs_tour": round(percentile, 1),
                "gap_to_tour": f"{abs(student_value - tour_mean):.1f} {tour_stats['unit']}",
                "status": "Above Tour" if percentile > 75 else 
                         "Near Tour" if percentile > 50 else 
                         "Below Tour" if percentile > 25 else 
                         "Needs Work",
            }
    
    return comparisons


def generate_coach_insights(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    improvement_rates: Dict[str, float]
) -> List[CoachInsight]:
    """
    Generate data-driven coaching insights.
    
    Returns:
        List of CoachInsight objects sorted by priority
    """
    insights = []
    
    # Check each metric for insights
    # 1. Lag Angle
    if current_metrics.get("lag_angle_impact", 25) < 22:
        template = COACH_INSIGHTS_DATABASE["lag_angle_low"]
        insights.append(CoachInsight(
            insight_id="lag_001",
            category=template["category"],
            title=template["title"],
            description=template["description"].format(
                value=current_metrics["lag_angle_impact"]
            ),
            supporting_data=f"lag_angle_impact: {current_metrics['lag_angle_impact']:.1f}°",
            recommended_action="Implement lag maintenance drills",
            lesson_plan_suggestion=template["lesson_plan"],
            impact_level="High",
            effort_required="Moderate",
            trend_chart=None,
            comparison_to_pros={
                "student": current_metrics["lag_angle_impact"],
                "tour_avg": 26,
                "gap": 26 - current_metrics["lag_angle_impact"],
            }
        ))
    
    # 2. X-Factor
    if current_metrics.get("xfactor_degrees", 45) < 40:
        # Check if improving
        if improvement_rates.get("xfactor_degrees", 0) > 0.5:
            template = COACH_INSIGHTS_DATABASE["xfactor_improving"]
            insights.append(CoachInsight(
                insight_id="xfactor_improving_001",
                category=template["category"],
                title=template["title"],
                description=template["description"].format(
                    baseline=baseline_metrics.get("xfactor_degrees", 35),
                    current=current_metrics["xfactor_degrees"]
                ),
                supporting_data=f"xfactor_degrees improved {improvement_rates['xfactor_degrees']:.2f}° per session",
                recommended_action="Graduate to speed training",
                lesson_plan_suggestion=template["lesson_plan"],
                impact_level="Medium",
                effort_required="Easy",
                trend_chart=None,
                comparison_to_pros={
                    "student": current_metrics["xfactor_degrees"],
                    "tour_avg": 48,
                    "gap": 48 - current_metrics["xfactor_degrees"],
                }
            ))
        else:
            template = COACH_INSIGHTS_DATABASE["xfactor_low"]
            insights.append(CoachInsight(
                insight_id="xfactor_low_001",
                category=template["category"],
                title=template["title"],
                description=template["description"].format(
                    value=current_metrics["xfactor_degrees"]
                ),
                supporting_data=f"xfactor_degrees: {current_metrics['xfactor_degrees']:.1f}°",
                recommended_action="Flexibility training + technique work",
                lesson_plan_suggestion=template["lesson_plan"],
                impact_level="High",
                effort_required="Challenging",
                trend_chart=None,
                comparison_to_pros={
                    "student": current_metrics["xfactor_degrees"],
                    "tour_avg": 48,
                    "gap": 48 - current_metrics["xfactor_degrees"],
                }
            ))
    
    # 3. Sequence
    if current_metrics.get("kinematic_sequence_score", 0.85) < 0.75:
        template = COACH_INSIGHTS_DATABASE["sequence_poor"]
        insights.append(CoachInsight(
            insight_id="seq_001",
            category=template["category"],
            title=template["title"],
            description=template["description"].format(
                value=current_metrics["kinematic_sequence_score"]
            ),
            supporting_data=f"kinematic_sequence_score: {current_metrics['kinematic_sequence_score']:.2f}",
            recommended_action="Downswing sequence drills",
            lesson_plan_suggestion=template["lesson_plan"],
            impact_level="High",
            effort_required="Moderate",
            trend_chart=None,
            comparison_to_pros={
                "student": current_metrics["kinematic_sequence_score"],
                "tour_avg": 0.92,
                "gap": 0.92 - current_metrics["kinematic_sequence_score"],
            }
        ))
    
    # 4. Injury Risk (if high)
    if current_metrics.get("injury_risk_score", 0) > 0.3:
        template = COACH_INSIGHTS_DATABASE["injury_risk_high"]
        insights.append(CoachInsight(
            insight_id="injury_001",
            category=template["category"],
            title=template["title"],
            description=template["description"].format(
                value=current_metrics["injury_risk_score"]
            ),
            supporting_data=f"injury_risk_score: {current_metrics['injury_risk_score']:.2f}",
            recommended_action="URGENT: Reduce practice volume, focus on mechanics",
            lesson_plan_suggestion=template["lesson_plan"],
            impact_level="High",
            effort_required="Easy",
            trend_chart=None,
            comparison_to_pros={
                "student": current_metrics["injury_risk_score"],
                "tour_avg": 0.15,
                "gap": current_metrics["injury_risk_score"] - 0.15,
            }
        ))
    
    # Sort by impact level
    impact_order = {"High": 0, "Medium": 1, "Low": 2}
    insights.sort(key=lambda x: impact_order[x.impact_level])
    
    return insights


def generate_coach_report(
    student_id: str,
    student_name: str,
    session_history: List[Dict],
    coach_id: str = "coach_001"
) -> CoachReport:
    """
    Generate complete coach-facing report.
    
    Args:
        student_id: Student identifier
        student_name: Student name
        session_history: List of session data chronologically
        coach_id: Coach identifier
        
    Returns:
        CoachReport with insights and recommendations
    """
    if not session_history:
        raise ValueError("Session history required")
    
    # Extract metrics
    baseline = session_history[0]
    current = session_history[-1]
    
    # Calculate improvement rates
    num_sessions = len(session_history)
    improvement_rates = {}
    
    key_metrics = [
        "kinematic_sequence_score",
        "xfactor_degrees",
        "lag_angle_impact",
        "club_speed_at_impact",
    ]
    
    for metric in key_metrics:
        if metric in baseline and metric in current:
            total_change = current[metric] - baseline[metric]
            improvement_rates[metric] = total_change / num_sessions
    
    # Create student profile
    student = StudentProfile(
        student_id=student_id,
        name=student_name,
        age=current.get("age", 35),
        handicap=current.get("handicap", 18),
        primary_goal=current.get("goal", "Improvement"),
        total_sessions=num_sessions,
        first_session_date=baseline.get("date", "Unknown"),
        last_session_date=current.get("date", "Unknown"),
        baseline_metrics={k: v for k, v in baseline.items() if isinstance(v, (int, float))},
        current_metrics={k: v for k, v in current.items() if isinstance(v, (int, float))},
        best_metrics={},  # TODO: Calculate from history
        improvement_rate=improvement_rates,
    )
    
    # Generate insights
    insights = generate_coach_insights(
        student.current_metrics,
        student.baseline_metrics,
        improvement_rates
    )
    
    # Determine this session focus
    if insights:
        this_session_focus = insights[0].title
        suggested_drills = [
            {"name": insights[0].recommended_action, "duration": 15}
        ]
    else:
        this_session_focus = "General technique review"
        suggested_drills = []
    
    return CoachReport(
        coach_id=coach_id,
        student=student,
        insights=insights,
        this_session_focus=this_session_focus,
        suggested_drills=suggested_drills,
        suggested_duration=60,
        next_milestone="Break 80" if student.handicap > 10 else "Single digits",
        estimated_sessions_to_goal=8,
        model_confidence=0.88,
    )


def format_coach_report_text(report: CoachReport) -> str:
    """Generate plain text coach report."""
    lines = [
        "=" * 80,
        "COACH DASHBOARD REPORT",
        f"Student: {report.student.name} | Handicap: {report.student.handicap}",
        f"Sessions: {report.student.total_sessions} | Last: {report.student.last_session_date}",
        "=" * 80,
        "",
    ]
    
    # Progress summary
    lines.extend([
        "📊 PROGRESS SUMMARY",
        "-" * 80,
        "",
    ])
    
    for metric, rate in report.student.improvement_rate.items():
        direction = "▲" if rate > 0 else "▼"
        lines.append(f"{metric}: {direction} {abs(rate):.2f} per session")
    
    lines.extend(["", "-" * 80, ""])
    
    # Insights
    if report.insights:
        lines.extend([
            "💡 DATA-DRIVEN INSIGHTS",
            "-" * 80,
            "",
        ])
        
        for i, insight in enumerate(report.insights[:3], 1):
            lines.extend([
                f"{i}. {insight.title}",
                f"   Category: {insight.category} | Impact: {insight.impact_level}",
                "",
                f"   {insight.description}",
                "",
                f"   Supporting Data: {insight.supporting_data}",
                "",
                f"   💪 Recommended Action: {insight.recommended_action}",
                f"   📋 Lesson Plan: {insight.lesson_plan_suggestion}",
                "",
            ])
            
            if insight.comparison_to_pros:
                cp = insight.comparison_to_pros
                lines.append(f"   vs Tour: Student {cp['student']:.1f} | Tour Avg {cp['tour_avg']:.1f} | Gap: {cp['gap']:.1f}")
                lines.append("")
            
            lines.append("-" * 80)
            lines.append("")
    
    # This session
    lines.extend([
        "🎯 THIS SESSION",
        "-" * 80,
        "",
        f"Focus: {report.this_session_focus}",
        f"Suggested Duration: {report.suggested_duration} minutes",
        "",
        "Drills:",
    ])
    
    for drill in report.suggested_drills:
        lines.append(f"  • {drill['name']} ({drill['duration']} min)")
    
    lines.extend([
        "",
        "-" * 80,
        "",
        f"🎯 NEXT MILESTONE: {report.next_milestone}",
        f"   Estimated: {report.estimated_sessions_to_goal} more sessions",
        "",
        f"Model Confidence: {report.model_confidence:.0%}",
        "",
        "=" * 80,
    ])
    
    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("COACH DASHBOARD DEMO")
    print("=" * 80)
    print()
    
    # Create sample session history
    session_history = [
        {
            "date": "2026-05-01",
            "kinematic_sequence_score": 0.68,
            "xfactor_degrees": 35,
            "lag_angle_impact": 16,
            "club_speed_at_impact": 88,
            "age": 35,
            "handicap": 18,
        },
        {
            "date": "2026-05-08",
            "kinematic_sequence_score": 0.71,
            "xfactor_degrees": 36,
            "lag_angle_impact": 17,
            "club_speed_at_impact": 89,
            "age": 35,
            "handicap": 17,
        },
        {
            "date": "2026-05-15",
            "kinematic_sequence_score": 0.74,
            "xfactor_degrees": 38,
            "lag_angle_impact": 18,
            "club_speed_at_impact": 90,
            "age": 35,
            "handicap": 16,
        },
    ]
    
    report = generate_coach_report(
        student_id="stu_001",
        student_name="John Doe",
        session_history=session_history,
        coach_id="coach_001"
    )
    
    print(format_coach_report_text(report))
    print()
    print("=" * 80)
    print("Coach dashboard demo complete!")
    print("=" * 80)
