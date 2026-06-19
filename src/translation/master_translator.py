"""
master_translator.py
Master Translation Layer for GolfBioMetrics.

Purpose: Unified interface to convert ML outputs into user-facing reports.
Orchestrates golfer, coach, and equipment fitter dashboards.

Usage:
    from src.translation.master_translator import MasterTranslator
    
    translator = MasterTranslator()
    
    # Generate all three reports
    golfer_report = translator.for_golfer(swing_data, golfer_id="john_001")
    coach_report = translator.for_coach(session_history, student_id="john_001")
    equipment_report = translator.for_fitter(swing_data, env_data, golfer_id="john_001")
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import asdict

# Import individual translators
from golfer_report import (
    generate_golfer_report,
    format_golfer_report_text,
    GolfProblem,
    GolferReport,
)
from coach_dashboard import (
    generate_coach_report,
    format_coach_report_text,
    CoachReport,
)
from equipment_fitter import (
    generate_equipment_profile,
    format_equipment_report,
    EquipmentProfile,
)


class MasterTranslator:
    """
    Master class for translating ML metrics to all user types.
    
    This is the main interface for the Translation Layer.
    """
    
    def __init__(self):
        """Initialize the master translator."""
        self.version = "1.0.0"
        self.supported_users = ["golfer", "coach", "equipment_fitter"]
    
    def for_golfer(
        self,
        swing_data: pd.Series,
        golfer_id: str = "unknown",
        handicap: Optional[float] = None,
        format_output: str = "text"
    ) -> str:
        """
        Generate golfer-facing report.
        
        Args:
            swing_data: Single swing analysis (58 features)
            golfer_id: Golfer identifier
            handicap: Known handicap (optional)
            format_output: "text", "dict", or "json"
            
        Returns:
            Formatted report string (or dict if format_output="dict")
        """
        report = generate_golfer_report(swing_data, golfer_id, handicap)
        
        if format_output == "text":
            return format_golfer_report_text(report)
        elif format_output == "dict":
            return self._report_to_dict(report)
        elif format_output == "json":
            import json
            return json.dumps(self._report_to_dict(report), indent=2)
        else:
            raise ValueError(f"Unknown format: {format_output}")
    
    def for_coach(
        self,
        session_history: List[Dict],
        student_name: str = "Unknown",
        student_id: str = "unknown",
        coach_id: str = "coach_001",
        format_output: str = "text"
    ) -> str:
        """
        Generate coach-facing report.
        
        Args:
            session_history: List of session data chronologically
            student_name: Student name
            student_id: Student identifier
            coach_id: Coach identifier
            format_output: "text", "dict", or "json"
            
        Returns:
            Formatted report string
        """
        report = generate_coach_report(student_id, student_name, session_history, coach_id)
        
        if format_output == "text":
            return format_coach_report_text(report)
        elif format_output == "dict":
            return self._coach_report_to_dict(report)
        elif format_output == "json":
            import json
            return json.dumps(self._coach_report_to_dict(report), indent=2)
        else:
            raise ValueError(f"Unknown format: {format_output}")
    
    def for_fitter(
        self,
        swing_data: pd.Series,
        environmental_conditions: Dict,
        golfer_id: str = "unknown",
        format_output: str = "text"
    ) -> str:
        """
        Generate equipment fitter report.
        
        Args:
            swing_data: Swing analysis (58 features)
            environmental_conditions: Dict with elevation, temp, wind
            golfer_id: Golfer identifier
            format_output: "text", "dict", or "json"
            
        Returns:
            Formatted report string
        """
        profile = generate_equipment_profile(swing_data, environmental_conditions, golfer_id)
        
        if format_output == "text":
            return format_equipment_report(profile)
        elif format_output == "dict":
            return self._equipment_report_to_dict(profile)
        elif format_output == "json":
            import json
            return json.dumps(self._equipment_report_to_dict(profile), indent=2)
        else:
            raise ValueError(f"Unknown format: {format_output}")
    
    def generate_all(
        self,
        swing_data: pd.Series,
        session_history: List[Dict] = None,
        environmental_conditions: Dict = None,
        golfer_id: str = "unknown",
        student_name: str = "Unknown",
    ) -> Dict[str, str]:
        """
        Generate all three reports at once.
        
        Returns:
            Dictionary with "golfer", "coach", "equipment_fitter" keys
        """
        results = {}
        
        # Golfer report
        results["golfer"] = self.for_golfer(swing_data, golfer_id, format_output="text")
        
        # Coach report (if session history provided)
        if session_history:
            results["coach"] = self.for_coach(
                session_history, student_name, golfer_id, format_output="text"
            )
        else:
            results["coach"] = "No session history provided. Cannot generate coach report."
        
        # Equipment report (if environmental conditions provided)
        if environmental_conditions:
            results["equipment_fitter"] = self.for_fitter(
                swing_data, environmental_conditions, golfer_id, format_output="text"
            )
        else:
            results["equipment_fitter"] = "No environmental data provided. Cannot generate equipment report."
        
        return results
    
    def _report_to_dict(self, report: GolferReport) -> Dict:
        """Convert GolferReport to dictionary."""
        return {
            "golfer_id": report.golfer_id,
            "analysis_date": report.analysis_date,
            "total_problems": report.total_problems,
            "high_priority_count": report.high_priority_count,
            "swing_speed_mph": report.swing_speed_mph,
            "handicap_estimate": report.handicap_estimate,
            "skill_level": report.skill_level,
            "problems": [
                {
                    "title": p.title,
                    "description": p.description,
                    "severity": p.severity,
                    "metric": p.primary_metric,
                    "current": p.current_value,
                    "target": p.target_value,
                    "yards_lost": p.yards_lost,
                    "drill": p.recommended_drill,
                    "expected_improvement": p.expected_improvement,
                }
                for p in report.problems
            ],
            "weekly_plan": report.weekly_plan,
            "expected_yards_gain": report.expected_yards_gain,
            "expected_handicap_reduction": report.expected_handicap_reduction,
            "timeline_weeks": report.timeline_weeks,
            "analysis_confidence": report.analysis_confidence,
        }
    
    def _coach_report_to_dict(self, report: CoachReport) -> Dict:
        """Convert CoachReport to dictionary."""
        return {
            "coach_id": report.coach_id,
            "student": {
                "name": report.student.name,
                "handicap": report.student.handicap,
                "total_sessions": report.student.total_sessions,
                "improvement_rate": report.student.improvement_rate,
            },
            "insights": [
                {
                    "title": i.title,
                    "category": i.category,
                    "description": i.description,
                    "impact": i.impact_level,
                    "action": i.recommended_action,
                    "lesson_plan": i.lesson_plan_suggestion,
                }
                for i in report.insights
            ],
            "this_session_focus": report.this_session_focus,
            "suggested_drills": report.suggested_drills,
            "next_milestone": report.next_milestone,
            "estimated_sessions": report.estimated_sessions_to_goal,
            "model_confidence": report.model_confidence,
        }
    
    def _equipment_report_to_dict(self, profile: EquipmentProfile) -> Dict:
        """Convert EquipmentProfile to dictionary."""
        return {
            "golfer_id": profile.golfer_id,
            "swing_physics": {
                "speed": profile.swing_speed_mph,
                "tempo": profile.tempo_ratio,
                "transition": profile.transition_aggression,
                "release": profile.release_timing,
                "xfactor": profile.xfactor_degrees,
                "sequence_quality": profile.kinematic_sequence_quality,
            },
            "driver_recommendations": profile.driver_recommendations,
            "priorities": profile.priorities_ranked,
            "special_considerations": profile.special_considerations,
        }


# ============================================================================
# DEMO / EXAMPLE USAGE
# ============================================================================

def demo_translation_layer():
    """
    Demonstrate the complete Translation Layer.
    Shows how ML outputs become user-facing reports.
    """
    print("=" * 80)
    print("GOLFBIOMETRICS TRANSLATION LAYER DEMO")
    print("Converting ML metrics to actionable golf intelligence")
    print("=" * 80)
    print()
    
    # Initialize translator
    translator = MasterTranslator()
    
    # Sample swing data (low lag angle, mediocre X-factor)
    sample_swing = pd.Series({
        "lag_angle_impact": 16.5,  # Poor - early release
        "xfactor_degrees": 34.0,   # Below average
        "swing_tempo_ratio": 2.2,  # Fast
        "kinematic_sequence_score": 0.68,  # Needs work
        "weight_transfer_timing_ms": -45,  # Late
        "club_speed_at_impact": 88.5,
        "age": 35,
        "handicap": 18,
    })
    
    # Session history (for coach)
    session_history = [
        {
            "date": "2026-05-01",
            "kinematic_sequence_score": 0.68,
            "xfactor_degrees": 35,
            "lag_angle_impact": 16,
            "club_speed_at_impact": 88,
        },
        {
            "date": "2026-05-15",
            "kinematic_sequence_score": 0.72,
            "xfactor_degrees": 36,
            "lag_angle_impact": 17,
            "club_speed_at_impact": 89,
        },
    ]
    
    # Environmental conditions (for fitter)
    env_conditions = {
        "elevation_m": 1609,  # Denver altitude
        "temperature_c": 25,
        "wind_typical": "Moderate",
    }
    
    golfer_id = "demo_golfer_001"
    student_name = "John Demo"
    
    # 1. GOLFER REPORT
    print("🏌️ SECTION 1: GOLFER REPORT")
    print("=" * 80)
    print()
    print("(What the golfer sees: plain English, actionable advice)")
    print("-" * 80)
    print()
    
    golfer_report = translator.for_golfer(
        sample_swing, golfer_id, handicap=18, format_output="text"
    )
    print(golfer_report[:1500] + "\n... [truncated for demo] ...\n")
    
    # 2. COACH REPORT
    print()
    print("👨‍🏫 SECTION 2: COACH DASHBOARD")
    print("=" * 80)
    print()
    print("(What the coach sees: progress tracking, lesson planning)")
    print("-" * 80)
    print()
    
    coach_report = translator.for_coach(
        session_history, student_name, golfer_id, format_output="text"
    )
    print(coach_report[:1500] + "\n... [truncated for demo] ...\n")
    
    # 3. EQUIPMENT FITTER REPORT
    print()
    print("⛳ SECTION 3: EQUIPMENT FITTER REPORT")
    print("=" * 80)
    print()
    print("(What the fitter sees: shaft recommendations, loft adjustments)")
    print("-" * 80)
    print()
    
    fitter_report = translator.for_fitter(
        sample_swing, env_conditions, golfer_id, format_output="text"
    )
    print(fitter_report)
    
    print()
    print("=" * 80)
    print("TRANSLATION LAYER DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  • Same ML backend (58 features, R² 0.85-0.96)")
    print("  • Three different user experiences")
    print("  • Golfers get drills and practice plans")
    print("  • Coaches get progress tracking and insights")
    print("  • Fitters get shaft specs and environmental adjustments")
    print()
    print("This is how we turn 'R² = 0.849' into 'Here's how to fix your slice.'")
    print("=" * 80)


if __name__ == "__main__":
    demo_translation_layer()
