from typing import Dict
from enum import Enum


# -------------------------------------------------
# Canonical Body Regions (single source of truth)
# -------------------------------------------------

class BodyRegion(str, Enum):
    NECK = "NECK"
    SHOULDER_UPPER_BACK = "SHOULDER_UPPER_BACK"
    THORACIC = "THORACIC"
    LUMBAR = "LUMBAR"
    HIP_PELVIS_GLUTES = "HIP_PELVIS_GLUTES"
    ARM_ELBOW_WRIST_HAND = "ARM_ELBOW_WRIST_HAND"
    LEG_KNEE_LOWERLEG = "LEG_KNEE_LOWERLEG"
    FOOT_ANKLE = "FOOT_ANKLE"


# -------------------------------------------------
# Backend Exercise Database Body Regions
# (from your screenshot)
# -------------------------------------------------
BACKEND_BODY_REGIONS = [
    "Neck",
    "Shoulders", 
    "Elbows/Forearms",
    "Wrists/Hands",
    "Hips/Glutes",
    "Knees",
    "Ankles/Feet"
]


# -------------------------------------------------
# Posture/Workstation → Backend Exercise Region Mapping
# This maps what we detect to what exercises are available
# -------------------------------------------------

POSTURE_WORKSTATION_TO_BACKEND = {
    # Neck issues
    "neck": "Neck",
    "neck_flexion": "Neck",
    
    # Shoulder/Upper back issues
    "shoulders": "Shoulders",
    "shoulder_elevation": "Shoulders",
    "upper_back": "Shoulders",
    
    # Arm/Elbow/Wrist issues
    "elbows/forearms": "Elbows/Forearms",
    "elbow": "Elbows/Forearms",
    "elbow_angle": "Elbows/Forearms",
    "wrists_hands": "Wrists/Hands",
    "wrist": "Wrists/Hands",
    "wrists": "Wrists/Hands",
    "hands": "Wrists/Hands",
    "wrist_deviation": "Wrists/Hands",
    
    # Lower back/pelvis issues
    "lower_back": "Hips/Glutes",  # Backend doesn't have separate lower_back
    "lumbar": "Hips/Glutes",
    "pelvic_tilt": "Hips/Glutes",
    "hip": "Hips/Glutes",
    "pelvis": "Hips/Glutes",
    "glutes": "Hips/Glutes",
    "hips_glutes": "Hips/Glutes",
    
    # Leg/Knee issues
    "knee": "Knees",
    "knees": "Knees",
    "lower_leg": "Knees",
    
    # Foot/Ankle issues
    "ankle": "Ankles/Feet",
    "ankles_feet": "Ankles/Feet",
    "foot": "Ankles/Feet",
}


# -------------------------------------------------
# Pain intensity calculation
# -------------------------------------------------

def calculate_pain_from_severity(severity: str) -> int:
    """Convert severity to pain score (0-10 VAS scale)"""
    severity_map = {
        "green": 2,   # Minor discomfort
        "yellow": 5,  # Moderate pain
        "red": 8,     # Severe pain
    }
    return severity_map.get(severity, 5)


# -------------------------------------------------
# Exercise onboarding builder
# -------------------------------------------------

def build_exercise_onboarding(
    user_context: Dict,
    final_iso: Dict
) -> Dict:
    """
    Converts posture + workstation ISO analysis
    into exercise onboarding format compatible
    with recommend_exercises()
    
    Returns body regions that match the backend exercise database.
    """

    pain_intensity: Dict[str, int] = {}
    body_regions_set = set()  # Use set to avoid duplicates
    
    # -------------------------------
    # 1. POSTURE → BODY REGIONS
    # -------------------------------
    posture = final_iso.get("posture", {})
    
    for metric, report in posture.items():
        severity = report.get("severity")
        
        if severity not in ["yellow", "red"]:
            continue
        
        # Map posture metric to backend region
        backend_region = POSTURE_WORKSTATION_TO_BACKEND.get(metric)
        
        if backend_region:
            body_regions_set.add(backend_region)
            
            # Calculate pain score
            pain_score = calculate_pain_from_severity(severity)
            
            # Keep highest pain score for each region
            pain_intensity[backend_region] = max(
                pain_intensity.get(backend_region, 0),
                pain_score
            )

    # -------------------------------
    # 2. WORKSTATION → BODY REGIONS
    # -------------------------------
    WORKSTATION_COMPONENT_MAP = {
        "monitor": "Neck",
        "worksurface": "Wrists/Hands",
        "chair": "Hips/Glutes",
    }
    
    workstation = final_iso.get("workstation", {})
    
    for component, rules in workstation.items():
        for rule_name, report in rules.items():
            severity = report.get("severity")
            
            if severity not in ["yellow", "red"]:
                continue
            
            backend_region = WORKSTATION_COMPONENT_MAP.get(component)
            
            if backend_region:
                body_regions_set.add(backend_region)
                
                pain_score = calculate_pain_from_severity(severity)
                pain_intensity[backend_region] = max(
                    pain_intensity.get(backend_region, 0),
                    pain_score
                )

    # -------------------------------
    # 3. SYMPTOM FALLBACK
    # -------------------------------
    if not body_regions_set:
        symptom_fallback = {
            "tingling": "Wrists/Hands",
            "stiffness": "Neck",
            "numbness": "Wrists/Hands",
            "pain": "Neck",
        }
        
        for symptom in user_context.get("optional_symptoms", []):
            backend_region = symptom_fallback.get(symptom)
            if backend_region:
                body_regions_set.add(backend_region)
                pain_intensity[backend_region] = 5  # Default moderate pain

    # -------------------------------
    # 4. ENSURE ALL REGIONS HAVE PAIN SCORES
    # -------------------------------
    # Make sure every region in body_regions has a pain score
    for region in body_regions_set:
        if region not in pain_intensity:
            pain_intensity[region] = 4  # Default to moderate-low if not calculated

    # Convert set to list for JSON serialization
    body_regions_list = sorted(list(body_regions_set))

    print(f"\n{'='*60}")
    print(f"EXERCISE ADAPTER OUTPUT:")
    print(f"{'='*60}")
    print(f"Body Regions: {body_regions_list}")
    print(f"Pain Intensity Map: {pain_intensity}")
    print(f"{'='*60}\n")

    return {
        "user_id": user_context.get("user_id"),
        "image_data": user_context.get("image_data"),
        "body_regions": body_regions_list,
        "pain_intensity": pain_intensity,
        "duration_pattern": user_context.get("duration_pattern"),
        "work_pattern": user_context.get("work_pattern"),
        "optional_symptoms": user_context.get("optional_symptoms", []),
    }


# -------------------------------------------------
# Helper function to validate backend compatibility
# -------------------------------------------------

def validate_backend_regions(body_regions: list) -> Dict:
    """
    Validates that all body regions match backend exercise database.
    Returns a report of valid and invalid regions.
    """
    valid_regions = []
    invalid_regions = []
    
    for region in body_regions:
        if region in BACKEND_BODY_REGIONS:
            valid_regions.append(region)
        else:
            invalid_regions.append(region)
    
    return {
        "valid": valid_regions,
        "invalid": invalid_regions,
        "all_valid": len(invalid_regions) == 0
    }