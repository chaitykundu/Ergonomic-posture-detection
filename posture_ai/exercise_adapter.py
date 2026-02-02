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
# Legacy / ISO → Canonical mapping
# -------------------------------------------------

LEGACY_TO_CANONICAL = {
    # Neck
    "neck": BodyRegion.NECK,

    # Upper body
    "shoulders": BodyRegion.SHOULDER_UPPER_BACK,
    "upper_back": BodyRegion.SHOULDER_UPPER_BACK,

    # Spine
    "thoracic": BodyRegion.THORACIC,
    "mid_back": BodyRegion.THORACIC,
    "lumbar": BodyRegion.LUMBAR,
    "lower_back": BodyRegion.LUMBAR,

    # Upper limb
    "arm": BodyRegion.ARM_ELBOW_WRIST_HAND,
    "elbow": BodyRegion.ARM_ELBOW_WRIST_HAND,
    "wrists": BodyRegion.ARM_ELBOW_WRIST_HAND,
    "hands": BodyRegion.ARM_ELBOW_WRIST_HAND,

    # Pelvis / hip
    "hip": BodyRegion.HIP_PELVIS_GLUTES,
    "pelvis": BodyRegion.HIP_PELVIS_GLUTES,
    "glutes": BodyRegion.HIP_PELVIS_GLUTES,

    # Lower limb
    "knee": BodyRegion.LEG_KNEE_LOWERLEG,
    "lower_leg": BodyRegion.LEG_KNEE_LOWERLEG,

    # Foot / ankle
    "ankle": BodyRegion.FOOT_ANKLE,
    "foot": BodyRegion.FOOT_ANKLE,
}


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
    """

    pain_intensity: Dict[str, int] = {}
    body_regions = set()

    # -------------------------------
    # 1. POSTURE → BODY REGIONS
    # -------------------------------
    POSTURE_TO_REGION = {
        "neck_flexion": "neck",
        "shoulder_elevation": "shoulders",
        "elbow_angle": "elbows/forearms",
        "wrist_deviation": "wrists_hands",
        "pelvic_tilt": "lower_back",
    }

    posture = final_iso.get("posture", {})
    for metric, report in posture.items():
        severity = report.get("severity")

        if severity not in ["yellow", "red"]:
            continue

        region = POSTURE_TO_REGION.get(metric)
        if not region:
            continue

        body_regions.add(region)

        if severity == "yellow":
            pain = 4
        elif severity == "red":
            pain = 8
        else:
            pain = 0

        pain_intensity[region] = max(
            pain_intensity.get(region, 0),
            pain
        )

    # -------------------------------
    # 2. WORKSTATION → BODY REGIONS
    # -------------------------------
    WORKSTATION_TO_REGION = {
        "monitor": "neck",
        "worksurface": "wrists_hands",
        "chair": "lower_back",
    }

    workstation = final_iso.get("workstation", {})
    for component, rules in workstation.items():
        for _, report in rules.items():
            severity = report.get("severity")

            if severity not in ["yellow", "red"]:
                continue

            region = WORKSTATION_TO_REGION.get(component)
            if not region:
                continue

            body_regions.add(region)
            pain_intensity[region] = max(
                pain_intensity.get(region, 0),
                6 if severity == "yellow" else 8
            )

    # -------------------------------
    # 3. SYMPTOM FALLBACK (IMPORTANT)
    # -------------------------------
    if not body_regions:
        symptom_fallback = {
            "tingling": "wrists_hands",
            "stiffness": "neck",
        }
        for symptom in user_context.get("optional_symptoms", []):
            region = symptom_fallback.get(symptom)
            if region:
                body_regions.add(region)
                pain_intensity[region] = 5

    return {
        "user_id": user_context.get("user_id"),
        "image_data": user_context.get("image_data"),
        "body_regions": list(body_regions),
        "pain_intensity": pain_intensity,
        "duration_pattern": user_context.get("duration_pattern"),
        "work_pattern": user_context.get("work_pattern"),
        "optional_symptoms": user_context.get("optional_symptoms", []),
    }
