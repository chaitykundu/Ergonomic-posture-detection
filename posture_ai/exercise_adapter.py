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
    into canonical exercise onboarding format
    """

    pain_intensity: Dict[str, int] = {}
    body_regions = set()

    # Extract problematic ISO regions
    for iso_region, report in final_iso.get("regions", {}).items():
        if report.get("status") in ["bad", "warning"]:
            legacy_key = iso_region.lower()

            canonical_region = LEGACY_TO_CANONICAL.get(legacy_key)
            if canonical_region:
                body_regions.add(canonical_region.value)
                pain_intensity[canonical_region.value] = report.get(
                    "severity", 5
                )

    return {
        "user_id": user_context.get("user_id"),
        "image_data": user_context.get("image_data"),
        "body_regions": list(body_regions),          # canonical only
        "pain_intensity": pain_intensity,            # keyed by canonical region
        "duration_pattern": user_context.get("duration_pattern"),
        "work_pattern": user_context.get("work_pattern"),
        "optional_symptoms": user_context.get(
            "optional_symptoms", []
        ),
    }
