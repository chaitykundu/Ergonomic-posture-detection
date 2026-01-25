# posture_ai/exercise_adapter.py

from typing import Dict


ISO_REGION_MAP = {
    "neck": "neck",
    "upper_back": "upper_back",
    "shoulders": "upper_back",
    "wrists": "wrists_hands",
}


def build_exercise_onboarding(
    user_context: Dict,
    final_iso: Dict
) -> Dict:
    """
    Converts posture + workstation ISO analysis
    into exercise onboarding format
    """

    pain_intensity = {}
    body_regions = set()

    # Example: extract problematic regions from ISO
    for region, report in final_iso.get("regions", {}).items():
        if report.get("status") in ["bad", "warning"]:
            mapped = ISO_REGION_MAP.get(region)
            if mapped:
                body_regions.add(mapped)
                pain_intensity[mapped] = report.get("severity", 5)

    return {
        "user_id": user_context["user_id"],
        "image_data": user_context["image_data"],
        "body_regions": list(body_regions),
        "pain_intensity": pain_intensity,
        "duration_pattern": user_context.get("duration_pattern"),
        "work_pattern": user_context.get("work_pattern"),
        "optional_symptoms": user_context.get("optional_symptoms", []),
    }
