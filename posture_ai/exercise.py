"""
exercise.py

Core logic for exercise recommendation based on onboarding data
and backend exercise catalog response.

Author: AI Recommendation Module
"""

from typing import Dict, List


# -------------------------------------------------------------------
# Body Region Mapping
# Maps onboarding body regions to backend exercise body_region values
# -------------------------------------------------------------------

BODY_REGION_MAP = {
    "neck": ["neck"],
    "upper_back": ["upper_back", "shoulders"],
    "wrists_hands": ["wrists"],
}


# -------------------------------------------------------------------
# Normalization Layer
# Converts raw onboarding data into internal decision-friendly format
# -------------------------------------------------------------------

def normalize_onboarding(onboarding: Dict) -> Dict:
    pain_map = onboarding.get("pain_intensity", {})

    priority_regions = sorted(
        pain_map.keys(),
        key=lambda k: pain_map[k],
        reverse=True
    )

    condition_type = (
        "acute"
        if onboarding.get("duration_pattern") == "less_than_1_week"
        else "chronic"
    )

    return {
        "user_id": onboarding.get("user_id"),
        "priority_regions": priority_regions,
        "pain_map": pain_map,
        "condition_type": condition_type,
        "symptoms": onboarding.get("optional_symptoms", []),
        "max_exercises": 3 if condition_type == "acute" else 5,
    }


# -------------------------------------------------------------------
# Body Region Resolver
# Converts onboarding regions to backend-compatible regions
# -------------------------------------------------------------------

def resolve_body_regions(onboarding_regions: List[str]) -> List[str]:
    resolved = set()
    for region in onboarding_regions:
        mapped = BODY_REGION_MAP.get(region, [])
        resolved.update(mapped)
    return list(resolved)


# -------------------------------------------------------------------
# Exercise Filtering
# Hard safety & relevance filtering
# -------------------------------------------------------------------

def filter_exercises_by_region(
    exercises: List[Dict],
    allowed_regions: List[str]
) -> List[Dict]:
    return [
        ex for ex in exercises
        if ex.get("body_region") in allowed_regions
    ]


# -------------------------------------------------------------------
# Scoring Logic
# Pain-aware prioritization on top of backend score
# -------------------------------------------------------------------

def pain_weighted_score(exercise: Dict, pain_map: Dict) -> int:
    base_score = exercise.get("score", 0)
    region = exercise.get("body_region")

    pain_level = pain_map.get(region, 5)

    if pain_level >= 8:
        return base_score + 3
    elif pain_level >= 5:
        return base_score + 1
    return base_score


def rank_exercises(
    exercises: List[Dict],
    pain_map: Dict
) -> List[Dict]:
    return sorted(
        exercises,
        key=lambda ex: pain_weighted_score(ex, pain_map),
        reverse=True
    )


# -------------------------------------------------------------------
# Session Builder
# Builds a safe, non-repetitive exercise session
# -------------------------------------------------------------------

def build_acute_session(
    exercises: List[Dict],
    max_exercises: int
) -> List[Dict]:
    session = []
    used_regions = set()

    for ex in exercises:
        region = ex.get("body_region")

        if region in used_regions:
            continue

        session.append(ex)
        used_regions.add(region)

        if len(session) >= max_exercises:
            break

    return session


# -------------------------------------------------------------------
# Public API
# Main function used by views / endpoints
# -------------------------------------------------------------------

def recommend_exercises(
    onboarding_data: Dict,
    exercise_api_response: Dict
) -> Dict:
    """
    Main recommendation entry point
    """

    normalized = normalize_onboarding(onboarding_data)

    exercises = exercise_api_response.get("exercises_list", [])

    resolved_regions = resolve_body_regions(
        onboarding_data.get("body_regions", [])
    )

    filtered = filter_exercises_by_region(
        exercises,
        resolved_regions
    )

    ranked = rank_exercises(
        filtered,
        normalized["pain_map"]
    )

    session = build_acute_session(
        ranked,
        normalized["max_exercises"]
    )

    return {
        "user_id": normalized["user_id"],
        "condition_type": normalized["condition_type"],
        "focus_regions": resolved_regions,
        "recommended_session": [
            {
                "id": ex["id"],
                "title": ex["title"],
                "body_region": ex["body_region"],
                "description": ex["description"],
                "video": ex["video"],
                "recommended_sets": 2,
                "recommended_duration": "15–20 seconds",
                "safety_note": (
                    "Stop immediately if pain increases "
                    "or tingling worsens"
                ),
            }
            for ex in session
        ]
    }
