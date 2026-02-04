"""
exercise.py

Core logic for exercise recommendation based on onboarding data
and backend exercise catalog response.

Author: AI Recommendation Module
"""

from typing import Dict, List
from collections import defaultdict
from posture_ai.future_risk import calculate_future_risk

# -------------------------------------------------------------------
# Body Region Mapping
# Maps onboarding body regions to backend exercise body_region values
# -------------------------------------------------------------------

BODY_REGION_MAP = {
    "neck": ["Neck"],
    "upper_back": ["Upper Back"],
    "shoulders": ["Shoulders"],
    "lower_back": ["Lower Back"],
    "elbows/forearms": ["Elbows/Forearms"],
    "wrists_hands": ["Wrists/Hands"],
    "knees": ["Knees"],
    "ankles_feet": ["Ankles/Feet"],
    "hips_glutes": ["Hips/Glutes"],
}
MIN_EXERCISES_PER_REGION = 3


# -------------------------------------------------------------------
# Region Normalizer (ADDED)
# -------------------------------------------------------------------

def normalize_region(raw_region: str) -> str | None:
    """
    Normalizes onboarding or backend region string
    to backend canonical value (e.g. 'neck' → 'Neck').
    """
    if not raw_region:
        return None

    key = raw_region.strip().lower()

    for onboarding_key, backend_regions in BODY_REGION_MAP.items():
        if key == onboarding_key.lower():
            return backend_regions[0]
        if key == backend_regions[0].lower():
            return backend_regions[0]

    return None


# -------------------------------------------------------------------
# Normalization Layer
# Converts raw onboarding data into internal decision-friendly format
# -------------------------------------------------------------------

def normalize_onboarding(onboarding: Dict) -> Dict:
    raw_pain_map = onboarding.get("pain_intensity", {})
    pain_map = {}

    # 🔹 Normalize pain map keys (ADDED)
    for raw_region, pain in raw_pain_map.items():
        normalized = normalize_region(raw_region)
        if normalized:
            pain_map[normalized.lower()] = pain

    priority_regions = sorted(
        pain_map.keys(),
        key=lambda k: pain_map[k],
        reverse=True
    )

    duration_code = onboarding.get("duration_pattern")

    DURATION_TO_CONDITION = {
        "Less than 1 week": "acute",
        "1-6 weeks": "acute",
        "More than 6 weeks": "chronic",
        "ON_OFF_MONTHS": "chronic",
    }

    condition_type = DURATION_TO_CONDITION.get(duration_code, "chronic")

    return {
        "user_id": onboarding.get("user_id"),
        "priority_regions": priority_regions,
        "pain_map": pain_map,
        "condition_type": condition_type,
        "symptoms": onboarding.get("optional_symptoms", []),
        #"max_exercises": 3 if condition_type == "acute" else 5,
        "min_exercises": 3 if condition_type == "acute" else 5,
    }


# -------------------------------------------------------------------
# Body Region Resolver
# Converts onboarding regions to backend-compatible regions
# -------------------------------------------------------------------

def resolve_body_regions(onboarding_regions: List[str]) -> List[str]:
    resolved = set()
    for region in onboarding_regions:
        mapped = normalize_region(region)
        if mapped:
            resolved.add(mapped)
    return list(resolved)


# -------------------------------------------------------------------
# Exercise Filtering
# Hard safety & relevance filtering
# -------------------------------------------------------------------

def validate_exercise(ex: Dict) -> bool:
    if not ex.get("id"):
        return False
    if not ex.get("body_region"):
        return False
    if not normalize_region(ex["body_region"]):
        return False
    return True


def filter_exercises_by_region(
    exercises: List[Dict],
    allowed_regions: List[str]
) -> List[Dict]:
    return [
        ex for ex in exercises
        if normalize_region(ex.get("body_region")) in allowed_regions
    ]


def resolve_pain(region: str | None, pain_map: Dict[str, int]) -> int:
    """
    Returns pain level for a region.
    Missing data is treated as no pain (0).
    """
    if not region:
        return 0
    return pain_map.get(region.lower(), 0)


# -------------------------------------------------------------------
# Scoring Logic
# Pain-aware prioritization on top of backend score
# -------------------------------------------------------------------

def pain_weighted_score(exercise: Dict, pain_map: Dict) -> int:
    base_score = exercise.get("score", 0)

    raw_region = exercise.get("body_region")
    normalized_region = normalize_region(raw_region)

    pain_level = resolve_pain(normalized_region, pain_map)

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

def get_vas_tier(pain: int) -> str:
    if pain >= 8:
        return "HIGH"
    elif pain >= 4:
        return "MED"
    return "LOW"

# -------------------------------------------------------------------
# Session Builder
# Builds a safe, non-repetitive exercise session
# -------------------------------------------------------------------

def build_acute_session(
    exercises: List[Dict],
    pain_map: Dict,
    min_exercises: int,
    max_exercises: int = 5
) -> List[Dict]:

    session = []
    used_regions = set()

    # ---------------------------------
    # 1. First pass: HIGH & MED regions
    # ---------------------------------
    for ex in exercises:
        region = normalize_region(ex.get("body_region"))
        if not region or region in used_regions:
            continue

        pain = pain_map.get(region.lower(), 0)
        tier = get_vas_tier(pain)

        # allow only safe intents per tier
        intent = ex.get("intent")

        if tier == "HIGH" and intent not in ["isometric", "relief"]:
            continue
        if tier == "MED" and intent not in ["mobility"]:
            continue

        session.append(ex)
        used_regions.add(region)

        if len(session) >= min_exercises:
            break

    # ---------------------------------
    # 2. Second pass: fill if needed
    # ---------------------------------
    if len(session) < min_exercises:
        for ex in exercises:
            region = normalize_region(ex.get("body_region"))
            if region in used_regions:
                continue

            session.append(ex)
            used_regions.add(region)

            if len(session) >= min_exercises:
                break

    return session[:max_exercises]
# -------------------------------------------------------------------
# Intent rules per VAS tier (ADD HERE)
# -------------------------------------------------------------------

VAS_INTENT_RULES = {
    "LOW": {"strengthening"},
    "MED": {"mobility", "stretching"},
    "HIGH": {"isometric", "relief"},
}
from collections import defaultdict

def build_region_based_session(
    exercises: list,
    pain_map: dict,
    min_total: int = 3,
    max_total: int = 5,
    max_per_region: int = 3
) -> list:

    region_buckets = defaultdict(list)

    # bucket by region
    for ex in exercises:
        region = normalize_region(ex.get("body_region"))
        if not region:
            continue
        region_buckets[region].append(ex)

    session = []
    active_regions = []

    # ---------------------------------
    # 1. Main region exercises
    # ---------------------------------
    for region, region_exs in region_buckets.items():
        pain = pain_map.get(region.lower(), 0)
        tier = get_vas_tier(pain)
        allowed_intents = VAS_INTENT_RULES[tier]

        # filter by intent
        filtered = [
        ex for ex in region_exs
        if not ex.get("intent") or ex.get("intent") in allowed_intents
    ]


        ranked = rank_exercises(filtered, pain_map)

        selected = ranked[:max_per_region]
        if selected:
            active_regions.append(tier)
            session.extend(selected)

    # ---------------------------------
    # 2. Cross-chain logic
    # ---------------------------------
    med_high_regions = sum(1 for t in active_regions if t in {"MED", "HIGH"})

    if med_high_regions >= 2:
        cross_chain = [
            ex for ex in exercises
            if ex.get("body_region") in {"Thoracic", "Upper Back"}
            and ex.get("intent") == "mobility"
        ]
        cross_chain = rank_exercises(cross_chain, pain_map)
        session.extend(cross_chain[:2])

    # ---------------------------------
    # 3. Clamp total count
    # ---------------------------------
    return session[:max_total]

# -------------------------------------------------------------------
# Body Region Badge Text
# -------------------------------------------------------------------
def get_region_badge(pain_level: int) -> Dict:
    if pain_level >= 4:
        return {
            "badge_color": "red",
            "badge_text": "🔴 Therapy Priority"
        }
    return {
        "badge_color": "green",
        "badge_text": "🟢 Maintenance"
    }


# -------------------------------------------------------------------
# AI Guidance Generator (ADD HERE)
# -------------------------------------------------------------------

def generate_ai_exercise_guidance(
    exercise: Dict,
    condition_type: str,
    pain_map: Dict
) -> Dict:
    """
    Uses AI logic to generate personalized
    description, duration, and safety notes
    based on pain level and exercise score.
    """

    region = exercise["body_region"]
    title = exercise["title"]
    score = exercise.get("score", 5)

    pain_level = pain_map.get(region.lower(), 5)

    # -----------------------------
    # Duration logic (pain + score)
    # -----------------------------
    if pain_level >= 8:
        duration = "8–12 seconds"
    elif score >= 8:
        duration = "15–20 seconds"
    else:
        duration = "10–15 seconds"

    # -----------------------------
    # Safety logic
    # -----------------------------
    if pain_level >= 8:
        safety_note = (
            "Stop immediately if pain, tingling, or numbness increases. "
            "Do not push through discomfort."
        )
    elif score < 6:
        safety_note = (
            "Perform gently and avoid forcing the movement. "
            "Stop if discomfort appears."
        )
    else:
        safety_note = (
            "Move slowly and maintain controlled breathing throughout."
        )
    # -----------------------------
    # Badge logic (NEW)
    # -----------------------------
    badge = "🔴 Therapy Priority" if pain_level > 4 else "🟢 Maintenance"

    # -----------------------------
    # Improvement percentage logic
    # -----------------------------
    base_improvement = score * 5  # score 1–10 → 5–50%

    if pain_level >= 8:
        base_improvement += 30
    elif pain_level >= 4:
        base_improvement += 15
    else:
        base_improvement += 5

    # HARD CAP
    improvement_percentage = min(base_improvement, 100)


    # -----------------------------
    # Description logic
    # -----------------------------
    description = (
        f"{title} is a controlled movement designed to improve mobility "
        f"and reduce stiffness in the {region.lower()} area."
    )
    print("AI guidance called for:", exercise["title"])

    return {
        "description": description,
        "recommended_duration": duration,
        "safety_note": safety_note,
        "region_vas": pain_level,
        "badge": badge,
        "improvement_percentage": improvement_percentage
    }

def calculate_average_pain(pain_map: Dict[str, int]) -> int:
    if not pain_map:
        return 0
    return round(sum(pain_map.values()) / len(pain_map))

def get_main_pain_region(pain_map: Dict[str, int]) -> Dict | None:
    if not pain_map:
        return None

    max_pain = max(pain_map.values())
    regions = [
        region.title()
        for region, pain in pain_map.items()
        if pain == max_pain
    ]

    return {
        "regions": regions,
        "vas": max_pain
    }

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
    main_pain_region = get_main_pain_region(normalized["pain_map"])
    average_pain = calculate_average_pain(normalized["pain_map"])
    future_risk = calculate_future_risk(
        average_pain=average_pain,
        affected_regions=len(normalized["pain_map"]),
        overall_severity=exercise_api_response.get("overall_severity", "green")
    )

    raw_exercises = exercise_api_response.get("exercises_list", [])

    validated_exercises = [
        ex for ex in raw_exercises
        if validate_exercise(ex)
    ]
    print("validation exercise",validated_exercises )

    resolved_regions = resolve_body_regions(
        onboarding_data.get("body_regions", [])
    )
    print("Resolved exercise",resolved_regions )

    filtered = filter_exercises_by_region(
        validated_exercises,
        resolved_regions
    )
    print("filtered exercise",filtered )

    ranked = rank_exercises(
        filtered,
        normalized["pain_map"]
    )

    session = build_region_based_session(
        ranked,
        normalized["pain_map"],
        #normalized["min_exercises"]
    )

    return {
    "condition_type": normalized["condition_type"],
    "focus_regions": resolved_regions,
    "main_pain_region": main_pain_region,
    "average_pain_vas": average_pain,
    "future_risk": future_risk,    
    "recommended_session": [
        {
            **{
                "id": ex["id"],
                "title": ex["title"],
                #"description":ex["description"]
                "body_region": ex["body_region"],
                "video": ex["video"],
                "recommended_sets": 2,
            },
            **generate_ai_exercise_guidance(
                ex,
                normalized["condition_type"],
                normalized["pain_map"]
            )
        }
        for ex in session
    ]
}

