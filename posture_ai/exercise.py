"""
exercise.py

Core logic for exercise recommendation based on onboarding data
and backend exercise catalog response.

UPDATED: Matches backend exercise database body regions exactly
"""

# -------------------------------------------------
# 🔴 RED-FLAG MESSAGE (GLOBAL, FIXED)
# -------------------------------------------------

RED_FLAG_LLM_MESSAGE = (
    "THIS IS A RED-FLAG CASE.\n\n"
    "The reported pain intensity is very high and may indicate a serious condition. "
    "No exercises are recommended at this time. "
    "Please consult a qualified healthcare professional "
    "(doctor or physiotherapist) for proper evaluation and treatment."
)

from typing import Dict, List
from collections import defaultdict

# -------------------------------------------------------------------
# Body Region Mapping
# Maps to EXACT backend exercise database body_region values
# Based on the screenshot you provided
# -------------------------------------------------------------------

BACKEND_BODY_REGIONS = [
    "Neck",
    "Shoulders",
    "Elbows/Forearms",
    "Wrists/Hands",
    "Hips/Glutes",
    "Knees",
    "Ankles/Feet"
]

# Normalization map for different input formats
BODY_REGION_NORMALIZE_MAP = {
    # Neck
    "neck": "Neck",
    
    # Shoulders
    "shoulders": "Shoulders",
    "shoulder": "Shoulders",
    "upper_back": "Shoulders",
    
    # Elbows/Forearms
    "elbows/forearms": "Elbows/Forearms",
    "elbow": "Elbows/Forearms",
    "forearms": "Elbows/Forearms",
    "forearm": "Elbows/Forearms",
    
    # Wrists/Hands
    "wrists/hands": "Wrists/Hands",
    "wrist": "Wrists/Hands",
    "wrists": "Wrists/Hands",
    "hands": "Wrists/Hands",
    "hand": "Wrists/Hands",
    "wrists_hands": "Wrists/Hands",
    
    # Hips/Glutes
    "hips/glutes": "Hips/Glutes",
    "hip": "Hips/Glutes",
    "hips": "Hips/Glutes",
    "glutes": "Hips/Glutes",
    "pelvis": "Hips/Glutes",
    "lower_back": "Hips/Glutes",
    "lumbar": "Hips/Glutes",
    "hips_glutes": "Hips/Glutes",
    
    # Knees
    "knees": "Knees",
    "knee": "Knees",
    "lower_leg": "Knees",
    
    # Ankles/Feet
    "ankles/feet": "Ankles/Feet",
    "ankle": "Ankles/Feet",
    "ankles": "Ankles/Feet",
    "foot": "Ankles/Feet",
    "feet": "Ankles/Feet",
    "ankles_feet": "Ankles/Feet",
}

MIN_EXERCISES_PER_REGION = 3


# -------------------------------------------------------------------
# Region Normalizer
# -------------------------------------------------------------------

def normalize_region(raw_region: str) -> str | None:
    """
    Normalizes any region string to backend canonical value.
    Returns None if region is not recognized.
    """
    if not raw_region:
        return None
    
    # Clean input
    key = raw_region.strip().lower()
    
    # Direct match with backend regions (case-insensitive)
    for backend_region in BACKEND_BODY_REGIONS:
        if key == backend_region.lower():
            return backend_region
    
    # Try normalization map
    normalized = BODY_REGION_NORMALIZE_MAP.get(key)
    if normalized:
        return normalized
    
    print(f"⚠️ Warning: Unknown region '{raw_region}' - no matching backend region")
    return None


# -------------------------------------------------------------------
# Normalization Layer
# Converts raw onboarding data into internal decision-friendly format
# -------------------------------------------------------------------

def normalize_onboarding(onboarding: Dict) -> Dict:
    raw_pain_map = onboarding.get("pain_intensity", {})
    pain_map = {}
    
    # 🔹 Normalize pain map keys
    for raw_region, pain in raw_pain_map.items():
        normalized = normalize_region(raw_region)
        if normalized:
            # Use normalized region as key (e.g., "Neck", "Shoulders")
            pain_map[normalized] = max(pain_map.get(normalized, 0), pain)
        else:
            print(f"⚠️ Skipping unknown region in pain map: {raw_region}")
    
    # Sort regions by pain level (highest first)
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
        "pain_map": pain_map,  # Now uses normalized region names
        "condition_type": condition_type,
        "symptoms": onboarding.get("optional_symptoms", []),
        "min_exercises": 3 if condition_type == "acute" else 5,
    }


# -------------------------------------------------------------------
# Body Region Resolver
# Converts onboarding regions to backend-compatible regions
# -------------------------------------------------------------------

def resolve_body_regions(onboarding_regions: List[str]) -> List[str]:
    """
    Ensures all body regions match backend exercise database.
    """
    resolved: List[str] = []
    
    for region in onboarding_regions:
        mapped = normalize_region(region)
        if mapped and mapped not in resolved:
            resolved.append(mapped)
    
    print(f"\n📍 Resolved body regions: {resolved}")
    return resolved


# -------------------------------------------------------------------
# Exercise Filtering
# Hard safety & relevance filtering
# -------------------------------------------------------------------

def validate_exercise(ex: Dict) -> bool:
    """Validates that exercise has required fields and valid region."""
    if not ex.get("id"):
        return False
    if not ex.get("body_region"):
        return False
    if normalize_region(ex["body_region"]) is None:
        print(f"⚠️ Invalid region in exercise: {ex.get('body_region')}")
        return False
    return True


def filter_exercises_by_region(
    exercises: List[Dict],
    allowed_regions: List[str]
) -> List[Dict]:
    """
    Filters exercises to only those matching allowed regions.
    """
    filtered = []
    
    for ex in exercises:
        ex_region = normalize_region(ex.get("body_region"))
        if ex_region in allowed_regions:
            filtered.append(ex)
    
    print(f"📊 Filtered {len(filtered)} exercises from {len(exercises)} total")
    return filtered


def resolve_pain(region: str, pain_map: Dict[str, int]) -> int:
    """
    Returns pain level for a region.
    Handles both normalized and non-normalized region names.
    """
    if not region:
        return 0
    
    # Try direct lookup
    if region in pain_map:
        return pain_map[region]
    
    # Try normalized version
    normalized = normalize_region(region)
    if normalized in pain_map:
        return pain_map[normalized]
    
    return 0


# -------------------------------------------------------------------
# Scoring Logic
# Pain-aware prioritization on top of backend score
# -------------------------------------------------------------------

def pain_weighted_score(exercise: Dict, pain_map: Dict) -> int:
    """
    Calculates weighted score based on exercise score and pain level.
    """
    base_score = exercise.get("score", 0)
    
    raw_region = exercise.get("body_region")
    normalized_region = normalize_region(raw_region)
    
    pain_level = resolve_pain(normalized_region, pain_map)
    
    # Boost score for high-pain regions
    if pain_level >= 8:
        return base_score + 3
    elif pain_level >= 5:
        return base_score + 1
    
    return base_score


def rank_exercises(
    exercises: List[Dict],
    pain_map: Dict
) -> List[Dict]:
    """Ranks exercises by pain-weighted score."""
    return sorted(
        exercises,
        key=lambda ex: pain_weighted_score(ex, pain_map),
        reverse=True
    )


def get_vas_tier(pain: int) -> str:
    """Categorizes pain level into tier."""
    if pain >= 8:
        return "HIGH"
    elif pain >= 4:
        return "MED"
    return "LOW"


# -------------------------------------------------------------------
# Intent rules per VAS tier
# -------------------------------------------------------------------

VAS_INTENT_RULES = {
    "LOW": {"strengthening"},
    "MED": {"mobility", "stretching"},
    "HIGH": {"isometric", "relief"},
}


def is_red_flag_case(pain_map: dict) -> bool:
    """Check if any region has pain >= 9."""
    return any(pain >= 9 for pain in pain_map.values())


MAX_BY_TIER = {
    "HIGH": 2,
    "MED": 1,
    "LOW": 1,
}


# -------------------------------------------------------------------
# Session Builder
# Builds a safe, region-aware exercise session
# -------------------------------------------------------------------

def build_region_based_session(
    exercises: list,
    pain_map: dict,
    max_total: int = 5,
) -> list:
    """
    Builds exercise session ensuring coverage of all painful regions.
    
    Strategy:
    1. Guarantee at least 1 exercise per painful region (PASS 1)
    2. Add more exercises based on pain tier limits (PASS 2)
    """
    
    # Group exercises by normalized region
    region_buckets = defaultdict(list)
    
    for ex in exercises:
        region = normalize_region(ex.get("body_region"))
        if not region:
            continue
        
        # Only include regions that have pain
        if region not in pain_map:
            continue
        
        region_buckets[region].append(ex)
    
    session = []
    
    print(f"\n{'='*60}")
    print(f"BUILDING EXERCISE SESSION")
    print(f"{'='*60}")
    print(f"Pain map: {pain_map}")
    print(f"Available regions: {list(region_buckets.keys())}")
    
    # PASS 1 — Guarantee 1 exercise per painful region
    for region in sorted(
        region_buckets.keys(),
        key=lambda r: pain_map.get(r, 0),
        reverse=True
    ):
        region_exs = region_buckets[region]
        pain = pain_map.get(region, 0)
        
        if pain <= 0:
            continue
        
        ranked = rank_exercises(region_exs, pain_map)
        
        if ranked:
            session.append(ranked[0])
            print(f"✓ Added 1 exercise for {region} (pain: {pain})")
    
    # PASS 2 — Add more exercises based on tier limits
    for region, region_exs in region_buckets.items():
        pain = pain_map.get(region, 0)
        
        if pain <= 0:
            continue
        
        tier = get_vas_tier(pain)
        limit = MAX_BY_TIER[tier]
        
        # Count how many exercises we already have for this region
        already = [
            ex for ex in session
            if normalize_region(ex.get("body_region")) == region
        ]
        
        remaining = limit - len(already)
        
        if remaining <= 0:
            continue
        
        # Add more exercises up to tier limit
        ranked = rank_exercises(region_exs, pain_map)
        session.extend(ranked[1:1 + remaining])
        
        print(f"✓ Added {remaining} more exercises for {region} (tier: {tier})")
    
    print(f"\nTotal exercises in session: {len(session)}")
    print(f"{'='*60}\n")
    
    return session[:max_total]


# -------------------------------------------------------------------
# Body Region Badge Text
# -------------------------------------------------------------------

def get_region_badge(pain_level: int) -> Dict:
    """Returns badge color and text based on pain level."""
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
# AI Guidance Generator
# -------------------------------------------------------------------

def generate_ai_exercise_guidance(
    exercise: Dict,
    condition_type: str,
    pain_map: Dict
) -> Dict:
    """
    Generates personalized guidance for each exercise.
    """
    
    region = normalize_region(exercise["body_region"])
    title = exercise["title"]
    score = exercise.get("score", 5)
    
    pain_level = resolve_pain(region, pain_map)
    
    # Duration logic
    if pain_level >= 8:
        duration = "8–12 seconds"
    elif score >= 8:
        duration = "15–20 seconds"
    else:
        duration = "10–15 seconds"
    
    # Safety logic
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
    
    # Badge logic
    badge = "🔴 Therapy Priority" if pain_level > 4 else "🟢 Maintenance"
    
    # Improvement percentage logic
    base_improvement = score * 5
    
    if pain_level >= 8:
        base_improvement += 30
    elif pain_level >= 4:
        base_improvement += 15
    else:
        base_improvement += 5
    
    improvement_percentage = min(base_improvement, 100)
    
    # Description
    description = (
        f"{title} is a controlled movement designed to improve mobility "
        f"and reduce stiffness in the {region} area."
    )
    
    return {
        #"description": description,
        "recommended_duration": duration,
        "safety_note": safety_note,
        "region_vas": pain_level,
        "badge": badge,
        "improvement_percentage": improvement_percentage
    }


def calculate_average_pain(pain_map: Dict[str, int]) -> int:
    """Calculates average pain across all regions."""
    if not pain_map:
        return 0
    return round(sum(pain_map.values()) / len(pain_map))


def get_main_pain_region(pain_map: Dict[str, int]) -> Dict | None:
    """Identifies the region(s) with highest pain."""
    if not pain_map:
        return None
    
    max_pain = max(pain_map.values())
    regions = [
        region
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
    Main exercise recommendation function.
    
    Args:
        onboarding_data: User context with body_regions and pain_intensity
        exercise_api_response: Response from backend exercise API
    
    Returns:
        Dict with exercise recommendations and metadata
    """
    
    print(f"\n{'='*60}")
    print(f"EXERCISE RECOMMENDATION ENGINE")
    print(f"{'='*60}")
    
    # 1. Normalize onboarding
    normalized = normalize_onboarding(onboarding_data)
    
    resolved_regions = resolve_body_regions(
        onboarding_data.get("body_regions", [])
    )
    
    print(f"Resolved regions: {resolved_regions}")
    print(f"Pain map: {normalized['pain_map']}")
    
    # 2. RED-FLAG CHECK
    if is_red_flag_case(normalized["pain_map"]):
        print("🔴 RED FLAG CASE DETECTED")
        return {
            "condition_type": normalized["condition_type"],
            "focus_regions": resolved_regions,
            "red_flag": True,
            "average_pain_vas": max(normalized["pain_map"].values()),
            "recommended_session": [],
            "llm_message": RED_FLAG_LLM_MESSAGE
        }
    
    # 3. Normal flow
    main_pain_region = get_main_pain_region(normalized["pain_map"])
    average_pain = calculate_average_pain(normalized["pain_map"])
    
    raw_exercises = exercise_api_response.get("exercises_list", [])
    
    print(f"Raw exercises from API: {len(raw_exercises)}")
    
    # Validate exercises
    validated_exercises = [
        ex for ex in raw_exercises
        if validate_exercise(ex)
    ]
    
    print(f"Validated exercises: {len(validated_exercises)}")
    
    # Filter by region
    filtered = filter_exercises_by_region(
        validated_exercises,
        resolved_regions
    )
    
    print(f"Filtered exercises: {len(filtered)}")
    
    # Rank by pain-weighted score
    ranked = rank_exercises(filtered, normalized["pain_map"])
    
    # Build session
    session = build_region_based_session(
        ranked,
        normalized["pain_map"]
    )
    
    print(f"\n✓ Final session: {len(session)} exercises")
    print(f"{'='*60}\n")
    
    return {
        "condition_type": normalized["condition_type"],
        "focus_regions": resolved_regions,
        "main_pain_region": main_pain_region,
        "average_pain_vas": average_pain,
        "recommended_session": [
            {
                "id": ex["id"],
                "title": ex["title"],
                "Purpose": ex.get("purpose", ""),
                "body_region": ex["body_region"],
                "muscles_addressed": ex.get("muscles_addressed", ""),
                "video": ex.get("video", ""),
                "description": ex.get("description", ""),
                "Contraindications":ex.get("contraindications", ""),
                "recommended_sets": 2,
                **generate_ai_exercise_guidance(
                    ex,
                    normalized["condition_type"],
                    normalized["pain_map"]
                )
            }
            for ex in session
        ]
    }