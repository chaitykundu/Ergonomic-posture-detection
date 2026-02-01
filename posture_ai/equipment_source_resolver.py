"""
Equipment Source Resolver
-------------------------
Centralized authority mapping for ergonomic equipment recommendations.

Why this exists:
- LLMs are unreliable at citing correct ergonomic authorities
- Sources must be deterministic for compliance (ISO / OSHA / DLI)

This module enforces correct sources AFTER AI generation.
"""

# --------------------------------------------------
# Canonical equipment → authority mapping
# --------------------------------------------------
EQUIPMENT_SOURCE_MAP = {
    "monitor_riser": [
        "Washington State DLI",
        "ISO 9241-5"
    ],
    "ergonomic_keyboard": [
        "OSHA Guidelines",
        "ISO 9241-5"
    ],
    "ergonomic_mouse": [
        "OSHA Guidelines",
        "ISO 9241-5"
    ],
    "lumbar_support": [
        "Washington State DLI",
        "ISO 9241-5"
    ],
    "chair": [
        "Washington State DLI",
        "OSHA"
    ],
    "footrest": [
        "Cornell Ergo",
        "ISO 9241-5"
    ],
    "default": [
        "General Ergonomic Guidelines"
    ]
}


# --------------------------------------------------
# Normalize equipment name → canonical key
# --------------------------------------------------
def normalize_equipment_key(name: str) -> str:
    """
    Converts free-text equipment names into canonical keys
    so we can reliably assign sources.
    """
    if not name:
        return "default"

    name = name.lower()

    if "monitor" in name or "riser" in name or "arm" in name:
        return "monitor_riser"
    if "keyboard" in name:
        return "ergonomic_keyboard"
    if "mouse" in name:
        return "ergonomic_mouse"
    if "lumbar" in name or "back" in name:
        return "lumbar_support"
    if "chair" in name or "seating" in name:
        return "chair"
    if "foot" in name:
        return "footrest"

    return "default"


# --------------------------------------------------
# Public API: enforce sources on AI output
# --------------------------------------------------
def apply_equipment_sources(equipment_recommendations: list) -> list:
    """
    Enforces correct ergonomic sources on equipment recommendations.

    Args:
        equipment_recommendations (list): AI-generated equipment list

    Returns:
        list: Equipment list with corrected `source` fields
    """
    if not isinstance(equipment_recommendations, list):
        return []

    resolved = []

    for item in equipment_recommendations:
        name = item.get("name", "")
        key = normalize_equipment_key(name)

        sources = EQUIPMENT_SOURCE_MAP.get(
            key,
            EQUIPMENT_SOURCE_MAP["default"]
        )

        item["source"] = "; ".join(sources)
        resolved.append(item)

    return resolved
