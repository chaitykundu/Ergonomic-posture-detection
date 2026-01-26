# posture_ai/red_flag_detector.py

from typing import Dict, Tuple


def detect_red_flags(onboarding: Dict) -> Tuple[bool, str]:
    """
    Returns (is_red_flag, reason)
    """

    red_flags = onboarding.get("red_flags", {})
    vas = onboarding.get("vas_score", 0)
    tingling_progressive = onboarding.get("tingling_progressive", False)

    if red_flags.get("recent_trauma"):
        return True, "Recent trauma or accident"

    if red_flags.get("severe_night_pain"):
        return True, "Severe night pain"

    if red_flags.get("fever"):
        return True, "Fever present"

    if red_flags.get("unexplained_weight_loss"):
        return True, "Unexplained weight loss"

    if red_flags.get("rapid_weakness"):
        return True, "Rapidly worsening weakness or numbness"

    if red_flags.get("bladder_bowel_issues"):
        return True, "Loss of bladder or bowel control"

    if vas >= 9:
        return True, "Very high pain level (VAS 9–10)"

    if vas >= 7 and tingling_progressive:
        return True, "Progressive tingling with high pain"

    return False, ""
