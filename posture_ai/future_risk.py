from typing import Dict

def calculate_future_risk(
    average_pain: int,
    affected_regions: int,
    overall_severity: str
) -> Dict:
    score = 0

    # Pain contribution
    if average_pain >= 7:
        score += 2
    elif average_pain >= 4:
        score += 1

    # Multi-region contribution
    if affected_regions >= 3:
        score += 2
    elif affected_regions == 2:
        score += 1

    # Ergonomic severity contribution
    if overall_severity == "red":
        score += 2
    elif overall_severity == "yellow":
        score += 1

    # Final classification
    if score >= 5:
        return {
            "level": "High",
            "label": "High – Priority intervention recommended",
            "color": "red"
        }
    elif score >= 3:
        return {
            "level": "Medium",
            "label": "Medium – Risk of progression if unaddressed",
            "color": "Yellow"
        }
    else:
        return {
            "level": "Low",
            "label": "Low – Maintain current programme",
            "color": "green"
        }
