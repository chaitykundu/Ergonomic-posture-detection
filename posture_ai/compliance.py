# -------------------------------
# Severity contribution (ISO-aligned)
# -------------------------------
SEVERITY_SCORE = {
    "green": 1.0,     # compliant
    "yellow": 0.4,   # deviation (closer to red than green)
    "red": 0.0        # violation
}

# -------------------------------
# Ergonomics-based weights
# -------------------------------
POSTURE_WEIGHTS = {
    "neck_flexion": 2.5,
    "wrist_deviation": 2.3,
    "pelvic_tilt": 1.8,          # trunk inclination proxy
    "elbow_angle": 1.2,
    "shoulder_elevation": 0.9
}

WORKSTATION_WEIGHTS = {
    "MonitorHeight": 2.0,
    "ViewingDistance": 1.5,
    "WorksurfaceHeight": 1.8,
    "ThighClearance": 1.2,
    "SeatDepth": 1.7
}


# -------------------------------
# Weighted scoring helper
# -------------------------------
def score_items_iso(items: dict, weights: dict, critical=None):
    critical = set(critical or [])

    total_weight = 0.0
    weighted_sum = 0.0
    yellow = red = 0
    critical_red = False

    for name, item in items.items():
        sev = item.get("severity")
        if sev not in SEVERITY_SCORE:
            continue

        w = weights.get(name, 1.0)
        s = SEVERITY_SCORE[sev]

        total_weight += w
        weighted_sum += (s ** 1.7) * w  # non-linear ISO penalty

        if sev == "yellow":
            yellow += 1
        elif sev == "red":
            red += 1
            if name in critical:
                critical_red = True

    if total_weight == 0:
        return 0.0

    score = (weighted_sum / total_weight) * 100

    # -------- ISO accumulation penalties --------
    if yellow >= 2:
        score *= 0.85
    if yellow >= 4:
        score *= 0.70

    # -------- red dominance --------
    if red >= 2:
        score = min(score, 35)

    # -------- critical joint override --------
    if critical_red:
        score = min(score, 30)

    return round(max(0, min(score, 100)), 1)


# -------------------------------
# Main compliance calculation
# -------------------------------
def calculate_compliance_percentage(final_iso: dict):
    posture = final_iso.get("posture", {})

    workstation = {}
    for component in final_iso.get("workstation", {}).values():
        workstation.update(component)

    posture_score = score_items_iso(
        posture,
        POSTURE_WEIGHTS,
        critical=["neck_flexion", "wrist_deviation"]
    )

    workstation_score = score_items_iso(
        workstation,
        WORKSTATION_WEIGHTS
    )

    # ISO: posture dominates environment
    overall = (posture_score * 0.65) + (workstation_score * 0.35)

    # Hard ISO non-compliance floor
    if posture_score < 30:
        overall = min(overall, 35)

    return round(max(0, min(overall, 100)), 1)

def iso_status(compliance_percent: float):
    if compliance_percent >= 70:
        return "green", "Compliant"
    elif compliance_percent >= 40:
        return "yellow", "Corrective action required"
    else:
        return "red", "Non-compliant – immediate correction required"

