# ==========================================================
# ISO Ergonomic Compliance Scoring System (Balanced Model)
# ==========================================================

# -------------------------------
# Severity contribution (ISO-aligned)
# -------------------------------
SEVERITY_SCORE = {
    "green": 1.0,      # fully compliant
    "yellow": 0.65,    # moderate deviation
    "red": 0.0         # violation
}

# Non-linear control (mild curvature)
POWER = 1.2  # reduced from 1.7 (less aggressive)


# -------------------------------
# Ergonomics-based weights
# -------------------------------
POSTURE_WEIGHTS = {
    "neck_flexion": 2.5,
    "wrist_deviation": 2.3,
    "pelvic_tilt": 1.8,
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


# ==========================================================
# Weighted ISO Scoring Engine
# ==========================================================
def score_items_iso(items: dict, weights: dict, critical=None):
    critical = set(critical or [])

    total_weight = 0.0
    weighted_sum = 0.0
    yellow = 0
    red = 0
    critical_red = False

    for name, item in items.items():
        sev = item.get("severity")
        if sev not in SEVERITY_SCORE:
            continue

        weight = weights.get(name, 1.0)
        severity_value = SEVERITY_SCORE[sev]

        total_weight += weight
        weighted_sum += (severity_value ** POWER) * weight

        if sev == "yellow":
            yellow += 1
        elif sev == "red":
            red += 1
            if name in critical:
                critical_red = True

    # No data case
    if total_weight == 0:
        return None

    score = (weighted_sum / total_weight) * 100

    # -------------------------------
    # Accumulation penalties
    # -------------------------------
    if yellow >= 2:
        score *= 0.90
    if yellow >= 4:
        score *= 0.80

    # -------------------------------
    # Smooth red penalties
    # -------------------------------
    if red >= 2:
        score *= 0.55
    if red >= 4:
        score *= 0.40

    # -------------------------------
    # Critical override (policy-based)
    # -------------------------------
    if critical_red:
        score = min(score, 30)

    return round(max(0, min(score, 100)), 1)


# ==========================================================
# Main Compliance Calculation
# ==========================================================
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

    # -------------------------------
    # Weighted blending
    # If workstation missing, don't penalize
    # -------------------------------
    if posture_score is None and workstation_score is None:
        overall = 0.0
    elif workstation_score is None:
        overall = posture_score
    elif posture_score is None:
        overall = workstation_score
    else:
        overall = (posture_score * 0.65) + (workstation_score * 0.35)

    # ISO floor rule
    if posture_score is not None and posture_score < 30:
        overall = min(overall, 35)

    return round(max(0, min(overall, 100)), 1)


# ==========================================================
# ISO Status Mapping
# ==========================================================
def iso_status(compliance_percent: float):

    if compliance_percent >= 70:
        return "green", "Compliant"
    elif compliance_percent >= 40:
        return "yellow", "Corrective action required"
    else:
        return "red", "Non-compliant – immediate correction required"