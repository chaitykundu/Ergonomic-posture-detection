SEVERITY_SCORE = {
    "green": 1.0,
    "yellow": 0.5,
    "red": 0.0
}

def calculate_compliance_percentage(final_iso: dict):
    def score_items(items):
        total = 0
        score = 0.0

        for item in items:
            severity = item.get("severity")
            if severity not in SEVERITY_SCORE:
                continue

            total += 1
            score += SEVERITY_SCORE[severity]

        return (score / total) * 100 if total else 0.0

    posture_items = final_iso.get("posture", {}).values()

    workstation_items = []
    for comp in final_iso.get("workstation", {}).values():
        workstation_items.extend(comp.values())

    posture_score = score_items(posture_items)
    workstation_score = score_items(workstation_items)

    overall = (posture_score + workstation_score) / 2

    # ISO safety override (recommended)
    if any(i.get("severity") == "red" for i in posture_items):
        overall = min(overall, 40)

    return round(overall, 1)
