def calculate_compliance_percentage(final_iso: dict) -> dict:
    """
    Calculates ISO 9241-5 compliance percentage.
    Returns posture %, workstation %, and overall %.
    """

    def score_section(section: dict):
        total = 0
        score = 0

        for metric, data in section.items():
            if not isinstance(data, dict):
                continue

            status = data.get("status")
            if status is None:
                continue

            total += 1

            if status == "ideal":
                score += 1
            elif status == "warning":
                score += 0.5
            elif status == "violation":
                score += 0

        return (score / total * 100) if total > 0 else 0

    posture_score = score_section(final_iso.get("posture", {}))
    workstation_score = score_section(final_iso.get("workstation", {}))

    overall_score = round((posture_score + workstation_score) / 2, 1)

    return {
        "posture_compliance_percent": round(posture_score, 1),
        "workstation_compliance_percent": round(workstation_score, 1),
        "overall_compliance_percent": overall_score
    }
