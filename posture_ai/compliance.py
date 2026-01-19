def calculate_compliance_percentage(final_iso: dict):
    """
    Robust ISO 9241-5 compliance calculator.
    Returns only overall compliance percentage.
    """

    def score_section(section: dict):
        total = 0
        score = 0

        for _, item in section.items():

            # 🚫 Skip raw numbers (numpy.float64, int, float)
            if not isinstance(item, dict):
                continue

            # ✅ Case 1: direct posture metric
            if "status" in item:
                total += 1
                if item["status"] == "ideal":
                    score += 1
                elif item["status"] == "warning":
                    score += 0.5

            # ✅ Case 2: nested workstation rules
            else:
                for _, rule in item.items():

                    if not isinstance(rule, dict):
                        continue

                    status = rule.get("status")
                    if status is None:
                        continue

                    total += 1
                    if status == "ideal":
                        score += 1
                    elif status == "warning":
                        score += 0.5

        return round((score / total) * 100, 1) if total else 0.0

    posture_score = score_section(final_iso.get("posture", {}))
    workstation_score = score_section(final_iso.get("workstation", {}))

    overall_score = round((posture_score + workstation_score) / 2, 1)

    return overall_score