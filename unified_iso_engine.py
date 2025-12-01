def merge_iso_reports(posture_report, workstation_report):
    # Start building a final combined dictionary
    final_output = {
        "posture": posture_report,
        "workstation": workstation_report,
        "overall_severity": "green"
    }

    # Collect every severity into one list
    all_severities = []

    # Posture severity
    for item in posture_report.values():
        all_severities.append(item["severity"])

    # Workstation severity
    for comp in workstation_report.values():
        for rule in comp.values():
            all_severities.append(rule["severity"])

    # Decide the final color:
    if "red" in all_severities:
        final_output["overall_severity"] = "red"
    elif "yellow" in all_severities:
        final_output["overall_severity"] = "yellow"
    else:
        final_output["overall_severity"] = "green"

    return final_output
