import cv2
import mediapipe as mp
import json

# Phase 1 – posture ISO engine
from webcam_detector import get_posture_report

# Phase 2 – workstation ISO engine
from postura_workstation import (
    compute_posture_anchors,
    detect_workstation_objects_raw,
    filter_workstation_for_person,
    evaluate_workstation_iso,
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# =====================================
# ⭐ Phase 3 – Unified ISO JSON Builder
# =====================================

def merge_iso_reports(posture_report, workstation_report):
    """
    Create a single unified ISO report used by AI, PDF, dashboard, etc.
    """

    final_output = {
        "posture": {},
        "workstation": {},
        "overall_severity": "green"
    }

    # Add posture metrics
    for key, data in posture_report.items():
        final_output["posture"][key] = {
            "angle": data["angle"],
            "severity": data["severity"]
        }

    # Add workstation metrics
    for comp, rules in workstation_report.items():
        final_output["workstation"][comp] = {}
        for rule_id, rule_data in rules.items():
            final_output["workstation"][comp][rule_id] = {
                "severity": rule_data["severity"],
                "delta": rule_data["delta"],
                "iso_clause": rule_data["iso_principle"]
            }

    # Calculate overall severity
    all_severities = []

    for key, data in posture_report.items():
        all_severities.append(data["severity"])

    for comp, rules in workstation_report.items():
        for rule_id, rule_data in rules.items():
            all_severities.append(rule_data["severity"])

    if "red" in all_severities:
        final_output["overall_severity"] = "red"
    elif "yellow" in all_severities:
        final_output["overall_severity"] = "yellow"
    else:
        final_output["overall_severity"] = "green"

    return final_output



def run_postura_iso():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            posture_report = {}
            workstation_report = {}

            # --------------------------------
            # 1️⃣ Pose / Posture (Phase 1)
            # --------------------------------
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape

                # posture metrics
                posture_report = get_posture_report(lm, w, h)

                # draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # posture anchors for workstation rules
                anchors = compute_posture_anchors(lm, frame.shape)
            else:
                anchors = None

            # --------------------------------
            # 2️⃣ Object detection (Phase 2)
            # --------------------------------
            components_raw = detect_workstation_objects_raw(frame)

            # --------------------------------
            # 3️⃣ Single-person workstation association
            # --------------------------------
            if anchors is not None:
                components = filter_workstation_for_person(
                    components_raw, anchors, frame.shape
                )

                workstation_report = evaluate_workstation_iso(
                    components, anchors, frame.shape
                )

            # --------------------------------
            # ⭐ 4️⃣ Phase 3 – Unified ISO Output
            # --------------------------------
            unified_iso_output = merge_iso_reports(posture_report, workstation_report)

            # Print unified JSON in terminal (LLM-ready)
            print("\n=== Unified ISO Output ===")
            print(json.dumps(unified_iso_output, indent=4))


            # --------------------------------
            # 5️⃣ Draw overlay
            # --------------------------------
            y = 25
            cv2.putText(frame, "[Posture]", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 25

            for key, val in posture_report.items():
                sev = val["severity"]
                color = (0, 255, 0) if sev == "green" else (0, 255, 255) if sev == "yellow" else (0, 0, 255)
                cv2.putText(frame, f"{key}: {sev}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y += 18

            y += 15
            cv2.putText(frame, "[Workstation]", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 25

            for comp, rules in workstation_report.items():
                cv2.putText(frame, f"- {comp}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y += 20
                for rule_id, res in rules.items():
                    sev = res["severity"]
                    color = (0, 255, 0) if sev == "green" else (0, 255, 255) if sev == "yellow" else (0, 0, 255)
                    cv2.putText(frame, f"{rule_id}: {sev}",
                                (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y += 18

            cv2.imshow("POSTURA – ISO Posture + Workstation (Unified Output)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    run_postura_iso()
