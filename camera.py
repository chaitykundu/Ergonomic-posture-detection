import cv2
import mediapipe as mp
import json

# Phase 1 – posture ISO engine
from posture_ai.webcam_detector import get_posture_report

# Phase 2 – workstation ISO engine
from posture_ai.postura_workstation import (
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
    Merge posture + workstation results into one unified JSON.
    This is used for LLM correction engine (Phase 4) and reporting.
    """

    final_output = {
        "posture": {},
        "workstation": {},
        "overall_severity": "green"
    }

    # -------------------------------
    # POSTURE METRICS
    # -------------------------------
    for key, data in posture_report.items():
        final_output["posture"][key] = {
            "angle": data["angle"],
            "severity": data["severity"]
        }

    # -------------------------------
    # WORKSTATION METRICS
    # -------------------------------
    for comp, rules in workstation_report.items():
        final_output["workstation"][comp] = {}
        for rule_id, rule_data in rules.items():
            final_output["workstation"][comp][rule_id] = {
                "severity": rule_data["severity"],
                "delta": rule_data["delta"],
                "iso_clause": rule_data["iso_principle"]
            }

    # -------------------------------
    # OVERALL SEVERITY (red > yellow > green)
    # -------------------------------
    severities = []

    for data in posture_report.values():
        severities.append(data["severity"])

    for rules in workstation_report.values():
        for rule in rules.values():
            severities.append(rule["severity"])

    if "red" in severities:
        final_output["overall_severity"] = "red"
    elif "yellow" in severities:
        final_output["overall_severity"] = "yellow"
    else:
        final_output["overall_severity"] = "green"

    return final_output



# =====================================
# ⭐ MAIN REAL-TIME SYSTEM
# =====================================

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

            # Safety check
            if not ret or frame is None:
                continue

            # 🟢 Resize webcam frame for YOLO STABILITY
            frame = cv2.resize(frame, (960, 720))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            posture_report = {}
            workstation_report = {}

            # --------------------------------
            # 1️⃣ Phase-1: POSTURE DETECTION
            # --------------------------------
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape

                posture_report = get_posture_report(lm, w, h)

                # Draw human body
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Anchors used by workstation evaluation
                anchors = compute_posture_anchors(lm, frame.shape)
            else:
                anchors = None

            # --------------------------------
            # 2️⃣ Phase-2: OBJECT DETECTION
            # --------------------------------
            components_raw = detect_workstation_objects_raw(frame)

            # --------------------------------
            # 3️⃣ Phase-2: SINGLE PERSON FILTER
            # --------------------------------
            if anchors is not None:
                components = filter_workstation_for_person(components_raw, anchors, frame.shape)
                workstation_report = evaluate_workstation_iso(components, anchors, frame.shape)

            # --------------------------------
            # ⭐ 4️⃣ Phase-3: UNIFIED ISO OUTPUT
            # --------------------------------
            unified_iso_output = merge_iso_reports(posture_report, workstation_report)

            # Print nicely formatted result
            print("\n=== Unified ISO Output ===")
            print(json.dumps(unified_iso_output, indent=4))


            # --------------------------------
            # 5️⃣ DRAW OVERLAY ON FRAME
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

            cv2.imshow("POSTURA – Full ISO Engine (Posture + Workstation + Unified Output)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    run_postura_iso()
