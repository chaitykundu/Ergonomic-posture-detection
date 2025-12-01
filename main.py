import cv2
import mediapipe as mp

# Phase-1 posture module
from webcam_detector import get_posture_report

# Phase-2 workstation module
from postura_workstation import (
    detect_workstation_objects,
    compute_posture_anchors,
    evaluate_workstation_iso
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def run_postura_main():
    cap = cv2.VideoCapture(0)

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

            # 1️⃣ PHASE 1 — POSTURE
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape

                # Compute posture angles
                posture_report = get_posture_report(lm, w, h, mp_pose)

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # 2️⃣ PHASE 2 — OBJECT DETECTION
            components = detect_workstation_objects(frame)

            # 3️⃣ ISO Workstation Evaluation
            if results.pose_landmarks:
                anchors = compute_posture_anchors(results.pose_landmarks.landmark, frame.shape)
                workstation_report = evaluate_workstation_iso(components, anchors, frame.shape)

            # 4️⃣ Display Combined Results
            y = 30
            cv2.putText(frame, "[Posture]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            y += 25
            for k, v in posture_report.items():
                color = (0,255,0) if v["severity"]=="green" else (0,255,255) if v["severity"]=="yellow" else (0,0,255)
                cv2.putText(frame, f"{k}: {v['severity']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y += 20

            y += 15
            cv2.putText(frame, "[Workstation]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            y += 25
            for comp, rules in workstation_report.items():
                cv2.putText(frame, f"- {comp}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                y += 20
                for rule, info in rules.items():
                    sev = info["severity"]
                    color = (0,255,0) if sev=="green" else (0,255,255) if sev=="yellow" else (0,0,255)
                    cv2.putText(frame, f"{rule}: {sev}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y += 18

            cv2.imshow("POSTURA – Full ISO System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_postura_main()
