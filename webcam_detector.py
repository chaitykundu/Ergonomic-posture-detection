import cv2
import mediapipe as mp
import numpy as np
import json

# Load ISO Config
iso_config = json.load(open("iso_posture_config.json"))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def classify_iso(angle_name, value):
    cfg = iso_config["posture_metrics"][angle_name]
    ideal = cfg["ideal"]
    delta = abs(value - ideal)

    if delta <= iso_config["compliance_levels"]["green"]["max_deviation_deg"]:
        severity = "green"
    elif delta <= iso_config["compliance_levels"]["yellow"]["max_deviation_deg"]:
        severity = "yellow"
    else:
        severity = "red"

    return severity, delta

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Key landmarks
            ear = [lm[mp_pose.PoseLandmark.RIGHT_EAR].x, lm[mp_pose.PoseLandmark.RIGHT_EAR].y]
            shoulder_r = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            shoulder_l = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            hip_r = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
            hip_l = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]

            # -------------------------
            # 1. NECK FLEXION
            # -------------------------
            vertical_point = [ear[0], ear[1] - 0.1]
            neck_flexion = calculate_angle(shoulder_r, ear, vertical_point)
            severity_neck, delta_neck = classify_iso("neck_flexion", neck_flexion)

            cv2.putText(frame, f"Neck: {int(neck_flexion)}° {severity_neck}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if severity_neck=="green" else (0,255,255) if severity_neck=="yellow" else (0,0,255), 2)

            # -------------------------
            # 2. SHOULDER ELEVATION
            # -------------------------
            shoulder_height_diff = abs(shoulder_r[1] - shoulder_l[1]) * 100  # normalized
            severity_shoulder, delta_shoulder = classify_iso("shoulder_elevation", shoulder_height_diff)

            cv2.putText(frame, f"Shoulder Elev: {int(shoulder_height_diff)} {severity_shoulder}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if severity_shoulder=="green" else (0,255,255) if severity_shoulder=="yellow" else (0,0,255), 2)

            # -------------------------
            # 3. ELBOW ANGLE
            # -------------------------
            elbow_angle = calculate_angle(shoulder_r, elbow, wrist)
            severity_elbow, delta_elbow = classify_iso("elbow_angle", elbow_angle)

            cv2.putText(frame, f"Elbow: {int(elbow_angle)}° {severity_elbow}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if severity_elbow=="green" else (0,255,255) if severity_elbow=="yellow" else (0,0,255), 2)

            # -------------------------
            # 4. WRIST DEVIATION
            # -------------------------
            wrist_deviation = calculate_angle(elbow, wrist, [wrist[0] + 0.1, wrist[1]])
            severity_wrist, delta_wrist = classify_iso("wrist_deviation", wrist_deviation)

            cv2.putText(frame, f"Wrist: {int(wrist_deviation)}° {severity_wrist}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if severity_wrist=="green" else (0,255,255) if severity_wrist=="yellow" else (0,0,255), 2)

            # -------------------------
            # 5. PELVIC TILT
            # -------------------------
            pelvic_tilt = (hip_r[1] - hip_l[1]) * 100
            severity_pelvis, delta_pelvis = classify_iso("pelvic_tilt", pelvic_tilt)

            cv2.putText(frame, f"Pelvic Tilt: {int(pelvic_tilt)} {severity_pelvis}",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if severity_pelvis=="green" else (0,255,255) if severity_pelvis=="yellow" else (0,0,255), 2)

        # Draw pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("POSTURA ISO Real-Time Detector", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
