import cv2
import mediapipe as mp
import numpy as np
import json

# Load ISO posture config
iso_config = json.load(open("iso_posture_config.json", "r"))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """
    Generic joint angle between three 2D points.
    a, b, c are [x, y].
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def classify_iso(angle_name: str, value: float):
    """
    Classify a posture metric against ISO posture config.
    angle_name must exist in iso_posture_config.json under posture_metrics.
    """
    cfg = iso_config["posture_metrics"][angle_name]
    ideal = cfg["ideal"]
    delta = abs(value - ideal)

    g = iso_config["compliance_levels"]["green"]["max_deviation_deg"]
    y = iso_config["compliance_levels"]["yellow"]["max_deviation_deg"]

    if delta <= g:
        severity = "green"
    elif delta <= y:
        severity = "yellow"
    else:
        severity = "red"

    return severity, delta


def get_posture_report(landmarks, w: int, h: int):
    """
    Compute key posture metrics + ISO severity.
    Input: landmarks from MediaPipe, frame width and height.
    Output: dict of posture metrics.
    """
    def px(lm):
        return [lm.x * w, lm.y * h]

    ear = px(landmarks[mp_pose.PoseLandmark.RIGHT_EAR])
    shoulder_r = px(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    shoulder_l = px(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
    elbow = px(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
    wrist = px(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
    hip_r = px(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
    hip_l = px(landmarks[mp_pose.PoseLandmark.LEFT_HIP])

    # 1. Neck flexion
    vertical_point = [ear[0], ear[1] - 100]
    neck_angle = calculate_angle(shoulder_r, ear, vertical_point)
    sev_neck, _ = classify_iso("neck_flexion", neck_angle)

    # 2. Shoulder elevation (difference in shoulder heights)
    shoulder_diff = abs(shoulder_r[1] - shoulder_l[1])
    sev_shoulder, _ = classify_iso("shoulder_elevation", shoulder_diff)

    # 3. Elbow angle
    elbow_angle = calculate_angle(shoulder_r, elbow, wrist)
    sev_elbow, _ = classify_iso("elbow_angle", elbow_angle)

    # 4. Wrist deviation
    wrist_ref = [wrist[0] + 100, wrist[1]]
    wrist_dev = calculate_angle(elbow, wrist, wrist_ref)
    sev_wrist, _ = classify_iso("wrist_deviation", wrist_dev)

    # 5. Pelvic tilt (difference in hip heights)
    pelvis_tilt = hip_r[1] - hip_l[1]
    sev_pelvis, _ = classify_iso("pelvic_tilt", pelvis_tilt)

    return {
        "neck_flexion": {"angle": neck_angle, "severity": sev_neck},
        "shoulder_elevation": {"angle": shoulder_diff, "severity": sev_shoulder},
        "elbow_angle": {"angle": elbow_angle, "severity": sev_elbow},
        "wrist_deviation": {"angle": wrist_dev, "severity": sev_wrist},
        "pelvic_tilt": {"angle": pelvis_tilt, "severity": sev_pelvis},
    }
