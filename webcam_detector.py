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


def get_posture_report(lm, w, h, mp_pose):
    """Compute posture angles + ISO classification"""
    def px(p): return [p.x * w, p.y * h]

    ear = px(lm[mp_pose.PoseLandmark.RIGHT_EAR])
    shoulder_r = px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    shoulder_l = px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
    elbow = px(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
    wrist = px(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
    hip_r = px(lm[mp_pose.PoseLandmark.RIGHT_HIP])
    hip_l = px(lm[mp_pose.PoseLandmark.LEFT_HIP])

    # Neck flexion
    vertical_point = [ear[0], ear[1] - 100]
    neck_angle = calculate_angle(shoulder_r, ear, vertical_point)
    sev_neck, _ = classify_iso("neck_flexion", neck_angle)

    # Shoulder elevation
    shoulder_diff = abs(shoulder_r[1] - shoulder_l[1])
    sev_shoulder, _ = classify_iso("shoulder_elevation", shoulder_diff)

    # Elbow angle
    elbow_angle = calculate_angle(shoulder_r, elbow, wrist)
    sev_elbow, _ = classify_iso("elbow_angle", elbow_angle)

    # Wrist deviation
    wrist_dev = calculate_angle(elbow, wrist, [wrist[0] + 100, wrist[1]])
    sev_wrist, _ = classify_iso("wrist_deviation", wrist_dev)

    # Pelvic tilt
    pelvis_tilt = hip_r[1] - hip_l[1]
    sev_pelvis, _ = classify_iso("pelvic_tilt", pelvis_tilt)

    return {
        "neck_flexion": {"angle": neck_angle, "severity": sev_neck},
        "shoulder_elevation": {"angle": shoulder_diff, "severity": sev_shoulder},
        "elbow_angle": {"angle": elbow_angle, "severity": sev_elbow},
        "wrist_deviation": {"angle": wrist_dev, "severity": sev_wrist},
        "pelvic_tilt": {"angle": pelvis_tilt, "severity": sev_pelvis},
    }
