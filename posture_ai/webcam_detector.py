import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

# Load ISO posture config
BASE_DIR = Path(__file__).resolve().parent.parent
iso_config = json.load(open(BASE_DIR / "files" / "iso_posture_config.json", "r"))

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


def angle_from_vertical(vec):
    vertical_up = np.array([0.0, -1.0])
    n = np.linalg.norm(vec)
    if n == 0:
        return 0.0
    cos = np.dot(vec, vertical_up) / n
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def get_posture_report(landmarks, w: int, h: int):
    def px(lm):
        return np.array([lm.x * w, lm.y * h], dtype=float)

    # ---- Key landmarks ----
    ear_r = px(landmarks[mp_pose.PoseLandmark.RIGHT_EAR])
    ear_l = px(landmarks[mp_pose.PoseLandmark.LEFT_EAR])
    shoulder_r = px(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    shoulder_l = px(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
    elbow = px(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
    wrist = px(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
    index_finger = px(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX])
    hip_r = px(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
    hip_l = px(landmarks[mp_pose.PoseLandmark.LEFT_HIP])

    # ---- Midpoints ----
    ear_mid = (ear_r + ear_l) / 2
    shoulder_mid = (shoulder_r + shoulder_l) / 2
    hip_mid = (hip_r + hip_l) / 2

    # ======================
    # 1. Neck flexion
    # ======================
    neck_vec = ear_mid - shoulder_mid
    neck_angle = np.clip(angle_from_vertical(neck_vec), 0, 90)
    sev_neck, _ = classify_iso("neck_flexion", neck_angle)

    # ======================
    # 2. Shoulder elevation (shrugging)
    # ======================
    torso_len = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_len == 0:
        shoulder_elevation_deg = 0.0
    else:
        elev_px = (
            abs(shoulder_r[1] - shoulder_mid[1]) +
            abs(shoulder_l[1] - shoulder_mid[1])
        ) / 2
        elev_ratio = elev_px / torso_len
        shoulder_elevation_deg = np.degrees(np.arctan(elev_ratio))

    shoulder_elevation_deg = np.clip(shoulder_elevation_deg, 0, 45)
    sev_shoulder, _ = classify_iso("shoulder_elevation", shoulder_elevation_deg)

    # ======================
    # 3. Elbow angle
    # ======================
    elbow_angle = calculate_angle(shoulder_mid, elbow, wrist)
    elbow_angle = np.clip(elbow_angle, 0, 180)
    sev_elbow, _ = classify_iso("elbow_angle", elbow_angle)

    # ======================
    # 4. Wrist deviation
    # ======================
    forearm_vec = wrist - elbow
    hand_vec = index_finger - wrist
    nf = np.linalg.norm(forearm_vec)
    nh = np.linalg.norm(hand_vec)

    if nf == 0 or nh == 0:
        wrist_dev = 0.0
    else:
        cos = np.dot(forearm_vec, hand_vec) / (nf * nh)
        wrist_dev = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    wrist_dev = np.clip(wrist_dev, 0, 90)
    sev_wrist, _ = classify_iso("wrist_deviation", wrist_dev)

    # ======================
    # 5. Trunk inclination (ISO pelvic proxy)
    # ======================
    trunk_vec = shoulder_mid - hip_mid
    trunk_angle = np.clip(angle_from_vertical(trunk_vec), 0, 90)
    sev_pelvis, _ = classify_iso("pelvic_tilt", trunk_angle)

    return {
        "neck_flexion": {"angle": neck_angle, "severity": sev_neck},
        "shoulder_elevation": {"angle": shoulder_elevation_deg, "severity": sev_shoulder},
        "elbow_angle": {"angle": elbow_angle, "severity": sev_elbow},
        "wrist_deviation": {"angle": wrist_dev, "severity": sev_wrist},
        "pelvic_tilt": {"angle": trunk_angle, "severity": sev_pelvis},
    }
