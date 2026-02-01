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
    sev_neck, dev_neck = classify_iso("neck_flexion", neck_angle)

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
    sev_shoulder, dev_shoulder = classify_iso("shoulder_elevation", shoulder_elevation_deg)

    # ======================
    # 3. Elbow angle
    # ======================
    elbow_angle = calculate_angle(shoulder_mid, elbow, wrist)
    elbow_angle = np.clip(elbow_angle, 0, 180)
    sev_elbow, dev_elbow = classify_iso("elbow_angle", elbow_angle)

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
    sev_wrist, dev_wrist = classify_iso("wrist_deviation", wrist_dev)

    # ======================
    # 5. Trunk inclination (ISO pelvic proxy)
    # ======================
    trunk_vec = shoulder_mid - hip_mid
    trunk_angle = np.clip(angle_from_vertical(trunk_vec), 0, 90)
    sev_pelvis, dev_pelvis = classify_iso("pelvic_tilt", trunk_angle)

    # Build return dictionary with conditional deviation
    def build_metric(angle, severity, deviation, iso):
        result = {"angle": angle, "severity": severity, "iso": iso}
        if severity != "green":
            result["deviation"] = deviation
        return result
    
    return {
        "neck_flexion": build_metric(neck_angle, sev_neck, dev_neck, "ISO 9241-5:2024"),
        "shoulder_elevation": build_metric(shoulder_elevation_deg, sev_shoulder, dev_shoulder, "ISO 9241-5:2024"),
        "elbow_angle": build_metric(elbow_angle, sev_elbow, dev_elbow, "ISO 9241-5:2024"),
        "wrist_deviation": build_metric(wrist_dev, sev_wrist, dev_wrist, "ISO 9241-5:2024"),
        "pelvic_tilt": build_metric(trunk_angle, sev_pelvis, dev_pelvis, "ISO 9241-5:2024"),
    }


def draw_posture_status(frame, posture_data, x_offset=20, y_offset=60):
    """
    Draw posture status cards on frame with ISO compliance indicators.
    """
    # Severity colors (BGR format for OpenCV)
    severity_colors = {
        "green": (76, 175, 80),    # Green
        "yellow": (255, 193, 7),   # Amber/Yellow
        "red": (244, 67, 54)       # Red
    }
    
    # Metric display names and ISO references
    metric_info = {
        "neck_flexion": {"name": "Neck Angle", "iso": "ISO 9241-5:2024"},
        "shoulder_elevation": {"name": "Shoulder Height", "iso": "ISO 9241-5:2024"},
        "elbow_angle": {"name": "Elbow Angle", "iso": "ISO 9241-5:2024 §5.2.1"},
        "wrist_deviation": {"name": "Wrist Deviation", "iso": "ISO 9241-5:2024"},
        "pelvic_tilt": {"name": "Trunk Angle", "iso": "ISO 9241-5:2024"}
    }
    
    card_height = 85
    card_width = 280
    y_pos = y_offset
    
    for metric_key, data in posture_data.items():
        if metric_key not in metric_info:
            continue
            
        severity = data["severity"]
        angle = data["angle"]
        deviation = data.get("deviation", 0)
        
        metric_name = metric_info[metric_key]["name"]
        iso_ref = metric_info[metric_key]["iso"]
        
        # Draw card background with slight transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (x_offset, y_pos), 
                     (x_offset + card_width, y_pos + card_height),
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Draw border with severity color
        color = severity_colors[severity]
        cv2.rectangle(frame,
                     (x_offset, y_pos),
                     (x_offset + card_width, y_pos + card_height),
                     color, 2)
        
        # Draw severity indicator circle
        circle_x = x_offset + 25
        circle_y = y_pos + 25
        cv2.circle(frame, (circle_x, circle_y), 12, color, -1)
        
        # Add checkmark or warning symbol
        if severity == "green":
            # Checkmark
            cv2.line(frame, (circle_x - 5, circle_y), (circle_x - 2, circle_y + 5), (255, 255, 255), 2)
            cv2.line(frame, (circle_x - 2, circle_y + 5), (circle_x + 5, circle_y - 5), (255, 255, 255), 2)
        else:
            # Exclamation mark
            cv2.line(frame, (circle_x, circle_y - 5), (circle_x, circle_y + 2), (255, 255, 255), 2)
            cv2.circle(frame, (circle_x, circle_y + 6), 1, (255, 255, 255), -1)
        
        # Draw metric name
        cv2.putText(frame, metric_name,
                   (x_offset + 45, y_pos + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (33, 33, 33), 2)
        
        # Draw status text - ENHANCED for better visibility
        if severity == "green":
            status_text = "Optimal"
            status_color = severity_colors["green"]
        elif severity == "yellow":
            status_text = f"{deviation:.0f}° deviation"
            status_color = severity_colors["yellow"]
        else:
            status_text = f"{deviation:.0f}° deviation"
            status_color = severity_colors["red"]
        
        # Add white background behind text for contrast
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x_offset + card_width - text_size[0] - 15
        text_y = y_pos + 20
        
        # Draw white rectangle behind text
        cv2.rectangle(frame,
                     (text_x - 5, text_y - text_size[1] - 3),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (255, 255, 255), -1)
        
        # Draw the status text
        cv2.putText(frame, status_text,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Draw ISO reference (fixed visibility)
        cv2.putText(
            frame,
            iso_ref,
            (x_offset + 45, y_pos + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        
        # Draw angle value - ENHANCED for better visibility
        angle_text = f"Current: {angle:.1f}°"
        cv2.putText(frame, angle_text,
                   (x_offset + 45, y_pos + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
        
        y_pos += card_height + 10
    
    return frame


def draw_overall_status(frame, posture_data, x_offset=20, y_offset=20):
    """
    Draw overall posture compliance status banner.
    """
    # Count severities
    severity_counts = {"green": 0, "yellow": 0, "red": 0}
    for data in posture_data.values():
        severity_counts[data["severity"]] += 1
    
    # Determine overall status
    if severity_counts["red"] > 0:
        overall_status = "Posture Needs Adjustment"
        status_color = (244, 67, 54)  # Red
    elif severity_counts["yellow"] > 0:
        overall_status = "Posture Acceptable"
        status_color = (255, 193, 7)  # Yellow
    else:
        overall_status = "Optimal Posture"
        status_color = (76, 175, 80)  # Green
    
    # Draw status banner
    banner_height = 40
    banner_width = 280
    
    overlay = frame.copy()
    cv2.rectangle(overlay,
                 (x_offset, y_offset),
                 (x_offset + banner_width, y_offset + banner_height),
                 status_color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw status text
    cv2.putText(frame, overall_status,
               (x_offset + 10, y_offset + 27),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame