import cv2
import numpy as np
import mediapipe as mp
import json
from ultralytics import YOLO
import math
from pathlib import Path

# ================================
# 1. Load ISO Workstation Config
# ================================
ISO_WORKSTATION_CONFIG_PATH = "iso_workstation_config.json"

if not Path(ISO_WORKSTATION_CONFIG_PATH).exists():
    raise FileNotFoundError(f"{ISO_WORKSTATION_CONFIG_PATH} not found in current directory.")

with open(ISO_WORKSTATION_CONFIG_PATH, "r") as f:
    WS_CFG = json.load(f)

WS_RULES = WS_CFG["workstation_component_rules"]

# ================================
# 2. Init Models (YOLO + Pose)
# ================================
# YOLOv8 small model for speed
yolo_model = YOLO("yolov8n.pt")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ================================
# 3. Pose → Anchor Extraction
# ================================
def compute_posture_anchors(pose_landmarks, frame_shape):
    """
    Compute human anchors required by iso_workstation_config.json
    (Eye_Keypoint_Y, Elbow_Keypoint_Y, Thigh_Keypoint_Y, Knee_Back_Keypoint_X, etc.)
    """
    h, w, _ = frame_shape

    def to_px(lm):
        return lm.x * w, lm.y * h, lm.z

    # Safely fetch landmarks (fallback if missing)
    def get_lm(index):
        return pose_landmarks[index]

    # Landmarks
    nose = get_lm(mp_pose.PoseLandmark.NOSE)
    leye = get_lm(mp_pose.PoseLandmark.LEFT_EYE)
    reye = get_lm(mp_pose.PoseLandmark.RIGHT_EYE)
    relbow = get_lm(mp_pose.PoseLandmark.RIGHT_ELBOW)
    rshoulder = get_lm(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    rhip = get_lm(mp_pose.PoseLandmark.RIGHT_HIP)
    rknee = get_lm(mp_pose.PoseLandmark.RIGHT_KNEE)

    nose_x, nose_y, nose_z = to_px(nose)
    leye_x, leye_y, leye_z = to_px(leye)
    reye_x, reye_y, reye_z = to_px(reye)
    relbow_x, relbow_y, relbow_z = to_px(relbow)
    rshoulder_x, rshoulder_y, rshoulder_z = to_px(rshoulder)
    rhip_x, rhip_y, rhip_z = to_px(rhip)
    rknee_x, rknee_y, rknee_z = to_px(rknee)

    # Eye Y = average of left & right eyes
    eye_y = (leye_y + reye_y) / 2.0
    eye_x = (leye_x + reye_x) / 2.0
    eye_z = (leye_z + reye_z) / 2.0

    # Thigh Y: midpoint between hip and knee (approx top of thigh)
    thigh_y = (rhip_y + rknee_y) / 2.0

    anchors = {
        "Eye_Keypoint_Y": eye_y,
        "Eye_Keypoint_X_Z": {"x": eye_x, "z": eye_z},  # for future depth work
        "Elbow_Keypoint_Y": relbow_y,
        "Thigh_Keypoint_Y": thigh_y,
        "Knee_Back_Keypoint_X": rknee_x,
        # Extra context if needed later
        "Shoulder_Keypoint_Y": rshoulder_y,
    }

    return anchors


# ================================
# 4. YOLO Object Detection
# ================================
def detect_workstation_objects(frame):
    """
    Run YOLO on frame, map YOLO classes to your workstation components:
      - monitor  -> tv / laptop / (monitor if custom trained)
      - worksurface -> table / dining table
      - chair   -> chair
    Returns: dict: component_key -> (x1, y1, x2, y2)
    """
    h, w, _ = frame.shape
    results = yolo_model(frame, conf=0.5, verbose=False)[0]

    components = {}

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = results.names[cls_id]  # e.g. 'tv', 'laptop', 'chair', 'dining table'
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Draw box + label on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Map YOLO class → workstation component keys in JSON
        for comp_key, comp_cfg in WS_RULES.items():
            target_class = comp_cfg["detection_class"].lower()

            # MONITOR mapping
            if target_class == "monitor":
                if cls_name in ["tv", "laptop", "monitor"]:
                    components["monitor"] = (x1, y1, x2, y2)

            # WORKSURFACE mapping
            elif target_class == "desk":
                if cls_name in ["dining table", "table", "desk"]:
                    components["worksurface"] = (x1, y1, x2, y2)

            # CHAIR mapping
            elif target_class == "chair":
                if cls_name == "chair":
                    components["chair"] = (x1, y1, x2, y2)

    return components, frame


# ================================
# 5. Anchor Extraction for Objects
# ================================
def extract_object_anchor(anchor_name, box):
    """
    From YOLO bounding box (x1, y1, x2, y2), extract the required anchor value.
    """
    if box is None:
        return None

    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    if anchor_name == "Monitor_Top_Y":
        return y1
    if anchor_name == "Monitor_Center_X_Z":
        # For now just store 2D center; Z estimation can be added later
        return {"x": center_x, "z": 0.0}
    if anchor_name == "Worksurface_Top_Y":
        return y1
    if anchor_name == "Worksurface_Bottom_Y":
        return y2
    if anchor_name == "Seat_Edge_X":
        # Assume seat edge is the front-most x (here we just use x2)
        return x2

    return None


def extract_human_anchor(anchor_name, posture_anchors):
    """
    Look up human anchor from computed posture_anchors.
    """
    if anchor_name in posture_anchors:
        return posture_anchors[anchor_name]
    return None


# ================================
# 6. Rule Evaluation Helpers
# ================================
def evaluate_viewing_distance(rule, monitor_box, frame_shape):
    """
    Approximate viewing distance using bounding box size ratio.
    This is a placeholder. For real cm, add camera calibration / depth.
    """
    if monitor_box is None:
        return None

    h, w, _ = frame_shape
    x1, y1, x2, y2 = monitor_box
    bw = x2 - x1
    bh = y2 - y1
    diag = math.sqrt(bw ** 2 + bh ** 2)
    frame_diag = math.sqrt(w ** 2 + h ** 2)
    ratio = diag / frame_diag  # 0–1

    # Simple heuristic: larger on screen → closer, smaller → farther
    # You MUST replace this with calibrated logic for real cm.
    # Example: distance_cm ~ k / ratio
    if ratio == 0:
        approx_cm = 999
    else:
        k = 120  # arbitrary scaling constant for demo
        approx_cm = k / ratio

    min_cm = rule.get("min_cm", 50)
    max_cm = rule.get("max_cm", 100)

    if min_cm <= approx_cm <= max_cm:
        severity = "green"
    elif approx_cm < min_cm * 0.8 or approx_cm > max_cm * 1.2:
        severity = "red"
    else:
        severity = "yellow"

    return {
        "distance_cm_est": approx_cm,
        "severity": severity,
        "iso_principle": rule["iso_principle"]
    }


def evaluate_rule_generic(rule, human_anchor_val, object_anchor_val):
    """
    Generic evaluation for rules with:
      - delta_max_tolerance
      - severity_map
      - min_clearance_cm
      - min_gap_cm
    """
    report = {
        "severity": "green",
        "delta": None,
        "iso_principle": rule["iso_principle"],
    }

    if human_anchor_val is None or object_anchor_val is None:
        report["severity"] = "unknown"
        return report

    delta = object_anchor_val - human_anchor_val
    report["delta"] = float(delta)

    # 1) delta_max_tolerance → yellow if outside
    if "delta_max_tolerance" in rule:
        tol = rule["delta_max_tolerance"]
        if abs(delta) > tol:
            report["severity"] = "yellow"

    # 2) severity_map → evaluate conditions (e.g. "monitor_top_y > eye_keypoint_y + 50")
    if "severity_map" in rule:
        # Build variable context
        ctx = {
            "monitor_top_y": object_anchor_val,
            "eye_keypoint_y": human_anchor_val,
            "worksurface_top_y": object_anchor_val,
            "elbow_keypoint_y": human_anchor_val,
            "worksurface_bottom_y": object_anchor_val,
            "thigh_keypoint_y": human_anchor_val,
            "abs": abs
        }
        for sev_label, expr in rule["severity_map"].items():
            try:
                # Safe-ish eval with restricted builtins
                if eval(expr, {"__builtins__": {"abs": abs}}, ctx):
                    report["severity"] = sev_label.lower()
            except Exception:
                # Ignore malformed expressions
                pass

    # 3) Clearance / gap rules (still using px as proxy; cm needs calibration)
    if "min_clearance_cm" in rule:
        # Here delta = object_bottom_y - thigh_y (positive = space)
        # For now we compare absolute delta in pixels as a stand-in for cm.
        min_clearance = rule["min_clearance_cm"]
        if delta < min_clearance:
            report["severity"] = "yellow"

    if "min_gap_cm" in rule:
        # Seat depth gap (seat_edge_x - knee_back_x); positive gap needed
        min_gap = rule["min_gap_cm"]
        if delta < min_gap:
            report["severity"] = "yellow"

    return report


def evaluate_workstation_iso(components, posture_anchors, frame_shape):
    """
    Main ISO comparator:
      - loops over workstation_component_rules
      - applies each requirement
      - returns report dict
    """
    report = {}

    for comp_key, comp_cfg in WS_RULES.items():
        comp_box = components.get(comp_key)
        comp_req = comp_cfg["requirements"]
        comp_result = {}

        for rule in comp_req:
            check_id = rule["check_id"]

            # Special handling for ViewingDistance
            if check_id == "ViewingDistance":
                if comp_key == "monitor":
                    vd_report = evaluate_viewing_distance(rule, comp_box, frame_shape)
                    if vd_report is not None:
                        comp_result[check_id] = vd_report
                continue

            # Generic rules
            human_anchor_name = rule["human_anchor"]
            object_anchor_name = rule["object_anchor"]

            human_val = extract_human_anchor(human_anchor_name, posture_anchors)
            obj_val = extract_object_anchor(object_anchor_name, comp_box)

            rule_report = evaluate_rule_generic(rule, human_val, obj_val)
            comp_result[check_id] = rule_report

        if comp_result:
            report[comp_key] = comp_result

    return report


# ================================
# 7. Main Real-time Loop
# ================================
def run_realtime():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)

            # --- Object detection (YOLO) ---
            components, frame = detect_workstation_objects(frame)

            # --- Posture anchors (from Pose) ---
            workstation_report = {}
            if pose_results.pose_landmarks:
                anchors = compute_posture_anchors(pose_results.pose_landmarks.landmark, frame.shape)

                # Draw pose skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # ISO workstation evaluation
                workstation_report = evaluate_workstation_iso(components, anchors, frame.shape)

                # Overlay some anchor reference (eye Y)
                eye_y = anchors["Eye_Keypoint_Y"]
                cv2.circle(frame, (50, int(eye_y)), 5, (0, 255, 0), -1)
                cv2.putText(frame, "Eye Y", (55, int(eye_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # --- Draw report on screen ---
            y0 = 30
            for comp_key, comp_data in workstation_report.items():
                cv2.putText(frame, f"[{comp_key}]", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                y0 += 22
                for check_id, res in comp_data.items():
                    sev = res["severity"]
                    color = (0, 255, 0) if sev == "green" else (0, 255, 255) if sev == "yellow" else (0, 0, 255)
                    txt = f" - {check_id}: {sev}"
                    cv2.putText(frame, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y0 += 18

            cv2.imshow("POSTURA – Phase 2 (Workstation ISO)", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
