import cv2
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
# 2. Init Models (YOLO + Pose meta)
# ================================
yolo_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose


# ================================
# 3. Pose → Anchor Extraction
# ================================
def compute_posture_anchors(pose_landmarks, frame_shape):
    """
    Compute human anchors needed by workstation rules.
    NOTE: Everything must return a NUMBER (no dicts).
    """
    h, w, _ = frame_shape

    def px(lm):
        return lm.x * w, lm.y * h, lm.z

    def get_lm(idx):
        return pose_landmarks[idx]

    leye = get_lm(mp_pose.PoseLandmark.LEFT_EYE)
    reye = get_lm(mp_pose.PoseLandmark.RIGHT_EYE)
    relbow = get_lm(mp_pose.PoseLandmark.RIGHT_ELBOW)
    rhip = get_lm(mp_pose.PoseLandmark.RIGHT_HIP)
    lhip = get_lm(mp_pose.PoseLandmark.LEFT_HIP)
    rknee = get_lm(mp_pose.PoseLandmark.RIGHT_KNEE)

    # Convert to px
    leye_x, leye_y, _ = px(leye)
    reye_x, reye_y, _ = px(reye)
    relbow_x, relbow_y, _ = px(relbow)
    rhip_x, rhip_y, _ = px(rhip)
    lhip_x, lhip_y, _ = px(lhip)
    rknee_x, rknee_y, _ = px(rknee)

    # Eye center (Y + X anchor)
    eye_y = (leye_y + reye_y) / 2.0
    eye_x = (leye_x + reye_x) / 2.0

    # Thigh midpoint Y
    thigh_y = (rhip_y + rknee_y) / 2.0

    # Body center X (used to select the correct workstation items)
    body_center_x = (lhip_x + rhip_x) / 2.0
    body_center_y = (lhip_y + rhip_y) / 2.0

    return {
        "Eye_Keypoint_Y": eye_y,
        "Eye_Keypoint_X_Z": eye_x,       # FIXED: single number only
        "Elbow_Keypoint_Y": relbow_y,
        "Thigh_Keypoint_Y": thigh_y,
        "Knee_Back_Keypoint_X": rknee_x,
        "Body_Center_X": body_center_x,
        "Body_Center_Y": body_center_y,
    }


# ================================
# 4. YOLO Object Detection (raw)
# ================================
def detect_workstation_objects_raw(frame):
    """
    Safe YOLO inference function:
    - Checks for empty frame
    - Resizes frame to safe size (640x480)
    - Scales boxes back to original size
    - Prevents PyTorch convolution errors
    """

    # 1️⃣ Safety check to avoid YOLO crash
    if frame is None or frame.size == 0:
        return {"monitor": [], "worksurface": [], "chair": []}

    original_h, original_w = frame.shape[:2]

    # 2️⃣ Resize for YOLO stability
    resized = cv2.resize(frame, (640, 480))

    try:
        results = yolo_model(resized, conf=0.5, verbose=False)[0]
    except Exception as e:
        print("YOLO failed:", e)
        return {"monitor": [], "worksurface": [], "chair": []}

    components_raw = {"monitor": [], "worksurface": [], "chair": []}

    # 3️⃣ Scaling factors back to original frame size
    scale_x = original_w / 640
    scale_y = original_h / 480

    # 4️⃣ Extract YOLO boxes
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0]

        # Scale to original frame
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 220, 0), 2)
        cv2.putText(frame, cls_name, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

        # Map object classes
        if cls_name in ["tv", "laptop", "monitor"]:
            components_raw["monitor"].append((x1, y1, x2, y2))

        if cls_name in ["table", "desk", "dining table"]:
            components_raw["worksurface"].append((x1, y1, x2, y2))

        if cls_name == "chair":
            components_raw["chair"].append((x1, y1, x2, y2))

    return components_raw



# ================================
# 5. One-person workstation filtering
# ================================
def filter_workstation_for_person(components_raw, anchors, frame_shape):
    """
    Select nearest workstation items (monitor/desk/chair)
    to the user based on body center.
    """
    h, w, _ = frame_shape
    cx_person = anchors["Body_Center_X"]
    cy_person = anchors["Body_Center_Y"]

    components = {}

    for comp_name, box_list in components_raw.items():
        if not box_list:
            continue

        best_box = None
        best_score = float("inf")

        for (x1, y1, x2, y2) in box_list:
            obj_cx = (x1 + x2) / 2
            obj_cy = (y1 + y2) / 2

            score = abs(obj_cx - cx_person) + abs(obj_cy - cy_person) * 0.4

            if score < best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)

        # Threshold: ignore objects too far from user
        if best_box and best_score < w * 0.6:
            components[comp_name] = best_box

    return components


# ================================
# 6. Anchor Extraction
# ================================
def extract_object_anchor(anchor_name, box):
    if box is None:
        return None

    x1, y1, x2, y2 = box

    if anchor_name == "Monitor_Top_Y":
        return y1

    if anchor_name == "Monitor_Center_X_Z":
        return (x1 + x2) / 2  # FIX: return numeric X only

    if anchor_name == "Worksurface_Top_Y":
        return y1

    if anchor_name == "Worksurface_Bottom_Y":
        return y2

    if anchor_name == "Seat_Edge_X":
        return x2

    return None


def extract_human_anchor(anchor_name, anchors):
    return anchors.get(anchor_name)


# ================================
# 7. Generic Rule Evaluation
# ================================
def evaluate_rule_generic(rule, human_val, obj_val):
    report = {
        "severity": "green",
        "delta": None,
        "iso_principle": rule["iso_principle"]
    }

    if human_val is None or obj_val is None:
        report["severity"] = "unknown"
        return report

    # MUST BE NUMBERS HERE
    delta = float(obj_val - human_val)
    report["delta"] = delta

    # Tolerance rule
    if "delta_max_tolerance" in rule:
        tol = rule["delta_max_tolerance"]
        if abs(delta) > tol:
            report["severity"] = "yellow"

    # Severity map
    if "severity_map" in rule:
        ctx = {
            "monitor_top_y": obj_val,
            "eye_keypoint_y": human_val,
            "abs": abs
        }
        for sev, expr in rule["severity_map"].items():
            try:
                if eval(expr, {"__builtins__": {"abs": abs}}, ctx):
                    report["severity"] = sev.lower()
            except:
                pass

    # Clearance rules (pixel approximation)
    if "min_clearance_cm" in rule:
        if delta < rule["min_clearance_cm"]:
            report["severity"] = "yellow"

    if "min_gap_cm" in rule:
        if delta < rule["min_gap_cm"]:
            report["severity"] = "yellow"

    return report


# ================================
# 8. Main ISO Workstation Evaluator
# ================================
def evaluate_workstation_iso(components, anchors, frame_shape):
    report = {}

    for comp_key, comp_cfg in WS_RULES.items():
        if comp_key not in components:
            continue

        comp_box = components[comp_key]
        comp_results = {}

        for rule in comp_cfg["requirements"]:
            human_anchor_name = rule["human_anchor"]
            object_anchor_name = rule["object_anchor"]

            human_val = extract_human_anchor(human_anchor_name, anchors)
            obj_val = extract_object_anchor(object_anchor_name, comp_box)

            comp_results[rule["check_id"]] = evaluate_rule_generic(
                rule, human_val, obj_val
            )

        report[comp_key] = comp_results

    return report
