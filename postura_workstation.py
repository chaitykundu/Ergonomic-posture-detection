import cv2
import numpy as np
import mediapipe as mp
import json
from ultralytics import YOLO
import math
from pathlib import Path

# Load ISO Workstation Config
ISO_WORKSTATION_CONFIG_PATH = "iso_workstation_config.json"

if not Path(ISO_WORKSTATION_CONFIG_PATH).exists():
    raise FileNotFoundError(f"{ISO_WORKSTATION_CONFIG_PATH} missing!")

WS_CFG = json.load(open(ISO_WORKSTATION_CONFIG_PATH))
WS_RULES = WS_CFG["workstation_component_rules"]

# Load YOLO
yolo_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose


def compute_posture_anchors(landmarks, frame_shape):
    h, w, _ = frame_shape

    def px(l): return (l.x * w, l.y * h, l.z)

    le = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    re = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
    el = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

    eye_xa, eye_ya, eye_za = px(le)
    eye_xb, eye_yb, eye_zb = px(re)
    eye_y = (eye_ya + eye_yb) / 2

    elbow_y = px(el)[1]
    thigh_y = (px(hip)[1] + px(knee)[1]) / 2
    knee_x = px(knee)[0]

    return {
        "Eye_Keypoint_Y": eye_y,
        "Elbow_Keypoint_Y": elbow_y,
        "Thigh_Keypoint_Y": thigh_y,
        "Knee_Back_Keypoint_X": knee_x,
    }


def detect_workstation_objects(frame):
    results = yolo_model(frame, conf=0.5, verbose=False)[0]
    components = {}

    for box in results.boxes:
        cls_name = results.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Drawing
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 225, 0), 2)
        cv2.putText(frame, cls_name, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 225, 0), 1)

        # Map objects
        if cls_name in ["tv", "laptop", "monitor"]:
            components["monitor"] = (x1, y1, x2, y2)
        if cls_name in ["table", "desk", "dining table"]:
            components["worksurface"] = (x1, y1, x2, y2)
        if cls_name == "chair":
            components["chair"] = (x1, y1, x2, y2)

    return components


def extract_object_anchor(anchor_name, box):
    if box is None:
        return None

    x1, y1, x2, y2 = box
    if anchor_name == "Monitor_Top_Y":
        return y1
    if anchor_name == "Worksurface_Top_Y":
        return y1
    if anchor_name == "Worksurface_Bottom_Y":
        return y2
    if anchor_name == "Seat_Edge_X":
        return x2
    return None


def evaluate_rule(rule, human, obj):
    if human is None or obj is None:
        return {"severity": "unknown", "delta": None}

    delta = obj - human
    severity = "green"

    if "delta_max_tolerance" in rule:
        if abs(delta) > rule["delta_max_tolerance"]:
            severity = "yellow"

    if "severity_map" in rule:
        ctx = {
            "monitor_top_y": obj,
            "eye_keypoint_y": human,
        }
        for sev, expr in rule["severity_map"].items():
            if eval(expr, {"__builtins__": {"abs": abs}}, ctx):
                severity = sev.lower()

    return {"severity": severity, "delta": delta}


def evaluate_workstation_iso(components, anchors, frame_shape):
    report = {}

    for comp_name, comp_rules in WS_RULES.items():
        if comp_name not in components:
            continue

        box = components[comp_name]
        comp_report = {}

        for rule in comp_rules["requirements"]:
            human = anchors.get(rule["human_anchor"])
            obj = extract_object_anchor(rule["object_anchor"], box)
            comp_report[rule["check_id"]] = evaluate_rule(rule, human, obj)

        report[comp_name] = comp_report

    return report
