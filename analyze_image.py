import cv2
import mediapipe as mp
import json
import os
import time

# Import from your project modules
from webcam_detector import get_posture_report
from postura_workstation import (
    compute_posture_anchors,
    detect_workstation_objects_raw,
    filter_workstation_for_person,
    evaluate_workstation_iso,
)

# -------------------------------
# Unified ISO Report (Phase 3)
# -------------------------------
def merge_iso_reports(posture, workstation):
    final = {
        "posture": posture,
        "workstation": workstation,
        "overall_severity": "green"
    }

    severities = []

    # Posture severities
    for data in posture.values():
        severities.append(data["severity"])

    # Workstation severities
    for comp in workstation.values():
        for rule in comp.values():
            severities.append(rule["severity"])

    # Overall severity
    if "red" in severities:
        final["overall_severity"] = "red"
    elif "yellow" in severities:
        final["overall_severity"] = "yellow"
    else:
        final["overall_severity"] = "green"

    return final


# -------------------------------
# Main Image Analysis Function
# -------------------------------
def analyze_image(image_path):

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"❌ ERROR: Could not load image: {image_path}")

    # Resize for consistent YOLO + Mediapipe performance
    frame = cv2.resize(frame, (960, 720))
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True,
                      min_detection_confidence=0.5) as pose:

        results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            print("❌ No human detected in the image.")
            return None

        # Extract pose landmarks
        lm = results.pose_landmarks.landmark

        # Get ISO posture metrics
        posture_report = get_posture_report(lm, W, H)

        # Compute anchors for workstation rules
        anchors = compute_posture_anchors(lm, frame.shape)

        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # --------------------------------------
    # Workstation Object Detection (YOLO)
    # --------------------------------------
    raw_components = detect_workstation_objects_raw(frame)

    # Single-person workstation filtering
    selected_components = filter_workstation_for_person(
        raw_components, anchors, frame.shape
    )

    # Evaluate workstation ISO rules
    workstation_report = evaluate_workstation_iso(
        selected_components, anchors, frame.shape
    )

    # --------------------------------------
    # Unified ISO Output
    # --------------------------------------
    final_iso = merge_iso_reports(posture_report, workstation_report)

    print("\n================ Unified ISO Analysis ================")
    print(json.dumps(final_iso, indent=4))

    # --------------------------------------
    # Save Annotated Image (Auto Folder)
    # --------------------------------------
    os.makedirs("output", exist_ok=True)

    timestamp = int(time.time())
    output_file = f"output/annotated_{timestamp}.jpg"

    success = cv2.imwrite(output_file, frame)

    if success:
        print(f"\n📸 Saved annotated image at: {output_file}")
    else:
        print("\n❌ Failed to save annotated image")

    return final_iso


# -------------------------------
# Script Entry Point
# -------------------------------
if __name__ == "__main__":
    DEFAULT_IMAGE = "files/pose1.webp"
    print(f" Analyzing {DEFAULT_IMAGE} ...")
    analyze_image(DEFAULT_IMAGE)
