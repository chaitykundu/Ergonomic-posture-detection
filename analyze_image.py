import cv2
import mediapipe as mp
import json
import os
import time

# ----------------------------------------
# Import Posture & Workstation Engines
# ----------------------------------------
from posture_ai.webcam_detector import get_posture_report
from posture_ai.user_context_loader import load_user_context
from posture_ai.postura_workstation import (
    compute_posture_anchors,
    detect_workstation_objects_raw,
    filter_workstation_for_person,
    evaluate_workstation_iso,
)

# ----------------------------------------
# Phase 3 – Unified ISO Output
# ----------------------------------------
from posture_ai.unified_iso_engine import merge_iso_reports
from posture_ai.arrow_annotation import apply_correction_arrows

# ----------------------------------------
# Phase 4 – GPT-4.1 Ergonomic Correction Engine
# ----------------------------------------
from posture_ai.ai_correction_engine import generate_ergonomic_correction

# ----------------------------------------
# Phase 5 – PDF Report Generator
# ----------------------------------------
from posture_ai.pdf_report_generator import generate_pdf_report

# ----------------------------------------
# Main Image Analysis Function (Phase 1–4)
# ----------------------------------------
def analyze_image(user_context: dict):

    print("\n===================================================")
    print("📸 Starting analysis from user context")
    print("===================================================\n")

    # ----------------------------------------
    # Extract image path from user context
    # ----------------------------------------
    image_path = user_context["image_data"]["image_path"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    print(f"📸 Image loaded from context: {image_path}")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"❌ ERROR: Could not load image: {image_path}")

    # Resize for consistent performance
    frame = cv2.resize(frame, (960, 720))
    H, W, _ = frame.shape

    # Convert for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ----------------------------------------
    # Phase 1 – Pose Detection
    # ----------------------------------------
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True,
                      min_detection_confidence=0.5) as pose:

        results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            print("❌ No human detected in this image.")
            return None

        # Pose landmarks
        lm = results.pose_landmarks.landmark

        # ISO posture metrics
        posture_report = get_posture_report(lm, W, H)

        # Posture anchors
        anchors = compute_posture_anchors(lm, frame.shape)

        # Draw skeleton
        #mp.solutions.drawing_utils.draw_landmarks(
            #frame,
           # results.pose_landmarks,
           # mp_pose.POSE_CONNECTIONS
        #)

    # ----------------------------------------
    # Phase 2 – Workstation Detection (YOLO)
    # ----------------------------------------
    raw_components = detect_workstation_objects_raw(frame)

    selected_components = filter_workstation_for_person(
        raw_components, anchors, frame.shape
    )

    workstation_report = evaluate_workstation_iso(
        selected_components, anchors, frame.shape
    )

    frame = apply_correction_arrows(frame, posture_report, workstation_report, lm, W, H)


    # ----------------------------------------
    # Phase 3 – Unified ISO Output
    # ----------------------------------------
    final_iso = merge_iso_reports(posture_report, workstation_report)

    print("\n================ Unified ISO Analysis ================")
    print(json.dumps(final_iso, indent=4))

    # ----------------------------------------
    # Phase 4 – GPT-4.1 Ergonomic Corrections
    # ----------------------------------------
    print("\n================ GPT-4.1 AI Correction ================")

    ai_report = generate_ergonomic_correction(final_iso, user_context)

    print(json.dumps(ai_report, indent=4))

    # ----------------------------------------
    # Phase 5 – PDF Report Generation
    # ----------------------------------------
    os.makedirs("output", exist_ok=True)
    timestamp = int(time.time())
    output_file = f"output/annotated_{timestamp}.jpg"

    if cv2.imwrite(output_file, frame):
        print(f"\n📸 Annotated image saved at: {output_file}")
    else:
        print("\n❌ Failed to save annotated image.")

    # Now generate the PDF report with ISO results + AI corrections
    generate_pdf_report(final_iso, ai_report, image_path=output_file)

    return final_iso, ai_report


# ----------------------------------------
# Script Entry Point
# ----------------------------------------
if __name__ == "__main__":
    print("🚀 Starting ISO Posture + Workstation + AI Analysis")

    USER_CONTEXT_PATH = "files/dummy.json"
    user_context = load_user_context(USER_CONTEXT_PATH)

    analyze_image(user_context)
