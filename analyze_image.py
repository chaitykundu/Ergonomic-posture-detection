import cv2
import mediapipe as mp
import json
import os
from dotenv import load_dotenv
load_dotenv()
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
from posture_ai.compliance import calculate_compliance_percentage
#from posture_ai.pdf_equipment_report import generate_equipment_pdf


# ----------------------------------------
# Phase 4 – GPT-4.1 Ergonomic Correction Engine
# ----------------------------------------
from posture_ai.ai_correction_engine import generate_ergonomic_correction

from posture_ai.exercise import recommend_exercises
from posture_ai.exercise_adapter import build_exercise_onboarding
from posture_ai.exercise_backend import fetch_exercises_from_backend
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")



# ----------------------------------------
# Phase 5 – PDF Report Generator
# ----------------------------------------
from posture_ai.pdf_report_generator import generate_pdf_report

# ----------------------------------------
# Main Image Analysis Function (Phase 1–4)
# ----------------------------------------
def analyze_image(user_context: dict):
    print("user context", user_context)
    print("\n===================================================")
    print("📸 Starting analysis from user context")
    print("===================================================\n")

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
            return {
                "success": False,
                "error_code": "NO_HUMAN_DETECTED",
                "message": "No human posture detected in the image. Please upload a clear image with a visible person.",
                "data": None
            }


        # Pose landmarks
        lm = results.pose_landmarks.landmark

        # ISO posture metrics
        posture_report = get_posture_report(lm, W, H)

        # Posture anchors
        anchors = compute_posture_anchors(lm, frame.shape)

        # Draw skeleton
        # mp.solutions.drawing_utils.draw_landmarks(
        #     frame,
        #    results.pose_landmarks,
        #    mp_pose.POSE_CONNECTIONS
        # )

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

    try:
        if posture_report and workstation_report:
            print("🎨 Drawing correction arrows...")
            frame = apply_correction_arrows(
                frame, posture_report, workstation_report, lm, W, H
            )
    except Exception as e:
        print(f"⚠️ Warning: Could not apply correction arrows: {e}")

    #frame = apply_correction_arrows(frame, posture_report, workstation_report, lm, W, H)


    # ----------------------------------------
    # Phase 3 – Unified ISO Output
    # ----------------------------------------
    final_iso = merge_iso_reports(posture_report, workstation_report)

    compliance_report = calculate_compliance_percentage(final_iso)
    def severity_from_score(compliance_percent: float):
        if compliance_percent >= 70:
            return "green"
        elif compliance_percent >= 40:
            return "yellow"
        else:
            return "red"

    final_iso["compliance"] = compliance_report
    final_iso["overall_severity"] = severity_from_score(compliance_report)

    print("\n================ Unified ISO Analysis ================")
    print(json.dumps(final_iso, indent=4))

    # ----------------------------------------
    # Phase 4 – GPT-4.1 Ergonomic Corrections
    # ----------------------------------------
    print("\n================ GPT-4.1 AI Correction ================")
    try:
        ai_report = generate_ergonomic_correction(final_iso, user_context)
        posture_corrections = ai_report.pop("posture_corrections", [])
        workstation_corrections = ai_report.pop("workstation_corrections", [])
        ai_report.update({
            "corrections": posture_corrections + workstation_corrections
        })
        print("my output ----------------------")
        print(json.dumps(ai_report, indent=4))
        
    except Exception as e:
        print(f"⚠️ Warning: AI correction failed: {e}")
        ai_report = {"error": str(e), "recommendations": []}


    # ----------------------------------------
    # Phase 6 – Exercise Recommendation Engine
    # ----------------------------------------

    try:
        exercise_onboarding = build_exercise_onboarding(
            user_context=user_context,
            final_iso=final_iso
        )
        print("Exercise onboarding:", exercise_onboarding)

        exercise_api_response = fetch_exercises_from_backend(
            body_regions=exercise_onboarding["body_regions"]
        )
        print("API response:", exercise_api_response)

        exercise_plan = recommend_exercises(
            onboarding_data=exercise_onboarding,
            exercise_api_response=exercise_api_response
        )
        print("Final plan:", exercise_plan)

        ai_report["exercise_recommendations"] = exercise_plan

    except Exception as e:
        print(f"⚠️ Exercise recommendation failed: {e}")
        ai_report["exercise_recommendations"] = {
            "error": str(e),
            "recommended_session": []
        }


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

    annotated_image_url = f"{BASE_URL}/output/annotated_{timestamp}.jpg"
    ai_report["annotated_image_url"] = annotated_image_url

    # # Now generate the PDF report with ISO results + AI corrections
    # generate_pdf_report(final_iso, ai_report, image_path=output_file)

    # # ----------------------------------------
    # # Equipment PDF Report (NEW)
    # # ----------------------------------------
    # from posture_ai.pdf_equipment_report import generate_equipment_pdf

    # equipment_pdf_path = generate_equipment_pdf(
    #     ai_report.get("equipment_recommendations", [])
    #     #ai_report.get("exercise_recommendations", None)
    # )

    #equipment_pdf_filename = os.path.basename(equipment_pdf_path)
    #equipment_pdf_url = f"{BASE_URL}/output/{equipment_pdf_filename}"

   # ai_report["equipment_pdf_url"] = equipment_pdf_url

    try:
        pdf_path = generate_pdf_report(
            final_iso_report=final_iso,
            ai_report=ai_report,
            image_path=output_file
        )
        pdf_filename = os.path.basename(pdf_path)
        pdf_url = f"{BASE_URL}/output/{pdf_filename}"

        ai_report["pdf_report_url"] = pdf_url

        from posture_ai.pdf_equipment_report import generate_equipment_excel

        equipment_xlsx_path = generate_equipment_excel(
            ai_report.get("equipment_recommendations", [])
        )

        filename = os.path.basename(equipment_xlsx_path)
        ai_report["equipment_excel_url"] = f"{BASE_URL}/output/{filename}"


        #ai_report["pdf_report_path"] = pdf_path
        print("✅ PDF report generated:", pdf_path)
    except Exception as e:
        print(f"⚠️ PDF generation failed: {e}")


    return final_iso, ai_report



# ----------------------------------------
# Script Entry Point
# ----------------------------------------
if __name__ == "__main__":
    print("🚀 Starting ISO Posture + Workstation + AI Analysis")

    USER_CONTEXT_PATH = "files/dummy.json"
    user_context = load_user_context(USER_CONTEXT_PATH)

    analyze_image(user_context)
