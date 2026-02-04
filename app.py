import os
import json
from datetime import datetime
from fastapi import Request

import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from analyze_image import analyze_image
from posture_ai.exercise import recommend_exercises
from dotenv import load_dotenv
load_dotenv()
from posture_ai.exercise_backend import fetch_exercises_from_backend
# from posture_ai.pdf_report_generator import generate_pdf_report



allowed_origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://10.10.12.62:8000",
]


app = FastAPI(
    title="POSTURA - Ergonomics Analyzer",
    description="Upload a photo → Get instant posture analysis + beautiful PDF report",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory="output"), name="output")

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILES_DIR = os.path.join(BASE_DIR, "files")


# Optional: Add a simple health check
@app.get("/")
async def root():
    return {
        "message": "POSTURA Ergonomics API is running!",
        "docs": "/docs",
        "time": datetime.now().isoformat()
    }


@app.post("/api/analyze-posture")
async def upload_image(
    #request: Request
    file: UploadFile = File(...),
    payload: str = Form(...)
):
    print("sent file", file)
    print("sent payload", payload)
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}
    
    # keep original name
    filename = file.filename
    full_path = os.path.join(FILES_DIR, filename)

    success = cv2.imwrite(full_path, img)
    
    if not success:
        return {"error": "Failed to save image"}
    
    relative_path = os.path.join("files", filename)

    try:
        payload: dict = json.loads(payload)
        payload.update({"image_data": {"image_path": relative_path}})
        output = analyze_image(payload)

        if output is None:
            return {
                "success": False,
                "error_code": "ANALYSIS_FAILED",
                "message": "Posture analysis failed unexpectedly.",
                "data": None
            }

        if isinstance(output, dict) and output.get("success") is False:
            return output

        return {
            "success": True,
            "data": output
        }

    except Exception as e:
        print("Error %s" % str(e))


@app.post("/api/analyze-exercises")
async def analyze_exercises(
    payload: str = Form(...)
):
    print("sent exercise payload", payload)

    try:
        # Parse JSON payload
        payload: dict = json.loads(payload)

        # Fetch exercises from backend
        exercise_api_response = fetch_exercises_from_backend(
            body_regions=payload.get("body_regions", [])
        )
        print("")

        # Generate exercise plan
        exercise_plan = recommend_exercises(
            onboarding_data=payload,
            exercise_api_response=exercise_api_response
        )

        return {
            "data": {
                "exercise_recommendations": exercise_plan
            }
        }

    except Exception as e:
        print("Error %s" % str(e))
        return {
            "error": str(e)
        }
