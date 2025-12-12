# main.py (or app.py)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from datetime import datetime
from analyze_image import analyze_image
from posture_ai.pdf_report_generator import generate_pdf_report

app = FastAPI(
    title="POSTURA - Ergonomics Analyzer",
    description="Upload a photo → Get instant posture analysis + beautiful PDF report",
    version="1.0.0"
)

# Directories
UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post(
    "/analyze",
    response_class=FileResponse,
    summary="Upload photo → Get posture report PDF",
    responses={
        200: {"content": {"application/pdf": {}}, "description": "PDF report generated successfully"},
        400: {"description": "No file uploaded"},
        422: {"description": "Invalid file type"},
        500: {"description": "Analysis failed"}
    }
)
async def analyze_posture(file: UploadFile = File(...)):
    """
    Upload a sitting posture photo and receive a professional PDF ergonomics report instantly.
    Supported formats: JPG, JPEG, PNG
    """
    
    # === 1. Validate file type ===
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=422, detail="Only JPG/JPEG/PNG images are allowed")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # === 2. Generate unique filenames with UUID (safe & unique) ===
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in {".jpg", ".jpeg", ".png"}:
        file_ext = ".jpg"  # fallback

    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    image_filename = f"photo_{timestamp}_{unique_id}{file_ext}"
    image_path = os.path.join(UPLOAD_DIR, image_filename)

    # === 3. Save uploaded image ===
    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Image saved: {image_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

    # === 4. Run AI posture analysis ===
    try:
        final_iso_report, ai_report = analyze_image(image_path)
    except Exception as e:
        # Clean up uploaded image on failure
        if os.path.exists(image_path):
            os.remove(image_path)
        raise HTTPException(status_code=500, detail=f"Posture analysis failed: {str(e)}")

    # === 5. Generate beautiful PDF report (unique name) ===
    try:
        # Let your generate_pdf_report create its own timestamped name
        pdf_path = generate_pdf_report(
            final_iso_report=final_iso_report,
            ai_report=ai_report,
            image_path=image_path  # annotated image
        )
        print(f"PDF generated: {pdf_path}")
    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    # === 6. Return the PDF for instant download ===
    # Use a clean name for the downloaded file
    download_filename = f"POSTURA_Ergonomics_Report_{timestamp}.pdf"

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=download_filename,
        headers={"Content-Disposition": f"attachment; filename={download_filename}"}
    )


# Optional: Add a simple health check
@app.get("/")
async def root():
    return {
        "message": "POSTURA Ergonomics API is running!",
        "docs": "/docs",
        "time": datetime.now().isoformat()
    }


# Run with: uvicorn main:app --reload