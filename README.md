Posture Detection System (AI-Powered)
===

A real-time AI-powered Posture Detection System that analyzes human posture using computer vision and deep learning. 
The system detects whether a person has correct or incorrect posture and provides real-time feedback through an API-based backend built with FastAPI.

Features
===

Deep Learning-based posture classification

Image-based posture analysis (extendable to real-time video)

FastAPI backend for high-performance inference

Modular and scalable architecture

Preprocessing pipeline for image normalization

Structured API responses for integration with frontend/mobile apps

Debug-friendly logging system

System Architecture
===

Client (Web / Mobile / Postman)
          │
          ▼
   FastAPI Server
          │
          ▼
 Image Preprocessing Module
          │
          ▼
   Trained ML Model (CNN / PyTorch)
          │
          ▼
 Posture Classification Output
          │
          ▼
 Structured JSON Response

Installation & Setup
===
1. Clone Repository

git clone https://github.com/chaitykundu/Ergonomic-posture-detection.git

cd posture-detection

2. Create Virtual Environment

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3. Install Dependencies

pip install -r requirements.txt

Run server
===

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Api
===
http://10.10.12.60:8000/upload/

NGROK
===
ngrok http 8000
