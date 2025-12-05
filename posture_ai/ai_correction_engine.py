import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import openai

# Load .env
load_dotenv()

# API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY missing in .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------
# SYSTEM PROMPT FOR GPT MODELS
# -----------------------------
AI_SYSTEM_PROMPT = """
You are POSTURA AI — a certified ergonomic assistant specializing in ISO 9241-5 posture and workstation evaluation.

You will receive a JSON object containing:
- posture metrics
- workstation metrics
- overall severity

Your responsibilities:
1. Identify KEY ergonomic risks (neck strain, wrist extension, etc.).
2. Explain WHY each risk violates ISO 9241-5 (including posture and workstation issues).
3. Provide **CLEAR** posture corrections.
4. Provide **CLEAR** workstation corrections.
5. Provide **3–5 ergonomic exercises** based on posture and workstation analysis.
6. **Summarize the risk level** (red/yellow/green) for posture and workstation.
7. Produce **STRICT JSON output only**, in the following format:

JSON structure:
{
  "posture_corrections": [],
  "workstation_corrections": [],
  "iso_explanations": [],
  "risk_summary": "",  # Summary of identified risks (neck strain, wrist, etc.)
  "exercise_recommendations": [],  # 3–5 exercises
  "final_advice": ""  # Final advice to improve ergonomics
}
"""

# -----------------------------------------
#  ★ Model Fallback Logic (Enterprise Safe)
# -----------------------------------------
MODEL_PRIORITY = [
    "gpt-4.1",      # Highest quality
    "gpt-4o",       # Excellent fallback
    "gpt-4o-mini"   # Cheapest, lowest reasoning
]


def call_openai_model(model, messages):
    """Attempts a single OpenAI model call."""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2
        )
    except openai.RateLimitError:
        print(f"⚠ Quota exceeded for {model}. Trying next model...")
        return None
    except Exception as e:
        print(f"⚠ Error calling {model}: {e}")
        return None


def generate_ergonomic_correction(unified_iso_report):
    """
    Phase-4 AI Correction Engine
    → Automatically handles GPT-4.1 → GPT-4o → GPT-4o-mini fallback.
    """

    user_input = f"""
    Here is the user's ISO evaluation data:

    {json.dumps(unified_iso_report, indent=4)}

    Provide ergonomic corrections in strict JSON format.
    """

    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    response = None

    # Try models in priority order
    for model in MODEL_PRIORITY:
        print(f"🤖 Trying model: {model} ...")
        response = call_openai_model(model, messages)
        if response is not None:
            print(f"✅ Model {model} succeeded.")
            break

    if response is None:
        return {"error": "All OpenAI models failed due to quota or connectivity issues."}

    # Extract text safely
    raw_text = response.choices[0].message.content

    # Parse JSON output
    try:
        parsed = json.loads(raw_text)
        return parsed
    except Exception:
        return {
            "error": "JSON parsing failed",
            "raw_output": raw_text
        }
