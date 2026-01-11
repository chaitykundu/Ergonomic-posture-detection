import json
import os
import re
from dotenv import load_dotenv

import openai
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

# --------------------------------------------------
# 1. Load API key and init client
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

# New-style client (recommended in current docs)
client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------------------------
# 2. System prompt (same logic, just cleaned a bit)
# --------------------------------------------------
AI_SYSTEM_PROMPT = """
You are POSTURA AI — a certified ergonomic assistant specializing in ISO 9241-5 posture and workstation evaluation.

You will receive a JSON object containing:
- posture metrics
- workstation metrics
- overall severity

Your responsibilities:
1. Identify KEY ergonomic risks (neck strain, wrist extension, etc.).
2. Explain WHY each risk violates ISO 9241-5 (including posture and workstation issues).
3. Provide CLEAR posture corrections.
4. Provide CLEAR workstation corrections.
5. Provide 3–5 ergonomic exercises based on posture and workstation analysis.
6. Summarize the risk level (red/yellow/green) for posture and workstation.
7. Produce STRICT JSON output only, in exactly this format:

{
  "posture_corrections": [],
  "workstation_corrections": [],
  "iso_explanations": [],
  "risk_summary": "",
  "exercise_recommendations": [],
  "final_advice": ""
}
"""


# --------------------------------------------------
# 3. Model priority list
# --------------------------------------------------
MODEL_PRIORITY = [
    "gpt-4.1",    # highest quality
    "gpt-4o",     # great fallback
    "gpt-4o-mini" # cheapest fallback
]


# --------------------------------------------------
# 4. Small helper: try to pull JSON out of messy text
# --------------------------------------------------
def _extract_json_block(raw_text: str) -> str:
    """
    Tries to extract a JSON object from the model output.
    Handles things like ```json ... ``` or extra explanation text.
    """
    # Direct JSON first
    raw_text = raw_text.strip()

    # If it already looks like JSON, just return
    if raw_text.startswith("{") and raw_text.endswith("}"):
        return raw_text

    # Try to find a {...} block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        return match.group(0)

    # Fallback: return as-is (will fail json.loads, but we keep raw)
    return raw_text


# --------------------------------------------------
# 5. Wrapper for one model call with proper errors
# --------------------------------------------------
def call_openai_model(model: str, messages):
    """
    Call one OpenAI chat model using the new client API.
    Returns the response or None on handled error.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return response

    except RateLimitError as e:
        print(f"⚠ Rate limit hit for {model}: {e}")
        return None

    except AuthenticationError as e:
        print(f"❌ Authentication error: {e}")
        return None

    except APIError as e:
        print(f"⚠ API error for {model}: {e}")
        return None

    except Exception as e:
        print(f"⚠ Unexpected OpenAI error for {model}: {e}")
        return None


# --------------------------------------------------
# 6. Main function used by your pipeline
# --------------------------------------------------
def generate_ergonomic_correction(unified_iso_report: dict, user_context: dict) -> dict:
    """
    Phase-4 AI Correction Engine.
    Takes the unified ISO JSON (posture + workstation + severity)
    and returns a structured ergonomic correction JSON.
    """

    user_input = (
    "User-reported health context (pre-assessment questionnaire):\n"
    + json.dumps(user_context, indent=4)
    + "\n\nISO posture + workstation evaluation:\n"
    + json.dumps(unified_iso_report, indent=4)
    + "\n\nIMPORTANT RULES:\n"
    "- Give higher priority to body regions with higher pain intensity\n"
    "- Escalate risk if pain intensity >= 7\n"
    "- Align advice with both ISO violations AND reported symptoms\n"
    "- Return ONLY the JSON object described in the system prompt"
    )

    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
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
        return {"error": "All OpenAI models failed due to quota, auth, or connectivity issues."}

    # New client returns: response.choices[0].message.content
    raw_text = response.choices[0].message.content or ""

    # Try to get clean JSON substring
    json_block = _extract_json_block(raw_text)

    try:
        parsed = json.loads(json_block)
        return parsed
    except Exception as e:
        print(f"⚠ JSON parse failed: {e}")
        return {
            "error": "JSON parsing failed",
            "raw_output": raw_text,
        }
