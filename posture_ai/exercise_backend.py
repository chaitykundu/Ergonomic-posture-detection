# posture_ai/exercise_backend.py

import os
import requests
from typing import List, Dict


def fetch_exercises_from_backend(body_regions: List[str]) -> Dict:
    """
    Fetch exercises filtered by body regions from backend API
    """

    exercise_api_url = os.getenv("EXERCISE_API_URL")

    if not exercise_api_url:
        raise RuntimeError("EXERCISE_API_URL is not set")

    response = requests.get(
        exercise_api_url,
        json={"body_regions": body_regions},
        timeout=10
    )

    response.raise_for_status()
    return response.json()
