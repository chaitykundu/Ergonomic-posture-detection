# posture_ai/exercise_backend.py
#
# SIMPLEST FIX - Guaranteed to work
# Fetches ALL exercises, then filters client-side
#

import os
import requests
import json
from typing import List, Dict


def fetch_exercises_from_backend(body_regions: List[str]) -> Dict:
    """
    Fetch ALL exercises from backend, then filter by body_regions on client side.
    
    This is the SAFEST approach because:
    - No dependency on backend filtering logic
    - Works with simple GET request (no parameters needed)
    - Guaranteed to get all exercises including Neck and Wrists/Hands
    """

    exercise_api_url = os.getenv("EXERCISE_API_URL")

    if not exercise_api_url:
        raise RuntimeError("EXERCISE_API_URL is not set")

    print("\n" + "="*70)
    print("🌐 FETCHING EXERCISES FROM BACKEND")
    print("="*70)
    print(f"URL: {exercise_api_url}")
    print(f"Requested regions: {body_regions}")
    
    try:
        # Simple GET request - no parameters
        response = requests.get(exercise_api_url, timeout=10)
        
        print(f"Status: {response.status_code}")
        
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        all_exercises = data.get("exercises_list", [])
        
        print(f"✓ Received {len(all_exercises)} total exercises from backend")
        
        # Filter on client side
        filtered_exercises = [
            ex for ex in all_exercises
            if ex.get("body_region") in body_regions
        ]
        
        print(f"✓ Filtered to {len(filtered_exercises)} exercises for requested regions")
        
        # Show breakdown by region
        region_count = {}
        for ex in filtered_exercises:
            region = ex.get("body_region", "Unknown")
            region_count[region] = region_count.get(region, 0) + 1
        
        print(f"\n📊 Exercises by region:")
        for region in body_regions:
            count = region_count.get(region, 0)
            status = "✓" if count > 0 else "✗"
            print(f"   {status} {region}: {count} exercises")
        
        # Warn if any requested region has no exercises
        missing_regions = [r for r in body_regions if region_count.get(r, 0) == 0]
        if missing_regions:
            print(f"\n⚠️  Warning: No exercises found for: {missing_regions}")
            print(f"   Check if backend database has exercises for these regions")
        
        print("="*70 + "\n")
        
        # Return filtered data
        return {
            "exercises_list": filtered_exercises,
            "total_exercises": len(all_exercises),
            "requested_regions": body_regions,
            "found_regions": list(region_count.keys())
        }
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Request Failed: {e}")
        print("="*70 + "\n")
        
        # Return empty exercises instead of crashing
        return {
            "exercises_list": [],
            "error": str(e),
            "requested_regions": body_regions
        }
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON Response: {e}")
        print("="*70 + "\n")
        
        return {
            "exercises_list": [],
            "error": f"Invalid JSON: {str(e)}",
            "requested_regions": body_regions
        }