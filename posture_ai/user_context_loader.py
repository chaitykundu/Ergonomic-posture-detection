import json

def load_user_context(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Assuming one user entry for now
    return data[0]
