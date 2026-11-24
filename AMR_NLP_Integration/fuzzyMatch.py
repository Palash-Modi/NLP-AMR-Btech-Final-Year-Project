import os
import numpy as np
from rapidfuzz import process, fuzz

TARGETS_FILENAME = "amr_targets.npy"

def get_discovered_locations():
    locations = {}
    # Check current directory or parent directory for the targets file
    path = TARGETS_FILENAME if os.path.exists(TARGETS_FILENAME) else f"../{TARGETS_FILENAME}"
    
    if os.path.exists(path):
        try:
            saved_targets = np.load(path, allow_pickle=True).item()
            if isinstance(saved_targets, dict):
                for k, v in saved_targets.items():
                    if isinstance(v, list) and len(v) == 2:
                        locations[k] = v
        except Exception:
            pass
    return locations

def extract_safest_coordinate(llm_target):
    if not llm_target or llm_target == "none":
        return None

    valid_locations = get_discovered_locations()
    valid_keys = list(valid_locations.keys())
    
    if not valid_keys:
        print("UI Log: No objects or colored balls discovered yet.")
        return None

    matches = process.extract(
        llm_target,
        valid_keys,
        scorer=fuzz.token_sort_ratio,
        limit=2
    )

    if not matches:
        return None

    best_match = matches[0]
    best_string, best_score, _ = best_match

    # Reject targets below similarity threshold
    if best_score < 75:
        print(f"UI Log: Object '{llm_target}' not found in discovered map (Confidence: {best_score}%).")
        return None

    # Prevent ambiguity collision if two colored balls have very similar names
    if len(matches) > 1:
        second_best = matches[1]
        if (best_score - second_best[1]) < 5:
            print(f"UI Log: Ambiguity collision between {best_string} and {second_best[0]}. Aborting.")
            return None

    print(f"UI Log: Resolved target '{llm_target}' to discovered object '{best_string}' at {valid_locations[best_string]}")
    return valid_locations[best_string]