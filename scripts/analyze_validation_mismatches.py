#!/usr/bin/env python3
"""
Analyze validation mismatches to understand why the model is choosing 290/330 incorrectly.

This will help us understand:
1. What the correct strategy selection logic should be
2. Why the model learned a simplified 290/330 heuristic
3. What training data patterns led to this behavior
4. How to fix the strategy logic in Recommendation 6
"""

import json
import sys
from collections import defaultdict, Counter

def load_data(file_path):
    """Load and parse JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_drive_from_prompt(prompt):
    """Extract drive distance from prompt text."""
    import re
    patterns = [
        r"average drive:\s*(\d{2,4})\s*yards",
        r"drives\s*(\d{2,4})\s*yards", 
        r"with a\s*(\d{2,4})-yard drive",
        r"drive is exactly\s*(\d{2,4})\s*yards",
        r"average drive of\s*(\d{2,4})\s*yards",
        r"drive is\s*(\d{2,4})\s*yards",
    ]
    for pat in patterns:
        m = re.search(pat, prompt, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except:
                pass
    return None

def analyze_validation_mismatches():
    """Analyze validation mismatches and training data patterns."""
    
    # Load validation data and inference results
    val_data = load_data("data/bethpage_black/val.jsonl")
    inference_data = load_data("outputs/bethpage-lora/inference_val_final.jsonl")
    training_data = load_data("data/bethpage_black/train_balanced.jsonl")
    
    print("=== VALIDATION MISMATCH ANALYSIS ===\n")
    
    # Create lookup for inference results
    inference_lookup = {}
    for item in inference_data:
        key = (item.get("hole"), item.get("prompt"))
        inference_lookup[key] = item.get("completion", "")
    
    # Analyze each validation case
    mismatches = []
    correct_matches = []
    
    for val_item in val_data:
        hole = val_item.get("hole")
        prompt = val_item.get("prompt", "")
        expected_cutoff = val_item.get("expected_cutoff_yards")
        
        # Get validation data (including validation examples marked as use_for_training=false)
        if val_item.get("split") != "val":
            continue
            
        # Get model prediction
        key = (hole, prompt)
        predicted_completion = inference_lookup.get(key, "")
        
        # Extract predicted cutoff
        import re
        pred_match = re.search(r"Strategy:\s*(\d+)\s*yards", predicted_completion)
        predicted_cutoff = int(pred_match.group(1)) if pred_match else None
        
        # Extract drive distance
        drive_distance = extract_drive_from_prompt(prompt)
        
        # Analyze match/mismatch
        if predicted_cutoff == expected_cutoff:
            correct_matches.append({
                "hole": hole,
                "drive": drive_distance,
                "expected": expected_cutoff,
                "predicted": predicted_cutoff,
                "prompt": prompt
            })
        else:
            mismatches.append({
                "hole": hole,
                "drive": drive_distance,
                "expected": expected_cutoff,
                "predicted": predicted_cutoff,
                "prompt": prompt
            })
    
    print(f"Total validation cases: {len(correct_matches) + len(mismatches)}")
    print(f"Correct predictions: {len(correct_matches)}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Accuracy: {len(correct_matches)/(len(correct_matches) + len(mismatches))*100:.1f}%\n")
    
    # Analyze mismatch patterns
    print("=== MISMATCH PATTERN ANALYSIS ===\n")
    
    # Group mismatches by drive distance
    drive_mismatches = defaultdict(list)
    for mm in mismatches:
        drive_mismatches[mm["drive"]].append(mm)
    
    for drive, cases in sorted(drive_mismatches.items()):
        print(f"Drive Distance: {drive} yards ({len(cases)} mismatches)")
        expected_cutoffs = [case["expected"] for case in cases]
        predicted_cutoffs = [case["predicted"] for case in cases]
        
        print(f"  Expected cutoffs: {sorted(set(expected_cutoffs))}")
        print(f"  Predicted cutoffs: {sorted(set(predicted_cutoffs))}")
        print(f"  Most common expected: {Counter(expected_cutoffs).most_common(1)[0]}")
        print(f"  Most common predicted: {Counter(predicted_cutoffs).most_common(1)[0]}")
        
        # Show a representative example
        example = cases[0]
        print(f"  Example: Hole {example['hole']}")
        print(f"    Expected: {example['expected']} | Predicted: {example['predicted']}")
        print(f"    Prompt: {example['prompt'][:100]}...")
        print()
    
    # Analyze training data to understand the model's learning
    print("=== TRAINING DATA ANALYSIS ===\n")
    
    training_cutoffs = []
    training_drives = []
    drive_to_cutoff_map = defaultdict(list)
    
    for item in training_data:
        # Skip negative examples
        if item.get("is_correct") is False or item.get("use_for_training") is False:
            continue
            
        expected_cutoff = item.get("expected_cutoff_yards")
        prompt = item.get("prompt", "")
        drive = extract_drive_from_prompt(prompt)
        
        if expected_cutoff and drive:
            training_cutoffs.append(expected_cutoff)
            training_drives.append(drive)
            drive_to_cutoff_map[drive].append(expected_cutoff)
    
    print(f"Training examples analyzed: {len(training_cutoffs)}")
    print(f"Unique cutoffs in training: {sorted(set(training_cutoffs))}")
    print(f"Cutoff frequency: {Counter(training_cutoffs).most_common()}")
    print(f"Drive range: {min(training_drives)} - {max(training_drives)} yards")
    print()
    
    # Analyze drive-to-cutoff mapping from training
    print("=== TRAINING DRIVE → CUTOFF PATTERNS ===\n")
    for drive in sorted(drive_to_cutoff_map.keys()):
        cutoffs = drive_to_cutoff_map[drive]
        print(f"Drive {drive} yards → {Counter(cutoffs).most_common()}")
    
    # Find the root cause of 290/330 bias
    print("\n=== ROOT CAUSE ANALYSIS ===\n")
    
    cutoff_290_drives = [d for d, c_list in drive_to_cutoff_map.items() 
                        if 290 in c_list]
    cutoff_330_drives = [d for d, c_list in drive_to_cutoff_map.items() 
                        if 330 in c_list]
    
    print(f"Drives that map to 290 cutoff: {sorted(cutoff_290_drives)}")
    print(f"Drives that map to 330 cutoff: {sorted(cutoff_330_drives)}")
    
    # Check if model learned incorrect generalizations
    print(f"\nModel's apparent logic:")
    print(f"- 330-yard drives → 330 strategy (model learned this)")
    print(f"- 290-yard drives → 290 strategy (model learned this)")
    print(f"- Other drives → default to 290 (model's fallback)")
    
    # Show what the correct logic should be
    print(f"\nCorrect logic analysis:")
    for mm in mismatches[:5]:  # Show first 5 mismatches
        drive = mm["drive"]
        expected = mm["expected"]
        predicted = mm["predicted"]
        print(f"Drive {drive} → Expected {expected}, Got {predicted}")
        print(f"  Issue: Model should learn optimal cutoff selection, not drive matching")

def main():
    analyze_validation_mismatches()

if __name__ == "__main__":
    main()
