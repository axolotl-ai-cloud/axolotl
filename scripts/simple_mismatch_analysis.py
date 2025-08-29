#!/usr/bin/env python3
"""
Simple analysis of validation mismatches and training patterns.
"""

import json
from collections import Counter, defaultdict
import re

def extract_drive_from_prompt(prompt):
    """Extract drive distance from prompt text."""
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

# Load training data
training_data = []
with open("data/bethpage_black/train_balanced.jsonl", 'r') as f:
    for line in f:
        if line.strip():
            training_data.append(json.loads(line))

# Load validation inference results  
val_inference = []
with open("outputs/bethpage-lora/inference_val_final.jsonl", 'r') as f:
    for line in f:
        if line.strip():
            val_inference.append(json.loads(line))

print("=== TRAINING DATA ANALYSIS ===")

# Analyze training patterns (only positive examples)
training_patterns = defaultdict(list)
cutoff_counts = Counter()

for item in training_data:
    if item.get("is_correct") is False or item.get("use_for_training") is False:
        continue
        
    expected_cutoff = item.get("expected_cutoff_yards")
    prompt = item.get("prompt", "")
    drive = extract_drive_from_prompt(prompt)
    
    if expected_cutoff and drive:
        training_patterns[drive].append(expected_cutoff)
        cutoff_counts[expected_cutoff] += 1

print(f"Training examples used: {sum(cutoff_counts.values())}")
print(f"Cutoff distribution: {cutoff_counts.most_common()}")
print()

# Show drive → cutoff mapping from training
print("TRAINING DRIVE → CUTOFF PATTERNS:")
for drive in sorted(training_patterns.keys()):
    cutoffs = training_patterns[drive]
    cutoff_dist = Counter(cutoffs)
    print(f"Drive {drive:3d} yards → {cutoff_dist.most_common()}")

print()

print("=== VALIDATION ANALYSIS ===")

# Parse validation log manually since it shows the mismatches clearly
validation_errors = [
    # From the log, manually extracted key patterns
    {"drive": 330, "expected": 320, "predicted": 330, "note": "Hole 7 cases"},
    {"drive": 330, "expected": 308, "predicted": 330, "note": "Hole 16 case"}, 
    {"drive": 290, "expected": 308, "predicted": 290, "note": "Hole 5 cases"},
    {"drive": 290, "expected": 330, "predicted": 290, "note": "Multiple holes"},
    {"drive": 280, "expected": 330, "predicted": 290, "note": "Multiple holes"},
    {"drive": 310, "expected": 330, "predicted": 290, "note": "Multiple holes"},
]

print("KEY MISMATCH PATTERNS:")
for error in validation_errors:
    drive = error["drive"]
    expected = error["expected"]
    predicted = error["predicted"]
    note = error["note"]
    print(f"Drive {drive} → Expected {expected}, Got {predicted} ({note})")

print()

print("=== ROOT CAUSE ANALYSIS ===")

# The model learned these patterns from training:
print("MODEL'S LEARNED LOGIC:")
print("1. If drive = 330 → predict 330 (learned from training)")
print("2. If drive = 290 → predict 290 (learned from training)")  
print("3. If drive = 280/310 → default to 290 (nearest learned pattern)")

print()

print("PROBLEMS WITH MODEL'S LOGIC:")
print("1. Drive 330 → should be 320 (not 330) for Hole 7")
print("2. Drive 330 → should be 308 (not 330) for Hole 16")
print("3. Drive 290 → should be 308 (not 290) for Hole 5") 
print("4. Drive 280/310 → should be 330 (not 290) for multiple holes")

print()

print("THE CORE ISSUE:")
print("Model learned DRIVE MATCHING instead of STRATEGY SELECTION")
print("- It thinks: drive distance = strategy distance")
print("- It should think: find optimal strategy ≤ drive distance")

print()

print("TRAINING DATA PROBLEMS:")
# Check if training data has the same issue
drive_330_training = training_patterns.get(330, [])
drive_290_training = training_patterns.get(290, [])

print(f"Training: 330-yard drives → {Counter(drive_330_training).most_common()}")
print(f"Training: 290-yard drives → {Counter(drive_290_training).most_common()}")

if 330 in drive_330_training and 290 in drive_290_training:
    print("✗ Training data teaches drive=strategy matching!")
    print("  This explains why model learned the wrong pattern")
else:
    print("✓ Training data has correct patterns")
    print("  Model mislearned despite good training data")

print()
print("RECOMMENDATION 6: Need strategy selection logic, not drive matching")
