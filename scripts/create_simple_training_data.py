#!/usr/bin/env python3
"""
Create a simple, clean training dataset focused on strategy selection logic.

This approach eliminates the complexity and focuses on core learning:
1. Simple, consistent prompt format
2. Clean "Strategy: N yards" completions only
3. No negative examples or confusing patterns
4. Direct mapping from drive distance to optimal strategy
"""

import json
import sys
from pathlib import Path

def load_data(file_path):
    """Load and parse JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_drive_distance(prompt):
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

def get_optimal_strategy(strategies, drive_distance):
    """Get the optimal strategy cutoff for a given drive distance."""
    if not strategies:
        return None
    
    cutoffs = sorted([s.get("cutoff_distance") for s in strategies 
                     if isinstance(s.get("cutoff_distance"), int)])
    if not cutoffs:
        return None
    
    # Find the highest cutoff that the golfer can achieve
    eligible = [c for c in cutoffs if c <= drive_distance]
    return max(eligible) if eligible else cutoffs[0]

def create_simple_prompt(hole, par, yardage, drive_distance):
    """Create a simple, consistent prompt format."""
    return f"Hole {hole}, Bethpage Black: Par {par}, {yardage} yards. Golfer drives {drive_distance} yards. Best strategy?"

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_simple_training_data.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Load original data
    data = load_data(input_file)
    print(f"Loaded {len(data)} records from {input_file}")
    
    # Track unique (hole, drive) combinations to avoid duplicates
    seen_combinations = set()
    simple_examples = []
    
    for record in data:
        # Skip if marked as incorrect or not for training
        if record.get("is_correct") is False or record.get("use_for_training") is False:
            continue
            
        hole = record.get("hole")
        par = record.get("par")
        yardage = record.get("yardage")
        prompt = record.get("prompt", "")
        strategies = record.get("tee_shot_strategies", [])
        
        # Extract drive distance
        drive_distance = extract_drive_distance(prompt)
        if not drive_distance:
            continue
            
        # Skip duplicates
        combo_key = (hole, drive_distance)
        if combo_key in seen_combinations:
            continue
        seen_combinations.add(combo_key)
        
        # Get optimal strategy
        optimal_cutoff = get_optimal_strategy(strategies, drive_distance)
        if not optimal_cutoff:
            continue
            
        # Create simple training example
        simple_prompt = create_simple_prompt(hole, par, yardage, drive_distance)
        simple_completion = f"Strategy: {optimal_cutoff} yards"
        
        simple_example = {
            "hole": hole,
            "prompt": simple_prompt,
            "completion": simple_completion,
            "par": par,
            "yardage": yardage,
            "drive_distance": drive_distance,
            "optimal_cutoff": optimal_cutoff,
            "expected_cutoff_yards": optimal_cutoff,
            "use_for_training": True,
            "split": "train"
        }
        
        simple_examples.append(simple_example)
    
    # Write simple training data
    with open(output_file, 'w') as f:
        for example in simple_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created {len(simple_examples)} simple training examples")
    print(f"Saved to {output_file}")
    
    # Show statistics
    drive_distances = [ex["drive_distance"] for ex in simple_examples]
    optimal_cutoffs = [ex["optimal_cutoff"] for ex in simple_examples]
    
    print(f"\nDrive distances: {min(drive_distances)} - {max(drive_distances)} yards")
    print(f"Strategy cutoffs: {min(optimal_cutoffs)} - {max(optimal_cutoffs)} yards")
    print(f"Unique holes: {len(set(ex['hole'] for ex in simple_examples))}")

if __name__ == "__main__":
    main()
