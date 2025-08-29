#!/usr/bin/env python3
"""
Create a strategy logic training dataset that teaches proper golf strategy selection.

Key principles to teach:
1. Choose the HIGHEST available cutoff ≤ drive distance
2. Strategy selection is about optimization, not drive matching
3. Multiple strategies may be available - pick the best one
4. Explicit reasoning about why one strategy is better than another
"""

import json
import sys
from collections import defaultdict
import random

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

def get_available_strategies(hole_data):
    """Get all available strategy cutoffs for a hole."""
    strategies = hole_data.get("tee_shot_strategies", [])
    cutoffs = []
    for strategy in strategies:
        cutoff = strategy.get("cutoff_distance")
        if isinstance(cutoff, int):
            cutoffs.append(cutoff)
    return sorted(set(cutoffs))

def get_optimal_strategy(available_cutoffs, drive_distance):
    """Get the optimal strategy cutoff for a given drive distance."""
    if not available_cutoffs:
        return None
    
    # Find all cutoffs that the golfer can achieve
    eligible = [c for c in available_cutoffs if c <= drive_distance]
    
    # Return the highest eligible cutoff (most aggressive/optimal)
    return max(eligible) if eligible else min(available_cutoffs)

def create_strategy_logic_examples(original_data):
    """Create training examples that teach strategy optimization logic."""
    
    # Group data by hole to understand available strategies
    hole_data = defaultdict(lambda: {"strategies": set(), "examples": []})
    
    for item in original_data:
        if item.get("is_correct") is False or item.get("use_for_training") is False:
            continue
            
        hole = item.get("hole")
        strategies = item.get("tee_shot_strategies", [])
        
        for strategy in strategies:
            cutoff = strategy.get("cutoff_distance")
            if isinstance(cutoff, int):
                hole_data[hole]["strategies"].add(cutoff)
        
        hole_data[hole]["examples"].append(item)
    
    strategy_examples = []
    
    # Create explicit strategy selection examples for each hole
    for hole, data in hole_data.items():
        available_cutoffs = sorted(data["strategies"])
        if len(available_cutoffs) < 2:
            continue  # Skip holes with only one strategy
        
        # Get hole info from first example
        first_example = data["examples"][0]
        par = first_example.get("par")
        yardage = first_example.get("yardage")
        
        # Create examples for different drive distances
        test_drives = [250, 270, 290, 310, 330, 350]
        
        for drive in test_drives:
            optimal_cutoff = get_optimal_strategy(available_cutoffs, drive)
            if not optimal_cutoff:
                continue
                
            # Skip if drive is much outside the useful range for this hole
            if drive < min(available_cutoffs) - 50 or drive > max(available_cutoffs) + 50:
                continue
                
            # Find non-optimal alternatives to contrast against
            non_optimal = [c for c in available_cutoffs if c != optimal_cutoff and c <= drive]
            
            # Create primary strategy selection example
            strategy_example = {
                "hole": hole,
                "prompt": f"Hole {hole}, Bethpage Black: Par {par}, {yardage} yards. Golfer drives {drive} yards. Best strategy?",
                "completion": f"Strategy: {optimal_cutoff} yards",
                "par": par,
                "yardage": yardage,
                "drive_distance": drive,
                "available_cutoffs": available_cutoffs,
                "optimal_cutoff": optimal_cutoff,
                "expected_cutoff_yards": optimal_cutoff,
                "reasoning": f"Choose {optimal_cutoff} (highest available ≤ {drive})",
                "use_for_training": True,
                "split": "train",
                "example_type": "strategy_optimization"
            }
            strategy_examples.append(strategy_example)
            
            # Create comparative reasoning examples if there are alternatives
            if non_optimal:
                for suboptimal in non_optimal:
                    comparison_example = {
                        "hole": hole,
                        "prompt": f"Hole {hole}, Par {par}, {yardage} yards. Golfer drives {drive} yards. Available: {suboptimal} and {optimal_cutoff} yards. Which is better?",
                        "completion": f"Strategy: {optimal_cutoff} yards",
                        "par": par,
                        "yardage": yardage,
                        "drive_distance": drive,
                        "available_cutoffs": [suboptimal, optimal_cutoff],
                        "optimal_cutoff": optimal_cutoff,
                        "expected_cutoff_yards": optimal_cutoff,
                        "reasoning": f"Choose {optimal_cutoff} over {suboptimal} (higher is better when achievable)",
                        "use_for_training": True,
                        "split": "train",
                        "example_type": "strategy_comparison"
                    }
                    strategy_examples.append(comparison_example)
            
            # Create explicit rule examples
            if optimal_cutoff < drive:
                rule_example = {
                    "hole": hole,
                    "prompt": f"Hole {hole}: Drive {drive} yards, strategies available: {', '.join(map(str, available_cutoffs))}. Rule: choose highest ≤ drive distance.",
                    "completion": f"Strategy: {optimal_cutoff} yards",
                    "par": par,
                    "yardage": yardage,
                    "drive_distance": drive,
                    "available_cutoffs": available_cutoffs,
                    "optimal_cutoff": optimal_cutoff,
                    "expected_cutoff_yards": optimal_cutoff,
                    "reasoning": f"Rule application: max({[c for c in available_cutoffs if c <= drive]}) = {optimal_cutoff}",
                    "use_for_training": True,
                    "split": "train",
                    "example_type": "rule_application"
                }
                strategy_examples.append(rule_example)
    
    return strategy_examples

def create_edge_case_examples():
    """Create examples for edge cases and common mistakes."""
    
    edge_cases = [
        # Drive matching mistakes
        {
            "hole": 99,
            "prompt": "Drive 290 yards. Strategies: 290, 330 yards. Common mistake: choose drive distance.",
            "completion": "Strategy: 290 yards",
            "drive_distance": 290,
            "optimal_cutoff": 290,
            "expected_cutoff_yards": 290,
            "reasoning": "When drive exactly matches available strategy, that's correct",
            "example_type": "edge_case_correct"
        },
        {
            "hole": 99,
            "prompt": "Drive 320 yards. Strategies: 290, 330 yards. Don't just match drive distance.",
            "completion": "Strategy: 290 yards",
            "drive_distance": 320,
            "optimal_cutoff": 290,
            "expected_cutoff_yards": 290,
            "reasoning": "Drive 320 can't achieve 330 strategy, so use 290",
            "example_type": "edge_case_sub_optimal"
        },
        {
            "hole": 99,
            "prompt": "Drive 340 yards. Strategies: 290, 330 yards. Choose optimal, not matching.",
            "completion": "Strategy: 330 yards",
            "drive_distance": 340,
            "optimal_cutoff": 330,
            "expected_cutoff_yards": 330,
            "reasoning": "Drive 340 can achieve 330 strategy (optimal choice)",
            "example_type": "edge_case_optimal"
        }
    ]
    
    return edge_cases

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_strategy_logic_dataset.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Load original data
    original_data = load_data(input_file)
    print(f"Loaded {len(original_data)} records from {input_file}")
    
    # Create strategy logic examples
    strategy_examples = create_strategy_logic_examples(original_data)
    print(f"Created {len(strategy_examples)} strategy logic examples")
    
    # Add edge cases
    edge_cases = create_edge_case_examples()
    print(f"Created {len(edge_cases)} edge case examples")
    
    # Combine and add original good examples (filtered)
    all_examples = []
    
    # Add original good examples
    for item in original_data:
        if item.get("is_correct") is False or item.get("use_for_training") is False:
            continue
        item["example_type"] = "original"
        all_examples.append(item)
    
    # Add new strategy logic examples
    all_examples.extend(strategy_examples)
    all_examples.extend(edge_cases)
    
    # Shuffle for better training
    random.shuffle(all_examples)
    
    # Write to output file
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Final dataset: {len(all_examples)} examples")
    print(f"  Original: {len([e for e in all_examples if e.get('example_type') == 'original'])}")
    print(f"  Strategy optimization: {len([e for e in all_examples if e.get('example_type') == 'strategy_optimization'])}")
    print(f"  Strategy comparison: {len([e for e in all_examples if e.get('example_type') == 'strategy_comparison'])}")
    print(f"  Rule application: {len([e for e in all_examples if e.get('example_type') == 'rule_application'])}")
    print(f"  Edge cases: {len([e for e in all_examples if e.get('example_type', '').startswith('edge_case')])}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
