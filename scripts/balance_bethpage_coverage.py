import argparse
import json
import os
import random
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def analyze_coverage(records: List[Dict]) -> Dict[str, Any]:
    """Analyze dataset coverage patterns"""
    coverage = {
        "hole_counts": Counter(),
        "drive_counts": Counter(),
        "strategy_counts": Counter(),
        "hole_drive_pairs": Counter(),
        "edge_cases": [],
        "instruction_types": Counter()
    }
    
    import re
    drive_re_list = [
        re.compile(r"average drive:\s*(\d{2,4})\s*yards", re.IGNORECASE),
        re.compile(r"drives\s*(\d{2,4})\s*yards", re.IGNORECASE),
        re.compile(r"with a\s*(\d{2,4})-yard drive", re.IGNORECASE),
        re.compile(r"drive is exactly\s*(\d{2,4})\s*yards", re.IGNORECASE),
        re.compile(r"average drive of\s*(\d{2,4})\s*yards", re.IGNORECASE),
        re.compile(r"drive is\s*(\d{2,4})\s*yards", re.IGNORECASE),
    ]
    
    def extract_drive(prompt: str) -> Optional[int]:
        for rx in drive_re_list:
            m = rx.search(prompt or "")
            if m:
                try:
                    return int(m.group(1))
                except:
                    pass
        return None
    
    for r in records:
        if r.get("is_correct") is False:
            continue  # Skip negatives
            
        hole = r.get("hole")
        prompt = r.get("prompt", "")
        drive = extract_drive(prompt)
        expected = r.get("expected_cutoff_yards")
        
        coverage["hole_counts"][hole] += 1
        if drive:
            coverage["drive_counts"][drive] += 1
            coverage["hole_drive_pairs"][(hole, drive)] += 1
        if expected:
            coverage["strategy_counts"][expected] += 1
            
        # Detect instruction patterns
        if "edge case" in prompt.lower():
            coverage["instruction_types"]["edge_case"] += 1
        elif "paraphrase" in prompt.lower():
            coverage["instruction_types"]["paraphrase"] += 1
        elif "synthetic" in prompt.lower():
            coverage["instruction_types"]["synthetic"] += 1
        elif "hole" in prompt.lower() and "what" in prompt.lower():
            coverage["instruction_types"]["direct_question"] += 1
        else:
            coverage["instruction_types"]["other"] += 1
            
        # Find edge cases: drives below available strategies
        strategies = r.get("tee_shot_strategies", [])
        if drive and strategies:
            min_strategy = min(s.get("cutoff_distance", 999) for s in strategies if s.get("cutoff_distance"))
            if drive < min_strategy:
                coverage["edge_cases"].append({
                    "hole": hole,
                    "drive": drive,
                    "min_strategy": min_strategy,
                    "expected": expected
                })
    
    return coverage

def generate_balanced_examples(records: List[Dict], target_count: int = 50) -> List[Dict]:
    """Generate additional examples to balance coverage"""
    coverage = analyze_coverage(records)
    new_examples = []
    
    # Template patterns based on existing data
    templates = [
        "Hole {hole}, Bethpage Black: Par {par}, {yardage} yards. Golfer's average drive: {drive} yards. Which tee shot strategy should be used?",
        "Strategy question: For Hole {hole}, Bethpage Black (Par {par}, {yardage} yards), what tee shot should a golfer with a {drive}-yard drive use?",
        "Edge case: Hole {hole}, Bethpage Black, Par {par}, {yardage} yards. Golfer's average drive: {drive} yards (below minimum strategy distance). Which tee shot strategy should be used?",
        "Paraphrase: Hole {hole} at Bethpage Black, Par {par}, {yardage} yards. Golfer drives {drive} yards. What is the optimal tee shot strategy?",
    ]
    
    # Get existing hole data for reference
    hole_data = {}
    for r in records:
        hole = r.get("hole")
        if hole and hole not in hole_data:
            hole_data[hole] = {
                "par": r.get("par"),
                "yardage": r.get("yardage"),
                "description": r.get("description"),
                "strategies": r.get("tee_shot_strategies", [])
            }
    
    # Generate examples for underrepresented holes/drives
    hole_counts = coverage["hole_counts"]
    drive_counts = coverage["drive_counts"]
    
    # Focus on holes with fewer examples
    underrep_holes = [h for h, count in hole_counts.items() if count < 3 and h in hole_data]
    
    # Drive distances to emphasize (including edge cases)
    focus_drives = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]
    
    count = 0
    for hole in underrep_holes:
        if count >= target_count:
            break
            
        hdata = hole_data[hole]
        available_cutoffs = sorted([s.get("cutoff_distance") for s in hdata["strategies"] if s.get("cutoff_distance")])
        
        for drive in focus_drives:
            if count >= target_count:
                break
                
            # Skip if we already have this combination
            if (hole, drive) in coverage["hole_drive_pairs"]:
                continue
                
            # Determine expected strategy
            eligible = [c for c in available_cutoffs if c <= drive]
            expected = max(eligible) if eligible else available_cutoffs[0] if available_cutoffs else drive
            
            # Choose template based on case type
            if drive < min(available_cutoffs) if available_cutoffs else False:
                template = templates[2]  # Edge case template
                template_type = "edge_case"
            else:
                template = random.choice(templates[:2] + templates[3:])  # Avoid edge case template
                template_type = "balanced"
            
            prompt = template.format(
                hole=hole,
                par=hdata["par"],
                yardage=hdata["yardage"],
                drive=drive
            )
            
            # Build strategies for this example
            if drive < min(available_cutoffs) if available_cutoffs else False:
                # For edge cases, show the minimum available strategy
                strategies = [{"cutoff_distance": min(available_cutoffs), "remaining_distance": hdata["yardage"] - min(available_cutoffs), "advantage": "Minimum available option", "strategy_id": f"cutoff_{min(available_cutoffs)}", "strategy_label": f"cutoff {min(available_cutoffs)} yards"}]
            else:
                # Show the chosen strategy
                strategies = [{"cutoff_distance": expected, "remaining_distance": hdata["yardage"] - expected, "advantage": None, "strategy_id": f"cutoff_{expected}", "strategy_label": f"cutoff {expected} yards"}]
            
            completion = f"Strategy: {expected} yards"
            
            new_example = {
                "hole": hole,
                "prompt": prompt,
                "completion": completion,
                "par": hdata["par"],
                "yardage": hdata["yardage"],
                "description": hdata["description"],
                "tee_shot_strategies": strategies,
                "expected_cutoff_yards": expected,
                "expected_strategy_id": f"cutoff_{expected}",
                "use_for_training": True,
                "split": "train",
                "generated": True,
                "template_type": template_type
            }
            
            new_examples.append(new_example)
            count += 1
    
    return new_examples

def print_coverage_report(coverage: Dict[str, Any]):
    """Print detailed coverage analysis"""
    print("=== COVERAGE ANALYSIS ===")
    print(f"Holes represented: {len(coverage['hole_counts'])}")
    print(f"Drive distances: {len(coverage['drive_counts'])}")
    print(f"Strategy values: {len(coverage['strategy_counts'])}")
    print(f"Edge cases found: {len(coverage['edge_cases'])}")
    
    print("\nHole distribution:")
    for hole, count in sorted(coverage['hole_counts'].items()):
        print(f"  Hole {hole}: {count} examples")
    
    print("\nDrive distribution:")
    for drive, count in sorted(coverage['drive_counts'].items()):
        print(f"  {drive} yards: {count} examples")
    
    print("\nInstruction types:")
    for itype, count in coverage['instruction_types'].items():
        print(f"  {itype}: {count} examples")
    
    if coverage['edge_cases']:
        print(f"\nEdge cases (drive < min strategy):")
        for case in coverage['edge_cases'][:5]:  # Show first 5
            print(f"  Hole {case['hole']}: {case['drive']} yard drive, min strategy {case['min_strategy']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL file")
    ap.add_argument("--output", required=True, help="Output balanced JSONL file")
    ap.add_argument("--generate-count", type=int, default=50, help="Number of examples to generate")
    ap.add_argument("--report-only", action="store_true", help="Only show coverage report, don't generate")
    args = ap.parse_args()
    
    records = load_jsonl(args.input)
    coverage = analyze_coverage(records)
    
    print_coverage_report(coverage)
    
    if args.report_only:
        return
    
    print(f"\nGenerating {args.generate_count} balanced examples...")
    new_examples = generate_balanced_examples(records, args.generate_count)
    
    print(f"Generated {len(new_examples)} new examples")
    
    # Combine and write
    all_records = records + new_examples
    
    with open(args.output, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(all_records)} total records to {args.output}")
    
    # Show updated coverage
    print("\n=== UPDATED COVERAGE ===")
    new_coverage = analyze_coverage(all_records)
    print(f"Total examples: {len(all_records)}")
    print(f"Holes: {len(new_coverage['hole_counts'])}")
    print(f"Drive distances: {len(new_coverage['drive_counts'])}")
    print(f"Strategy values: {len(new_coverage['strategy_counts'])}")

if __name__ == "__main__":
    main()
