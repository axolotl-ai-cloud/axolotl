#!/usr/bin/env python3
"""
Fix inconsistent training data by standardizing strategy selection completions.
All strategy tasks should return "Strategy: X yards" format.
"""

import json
import re
from pathlib import Path

def extract_strategy_number(completion_text):
    """Extract the strategy number from various completion formats."""
    # Look for patterns like "290-yard strategy", "330-yard cutoff", "Strategy: 290 yards"
    patterns = [
        r'Strategy:\s*(\d+)\s*yards',  # "Strategy: 290 yards"
        r'(\d+)-yard\s+(?:strategy|cutoff)',  # "290-yard strategy" or "330-yard cutoff"
        r'use\s+the\s+(\d+)-yard',  # "use the 290-yard strategy"
        r'select\s+the\s+(\d+)-yard',  # "Select the 308-yard cutoff"
        r'(\d+)\s+yards',  # Any "X yards" pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, completion_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    print(f"WARNING: Could not extract strategy from: {completion_text}")
    return None

def fix_strategy_completion(record):
    """Fix strategy selection completion to standard format."""
    if record.get('task_type') != 'strategy_selection':
        return record
    
    completion = record['completion']
    
    # If already in correct format, keep it
    if completion.startswith('Strategy:') and 'yards' in completion:
        return record
    
    # Extract the expected strategy number
    expected_yards = record.get('expected_cutoff_yards')
    if expected_yards:
        # Use the expected value
        strategy_yards = expected_yards
    else:
        # Try to extract from completion text
        strategy_yards = extract_strategy_number(completion)
    
    if strategy_yards:
        # Fix the completion
        record['completion'] = f"Strategy: {strategy_yards} yards"
        print(f"Fixed hole {record.get('hole', '?')}: {completion[:50]}... -> Strategy: {strategy_yards} yards")
    else:
        print(f"ERROR: Could not fix hole {record.get('hole', '?')}: {completion}")
    
    return record

def main():
    input_file = Path("data/bethpage_black/train_multitask_perfect.jsonl")
    output_file = Path("data/bethpage_black/train_multitask_fixed.jsonl")
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    fixed_count = 0
    total_strategy_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                record = json.loads(line.strip())
                
                if record.get('task_type') == 'strategy_selection':
                    total_strategy_count += 1
                    original_completion = record['completion']
                    
                    record = fix_strategy_completion(record)
                    
                    if record['completion'] != original_completion:
                        fixed_count += 1
                
                # Write the (possibly fixed) record
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"ERROR: Processing line {line_num}: {e}")
    
    print(f"\nSummary:")
    print(f"Total strategy selection tasks: {total_strategy_count}")
    print(f"Fixed completions: {fixed_count}")
    print(f"Already correct: {total_strategy_count - fixed_count}")
    print(f"Fixed dataset saved to: {output_file}")

if __name__ == "__main__":
    main()
