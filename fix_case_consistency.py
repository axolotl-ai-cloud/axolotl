#!/usr/bin/env python3
"""
Fix case consistency between task labels in prompts and metadata.
Makes both use lowercase for consistency.
"""

import json

def fix_case_consistency(input_file, output_file):
    records = []
    changes_made = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                
                # Fix prompt task labels - convert to lowercase
                prompt = record.get("prompt", "")
                
                # Replace uppercase task labels with lowercase
                if "TASK: STRATEGY_SELECTION" in prompt:
                    prompt = prompt.replace("TASK: STRATEGY_SELECTION", "TASK: strategy_selection")
                    changes_made += 1
                
                if "TASK: DESCRIPTION_SYNTHESIS" in prompt:
                    prompt = prompt.replace("TASK: DESCRIPTION_SYNTHESIS", "TASK: description_synthesis")
                    changes_made += 1
                
                record["prompt"] = prompt
                records.append(record)
    
    # Write fixed records
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Fixed {changes_made} case inconsistencies")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"📊 Total records: {len(records)}")

if __name__ == "__main__":
    fix_case_consistency(
        "data/bethpage_black/train_multitask_fixed.jsonl",
        "data/bethpage_black/train_multitask_case_fixed.jsonl"
    )
