#!/usr/bin/env python3
"""
Process Hole Descriptions for Training Data
Converts the filled-out hole_descriptions_input.json into training examples
Handles de-duplication against existing dataset descriptions
"""

import json
import os
from difflib import SequenceMatcher

def similarity(a, b):
    """Calculate similarity between two strings (0-1)"""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def extract_existing_descriptions():
    """Extract all existing descriptions from the training dataset"""
    existing_descriptions = {}
    data_file = "data/bethpage_black/train_strategy_logic.jsonl"
    
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                hole = example.get('hole')
                description = example.get('description', '').strip()
                
                if hole and description and len(description) > 10:
                    if hole not in existing_descriptions:
                        existing_descriptions[hole] = set()
                    existing_descriptions[hole].add(description)
    
    return existing_descriptions

def deduplicate_descriptions(hole_num, new_descriptions, existing_descriptions):
    """Remove duplicates and very similar descriptions"""
    hole_existing = existing_descriptions.get(hole_num, set())
    unique_descriptions = []
    
    for new_desc in new_descriptions:
        new_desc = new_desc.strip()
        if not new_desc or len(new_desc) < 20:  # Skip empty or very short
            continue
            
        is_duplicate = False
        
        # Check against existing descriptions
        for existing_desc in hole_existing:
            if similarity(new_desc, existing_desc) > 0.85:  # 85% similarity threshold
                print(f"  Hole {hole_num}: Skipping duplicate (85% similar to existing)")
                print(f"    New: {new_desc[:60]}...")
                print(f"    Existing: {existing_desc[:60]}...")
                is_duplicate = True
                break
        
        # Check against other new descriptions for this hole
        if not is_duplicate:
            for existing_new in unique_descriptions:
                if similarity(new_desc, existing_new) > 0.85:
                    print(f"  Hole {hole_num}: Skipping duplicate within new descriptions")
                    print(f"    Duplicate: {new_desc[:60]}...")
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_descriptions.append(new_desc)
    
    return unique_descriptions

def generate_description_training_examples(hole_num, par, yardage, descriptions):
    """Generate training examples for description synthesis"""
    if len(descriptions) < 2:
        return []  # Need at least 2 descriptions to synthesize
    
    examples = []
    
    # Create synthesis examples with different combinations
    if len(descriptions) == 2:
        # Two descriptions -> synthesis
        prompt = f"Combine these 2 descriptions of Hole {hole_num} at Bethpage Black (Par {par}, {yardage} yards) into one cohesive narrative:\n\n"
        prompt += f"1. {descriptions[0]}\n\n"
        prompt += f"2. {descriptions[1]}\n\n"
        prompt += "Synthesized description:"
        
        # Create a sample synthesis (you can refine this)
        completion = f"Hole {hole_num} combines the key challenges from both perspectives, creating a demanding test that requires strategic thinking and precise execution."
        
        examples.append({
            "hole": hole_num,
            "prompt": prompt,
            "completion": completion,
            "par": par,
            "yardage": yardage,
            "source_descriptions": descriptions,
            "example_type": "description_synthesis",
            "use_for_training": True,
            "split": "train"
        })
    
    elif len(descriptions) == 3:
        # Three descriptions -> synthesis
        prompt = f"Combine these 3 descriptions of Hole {hole_num} at Bethpage Black (Par {par}, {yardage} yards) into one cohesive narrative:\n\n"
        prompt += f"1. {descriptions[0]}\n\n"
        prompt += f"2. {descriptions[1]}\n\n"
        prompt += f"3. {descriptions[2]}\n\n"
        prompt += "Synthesized description:"
        
        completion = f"Hole {hole_num} presents a complex challenge that demands careful consideration of multiple strategic elements and course features."
        
        examples.append({
            "hole": hole_num,
            "prompt": prompt,
            "completion": completion,
            "par": par,
            "yardage": yardage,
            "source_descriptions": descriptions,
            "example_type": "description_synthesis",
            "use_for_training": True,
            "split": "train"
        })
        
        # Also create 2-description combinations
        for i in range(len(descriptions)):
            for j in range(i + 1, len(descriptions)):
                pair = [descriptions[i], descriptions[j]]
                prompt = f"Combine these 2 descriptions of Hole {hole_num} at Bethpage Black (Par {par}, {yardage} yards) into one cohesive narrative:\n\n"
                prompt += f"1. {pair[0]}\n\n"
                prompt += f"2. {pair[1]}\n\n"
                prompt += "Synthesized description:"
                
                completion = f"Hole {hole_num} requires strategic thinking to navigate its key challenges effectively."
                
                examples.append({
                    "hole": hole_num,
                    "prompt": prompt,
                    "completion": completion,
                    "par": par,
                    "yardage": yardage,
                    "source_descriptions": pair,
                    "example_type": "description_synthesis_pair",
                    "use_for_training": True,
                    "split": "train"
                })
    
    return examples

def process_hole_descriptions(input_file, output_file):
    """Main processing function"""
    print("🏌️ Processing Hole Descriptions for Training")
    print("=" * 50)
    
    # Load input data
    with open(input_file, 'r') as f:
        hole_data = json.load(f)
    
    # Extract existing descriptions for de-duplication
    print("📊 Extracting existing descriptions...")
    existing_descriptions = extract_existing_descriptions()
    
    # Process each hole
    all_training_examples = []
    total_descriptions = 0
    total_examples = 0
    
    for hole_key, data in hole_data.items():
        hole_num = int(hole_key.replace('hole_', ''))
        par = data['par']
        yardage = data['yardage']
        
        # Collect non-empty descriptions
        new_descriptions = []
        for i in range(1, 4):  # description_1, description_2, description_3
            desc = data.get(f'description_{i}', '').strip()
            if desc:
                new_descriptions.append(desc)
        
        if not new_descriptions:
            print(f"  Hole {hole_num}: No descriptions provided")
            continue
        
        print(f"  Hole {hole_num}: Processing {len(new_descriptions)} descriptions")
        
        # De-duplicate
        unique_descriptions = deduplicate_descriptions(hole_num, new_descriptions, existing_descriptions)
        
        if len(unique_descriptions) < len(new_descriptions):
            print(f"    Deduplicated: {len(new_descriptions)} → {len(unique_descriptions)}")
        
        if len(unique_descriptions) < 2:
            print(f"    Skipping: Need at least 2 unique descriptions for synthesis")
            continue
        
        # Generate training examples
        training_examples = generate_description_training_examples(hole_num, par, yardage, unique_descriptions)
        all_training_examples.extend(training_examples)
        
        total_descriptions += len(unique_descriptions)
        total_examples += len(training_examples)
        print(f"    Generated {len(training_examples)} training examples")
    
    # Save training examples
    print(f"\n📝 Summary:")
    print(f"  Total unique descriptions: {total_descriptions}")
    print(f"  Total training examples: {total_examples}")
    
    if all_training_examples:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for example in all_training_examples:
                f.write(json.dumps(example) + '\n')
        print(f"  Saved to: {output_file}")
    else:
        print("  No training examples generated")
    
    return all_training_examples

def merge_with_existing_dataset(description_file, strategy_file, output_file):
    """Merge description training with existing strategy training"""
    print(f"\n🔄 Merging datasets...")
    
    # Load existing strategy training data
    strategy_examples = []
    with open(strategy_file, 'r') as f:
        for line in f:
            strategy_examples.append(json.loads(line.strip()))
    
    # Load description training data
    description_examples = []
    if os.path.exists(description_file):
        with open(description_file, 'r') as f:
            for line in f:
                description_examples.append(json.loads(line.strip()))
    
    # Combine datasets
    all_examples = strategy_examples + description_examples
    
    # Save merged dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"  Strategy examples: {len(strategy_examples)}")
    print(f"  Description examples: {len(description_examples)}")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Merged dataset saved to: {output_file}")
    
    return len(all_examples)

def main():
    input_file = "hole_descriptions_input.json"
    description_output = "data/bethpage_black/description_training.jsonl"
    strategy_file = "data/bethpage_black/train_strategy_logic.jsonl"
    merged_output = "data/bethpage_black/train_enhanced.jsonl"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please fill out the template first!")
        return
    
    # Process descriptions
    process_hole_descriptions(input_file, description_output)
    
    # Merge with existing strategy data
    total_examples = merge_with_existing_dataset(description_output, strategy_file, merged_output)
    
    print(f"\n✅ Processing complete!")
    print(f"Enhanced training dataset ready: {merged_output}")
    print(f"Total training examples: {total_examples}")

if __name__ == "__main__":
    main()
