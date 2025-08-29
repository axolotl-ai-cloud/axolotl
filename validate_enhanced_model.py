#!/usr/bin/env python3
"""
Validate the enhanced model performance on both strategy selection and description synthesis.
"""
import json
import re

def validate_enhanced_model():
    """Validate enhanced model on both tasks"""
    
    # Read the inference results
    with open("outputs/bethpage-lora/inference_full_enhanced.jsonl", "r") as f:
        results = [json.loads(line) for line in f]
    
    strategy_errors = 0
    description_errors = 0
    strategy_total = 0
    description_total = 0
    
    validation_log = []
    
    for result in results:
        prompt = result["prompt"]
        completion = result["completion"]
        hole = result.get("hole", "Unknown")
        
        # Check if this is a description synthesis task
        if "Combine these" in prompt and "Synthesized description:" in prompt:
            description_total += 1
            
            # Description synthesis should NOT contain "Strategy: X yards"
            if re.search(r"Strategy:\s*\d+\s*yards", completion, re.IGNORECASE):
                description_errors += 1
                validation_log.append(f"ERROR: Description synthesis task returned strategy format")
                validation_log.append(f"Hole {hole} | Prompt: {prompt[:100]}...")
                validation_log.append(f"Completion: {completion}")
                validation_log.append("")
            else:
                # Check if it's a reasonable description
                if len(completion.strip()) < 10:
                    description_errors += 1
                    validation_log.append(f"ERROR: Description synthesis too short")
                    validation_log.append(f"Hole {hole} | Completion: {completion}")
                    validation_log.append("")
        
        # Check if this is a strategy selection task
        else:
            strategy_total += 1
            
            # Extract strategy from completion
            strategy_match = re.search(r"Strategy:\s*(\d+)\s*yards", completion, re.IGNORECASE)
            
            if not strategy_match:
                strategy_errors += 1
                validation_log.append(f"ERROR: Strategy task did not return strategy format")
                validation_log.append(f"Hole {hole} | Completion: {completion}")
                validation_log.append("")
                continue
            
            chosen_strategy = int(strategy_match.group(1))
            
            # Compare with expected for strategy tasks
            if "expected_cutoff_yards" in result:
                expected_strategy = result["expected_cutoff_yards"]
                if chosen_strategy != expected_strategy:
                    strategy_errors += 1
                    validation_log.append(f"ERROR: Strategy mismatch")
                    validation_log.append(f"Hole {hole} | Expected: {expected_strategy} | Chosen: {chosen_strategy}")
                    validation_log.append(f"Prompt: {prompt}")
                    validation_log.append(f"Completion: {completion}")
                    validation_log.append("")
    
    # Write detailed validation log
    with open("outputs/bethpage-lora/enhanced_validation_detailed.txt", "w") as f:
        f.write(f"Enhanced Model Validation Results\\n")
        f.write(f"=================================\\n\\n")
        f.write(f"Strategy Tasks: {strategy_total} total, {strategy_errors} errors ({100*strategy_errors/strategy_total:.1f}% error rate)\\n")
        f.write(f"Description Tasks: {description_total} total, {description_errors} errors ({100*description_errors/description_total:.1f}% error rate)\\n")
        f.write(f"Overall: {strategy_total + description_total} total, {strategy_errors + description_errors} errors ({100*(strategy_errors + description_errors)/(strategy_total + description_total):.1f}% error rate)\\n\\n")
        
        f.write("DETAILED ERROR LOG:\\n")
        f.write("==================\\n\\n")
        for entry in validation_log:
            f.write(entry + "\\n")
    
    # Print summary
    print(f"Enhanced Model Validation Results")
    print(f"=================================")
    print(f"Strategy Tasks: {strategy_total} total, {strategy_errors} errors ({100*strategy_errors/strategy_total:.1f}% error rate)")
    print(f"Description Tasks: {description_total} total, {description_errors} errors ({100*description_errors/description_total:.1f}% error rate)")
    print(f"Overall: {strategy_total + description_total} total, {strategy_errors + description_errors} errors ({100*(strategy_errors + description_errors)/(strategy_total + description_total):.1f}% error rate)")
    
    return {
        'strategy_errors': strategy_errors,
        'strategy_total': strategy_total, 
        'description_errors': description_errors,
        'description_total': description_total
    }

if __name__ == "__main__":
    validate_enhanced_model()
