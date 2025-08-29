#!/usr/bin/env python3
"""
Analyze token usage patterns in the training data to optimize context windows.

This will help us understand:
1. How many tokens are used by prompts vs completions
2. If we're wasting context window space
3. Optimal prompt_max and completion_max settings
4. Token distribution across different prompt types
"""

import json
import sys
from transformers import GPT2TokenizerFast
from collections import defaultdict
import statistics

def load_data(file_path):
    """Load and parse JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def analyze_tokens(data_file):
    """Analyze token usage in the dataset."""
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    data = load_data(data_file)
    print(f"Analyzing {len(data)} examples from {data_file}")
    
    # Current settings
    FORMAT_INSTR = " Answer: Strategy: <N> yards"
    prompt_max = 128
    completion_max = 16
    
    # Statistics collections
    prompt_lengths = []
    completion_lengths = []
    total_lengths = []
    format_instr_length = len(tokenizer(FORMAT_INSTR)["input_ids"])
    
    prompt_truncated = 0
    completion_truncated = 0
    
    # Analyze each example
    for i, example in enumerate(data):
        # Skip negative examples
        if example.get("is_correct") is False or example.get("use_for_training") is False:
            continue
            
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        
        # Create standard completion (Strategy: N yards format)
        expected_cutoff = example.get("expected_cutoff_yards")
        if expected_cutoff:
            std_completion = f"Strategy: {expected_cutoff} yards"
        else:
            std_completion = completion
            
        # Tokenize
        prompt_with_instr = prompt + FORMAT_INSTR
        prompt_tokens = tokenizer(prompt_with_instr)["input_ids"]
        completion_tokens = tokenizer(std_completion)["input_ids"]
        
        # Record lengths
        prompt_len = len(prompt_tokens)
        completion_len = len(completion_tokens)
        total_len = prompt_len + completion_len
        
        prompt_lengths.append(prompt_len)
        completion_lengths.append(completion_len)
        total_lengths.append(total_len)
        
        # Check truncation
        if prompt_len > prompt_max:
            prompt_truncated += 1
        if completion_len > completion_max:
            completion_truncated += 1
            
        # Show first few examples
        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"  Prompt: {prompt}")
            print(f"  Completion: {std_completion}")
            print(f"  Prompt tokens: {prompt_len} (max: {prompt_max})")
            print(f"  Completion tokens: {completion_len} (max: {completion_max})")
            print(f"  Total tokens: {total_len}")
    
    # Summary statistics
    print(f"\n=== TOKEN USAGE ANALYSIS ===")
    print(f"Total training examples: {len(prompt_lengths)}")
    print(f"Format instruction tokens: {format_instr_length}")
    
    print(f"\nPROMPT TOKENS:")
    print(f"  Min: {min(prompt_lengths)}")
    print(f"  Max: {max(prompt_lengths)}")
    print(f"  Mean: {statistics.mean(prompt_lengths):.1f}")
    print(f"  Median: {statistics.median(prompt_lengths):.1f}")
    print(f"  95th percentile: {sorted(prompt_lengths)[int(0.95 * len(prompt_lengths))]}")
    print(f"  Truncated: {prompt_truncated}/{len(prompt_lengths)} ({100*prompt_truncated/len(prompt_lengths):.1f}%)")
    
    print(f"\nCOMPLETION TOKENS:")
    print(f"  Min: {min(completion_lengths)}")
    print(f"  Max: {max(completion_lengths)}")
    print(f"  Mean: {statistics.mean(completion_lengths):.1f}")
    print(f"  Median: {statistics.median(completion_lengths):.1f}")
    print(f"  95th percentile: {sorted(completion_lengths)[int(0.95 * len(completion_lengths))]}")
    print(f"  Truncated: {completion_truncated}/{len(completion_lengths)} ({100*completion_truncated/len(completion_lengths):.1f}%)")
    
    print(f"\nTOTAL TOKENS:")
    print(f"  Min: {min(total_lengths)}")
    print(f"  Max: {max(total_lengths)}")
    print(f"  Mean: {statistics.mean(total_lengths):.1f}")
    print(f"  Current limit: {prompt_max + completion_max}")
    
    # Recommendations
    print(f"\n=== OPTIMIZATION RECOMMENDATIONS ===")
    
    # Optimal prompt_max (covers 98% of examples)
    optimal_prompt_max = sorted(prompt_lengths)[int(0.98 * len(prompt_lengths))]
    print(f"Optimal prompt_max: {optimal_prompt_max} (covers 98% of examples)")
    
    # Optimal completion_max (covers all examples + buffer)
    optimal_completion_max = max(completion_lengths) + 2
    print(f"Optimal completion_max: {optimal_completion_max} (covers all + buffer)")
    
    # Context efficiency
    current_total = prompt_max + completion_max
    optimal_total = optimal_prompt_max + optimal_completion_max
    efficiency_gain = current_total - optimal_total
    
    print(f"\nCONTEXT EFFICIENCY:")
    print(f"  Current total: {current_total} tokens")
    print(f"  Optimal total: {optimal_total} tokens")
    print(f"  Efficiency gain: {efficiency_gain} tokens ({100*efficiency_gain/current_total:.1f}%)")
    
    # Check for problematic patterns
    long_prompts = [i for i, length in enumerate(prompt_lengths) if length > prompt_max]
    if long_prompts:
        print(f"\n=== TRUNCATED PROMPTS (first 3) ===")
        for i in long_prompts[:3]:
            example = [ex for ex in data if ex.get("use_for_training") != False][i]
            prompt = example.get("prompt", "")
            print(f"Example {i+1}: {len(tokenizer(prompt + FORMAT_INSTR)['input_ids'])} tokens")
            print(f"  Prompt: {prompt}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_token_usage.py <data_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    analyze_tokens(data_file)

if __name__ == "__main__":
    main()
