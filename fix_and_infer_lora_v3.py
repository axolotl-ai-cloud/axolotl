#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile
import json
import re
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

# ─── Helpers ────────────────────────────────────────────────────────────────

def prepare_adapter_folder(adapter_dir: str) -> str:
    """
    Create a temp folder with:
      - adapter_config.json (inference_mode=True)
      - adapter_model.safetensors (prefix-patched if needed)
    Returns path to that temp folder.
    """
    tmpdir = tempfile.mkdtemp(prefix="fixed_lora_")
    os.makedirs(tmpdir, exist_ok=True)

    # Copy config and enforce inference_mode
    cfg_src = os.path.join(adapter_dir, "adapter_config.json")
    cfg_dst = os.path.join(tmpdir, "adapter_config.json")
    shutil.copy(cfg_src, cfg_dst)
    peft_cfg = PeftConfig.from_pretrained(tmpdir)
    peft_cfg.inference_mode = True
    peft_cfg.save_pretrained(tmpdir)

    # Load & re-prefix safetensors
    st_src = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = load_file(st_src, device="cpu")
    # detect if keys need prefix
    prefix = "base_model.model."
    needs_prefix = not next(iter(sd)).startswith(prefix)
    fixed = {}
    if needs_prefix:
        for k, v in sd.items():
            fixed[prefix + k] = v
    else:
        fixed = sd
    save_file(fixed, os.path.join(tmpdir, "adapter_model.safetensors"))

    return tmpdir


def check_bleed(completion: str, current_hole: int):
    """
    Look for any [Hole N] where N != current_hole.
    Returns (bleed_detected: bool, offending_hole: int|None).
    """
    markers = re.findall(r"\[Hole\s*(\d+)\]", completion)
    for m in markers:
        num = int(m)
        if num != current_hole:
            return True, num
    return False, None


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a LoRA adapter, run inference on JSONL prompts, emit JSONL output"
    )
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="Path to raw adapter folder (adapter_config.json + adapter_model.safetensors)"
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to JSONL file with hole prompts"
    )
    parser.add_argument(
        "--model_name",
        default="gpt2",
        help="Base model name or path (default: gpt2)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=60,
        help="Max new tokens to generate per prompt"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the JSONL inference results"
    )
    return parser.parse_args()


# ─── Main ───────────────────────────────────────────────────────────────────

def run_inference(adapter_dir, jsonl_file, output_file, model_name="gpt2", max_tokens=60):
    fixed_adapter = prepare_adapter_folder(adapter_dir)
    print("Using adapter folder:", fixed_adapter)

    # Force CPU to avoid GPU compatibility issues with RTX 5060 Ti
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base, fixed_adapter, local_files_only=True)
    model.to(device).eval()

    hole_re = re.compile(r"\[Hole\s*(\d+)\]")
    std_re = re.compile(r"^\s*Strategy:\s*(\d{2,4})\s*yards\s*$", re.IGNORECASE)
    records = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompt = entry["prompt"]
            # Append format instruction so the model emits the standardized line
            prompt_plus = prompt + " Answer only in this format: Strategy: <N> yards"
            m = hole_re.search(prompt)
            expected_hole = int(m.group(1)) if m else None
            inputs = tokenizer(prompt_plus, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            gen_ids = outputs[0, inputs["input_ids"].shape[-1]:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            bleed_flag, bleed_hole = False, None
            if expected_hole is not None:
                bleed_flag, bleed_hole = check_bleed(raw, expected_hole)
            clean = re.split(r"\[Hole\s*\d+\]", raw, maxsplit=1)[0].strip()
            # If the model produced extra text, try to extract the standardized line
            mstd = std_re.search(clean)
            if mstd:
                clean = f"Strategy: {mstd.group(1)} yards"
            else:
                # Fallback: try to extract any number and coerce
                mnum = re.search(r"(\d{2,4})\s*-?\s*yard", clean, flags=re.IGNORECASE)
                if mnum:
                    clean = f"Strategy: {int(mnum.group(1))} yards"
            rec = {
                **entry,
                "raw_completion": raw,
                "completion": clean,
                "bleed_flag": bleed_flag,
                "bleed_hole": bleed_hole,
            }
            records.append(rec)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec) + "\n")
    # Avoid Unicode symbols that may fail on Windows cp1252 consoles
    print(f"Wrote {len(records)} records to {output_file}")
    return records


def main():
    args = parse_args()
    run_inference(
        adapter_dir=args.adapter_dir,
        jsonl_file=args.jsonl,
        output_file=args.output,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()