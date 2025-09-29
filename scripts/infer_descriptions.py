#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import tempfile
from typing import List, Optional

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


def prepare_adapter_folder(adapter_dir: str) -> str:
    """
    Create a temp folder with:
      - adapter_config.json (inference_mode=True)
      - adapter_model.safetensors (prefix-patched if needed)
    Returns path to that temp folder.
    """
    tmpdir = tempfile.mkdtemp(prefix="fixed_lora_desc_")
    os.makedirs(tmpdir, exist_ok=True)

    cfg_src = os.path.join(adapter_dir, "adapter_config.json")
    st_src = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not (os.path.isfile(cfg_src) and os.path.isfile(st_src)):
        raise FileNotFoundError(f"adapter files not found in {adapter_dir}")

    # Copy config and enforce inference_mode
    cfg_dst = os.path.join(tmpdir, "adapter_config.json")
    shutil.copy(cfg_src, cfg_dst)
    peft_cfg = PeftConfig.from_pretrained(tmpdir)
    peft_cfg.inference_mode = True
    peft_cfg.save_pretrained(tmpdir)

    # Load & re-prefix safetensors if needed
    sd = load_file(st_src, device="cpu")
    prefix = "base_model.model."
    needs_prefix = not next(iter(sd)).startswith(prefix)
    if needs_prefix:
        fixed = {prefix + k: v for k, v in sd.items()}
    else:
        fixed = sd
    save_file(fixed, os.path.join(tmpdir, "adapter_model.safetensors"))
    return tmpdir


def parse_args():
    p = argparse.ArgumentParser(description="Run description_synthesis inference with a LoRA adapter")
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--jsonl", required=True, help="JSONL with prompts (will skip non-description_synthesis)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--model_name", default="gpt2-medium")
    p.add_argument("--max_tokens", type=int, default=120)
    p.add_argument("--sample", action="store_true", help="Use sampling instead of greedy")
    return p.parse_args()


def run(adapter_dir: str, jsonl_path: str, out_path: str, model_name: str = "gpt2-medium", max_tokens: int = 120, sample: bool = False):
    fixed_adapter = prepare_adapter_folder(adapter_dir)
    print("Using adapter folder:", fixed_adapter)

    device = torch.device("cpu")  # safer on Windows/consumer GPUs
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base, fixed_adapter, local_files_only=True)
    model.to(device).eval()

    def gen_one(prompt: str) -> str:
        # Add a minimal guard against rambling
        stop_hint = "\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs = dict(
            max_new_tokens=max_tokens,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=sample,
            temperature=0.8 if sample else None,
            top_p=0.9 if sample else None,
        )
        outputs = model.generate(**inputs, **{k: v for k, v in kwargs.items() if v is not None})
        gen_ids = outputs[0, inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Trim at double-newline if present
        parts = re.split(r"\n\n+", text)
        return parts[0].strip()

    # Read JSONL and filter to description tasks
    kept = []
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if str(obj.get("task_type", "")).strip() != "description_synthesis":
                continue
            kept.append(obj)

    outputs = []
    for obj in kept:
        prompt = obj["prompt"]
        completion = gen_one(prompt)
        rec = {**obj, "raw_completion": completion, "completion": completion}
        outputs.append(rec)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in outputs:
            out.write(json.dumps(rec) + "\n")

    # Simple stats
    lengths = [len(r["completion"]) for r in outputs]
    avg = sum(lengths) / len(lengths) if lengths else 0
    any_strategy = any("Strategy:" in r["completion"] for r in outputs)
    print({
        "wrote": len(outputs),
        "avg_chars": round(avg, 1),
        "contains_strategy_tag": any_strategy,
        "out": out_path,
    })


if __name__ == "__main__":
    args = parse_args()
    run(
        adapter_dir=args.adapter_dir,
        jsonl_path=args.jsonl,
        out_path=args.output,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        sample=args.sample,
    )
