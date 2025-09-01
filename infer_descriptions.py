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


def prepare_adapter_folder(adapter_dir: str) -> str:
    """Create a temp folder with a safe, inference-mode LoRA adapter.
    Returns the path to the temp folder.
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

    # Load and ensure the safetensors keys are prefixed as expected by PeftModel
    st_src = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = load_file(st_src, device="cpu")
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


def parse_args():
    p = argparse.ArgumentParser(description="Run description_synthesis inference with a LoRA adapter")
    p.add_argument("--adapter_dir", required=True, help="Path to trained LoRA checkpoint directory")
    p.add_argument("--jsonl", required=True, help="Input JSONL with description_synthesis prompts")
    p.add_argument("--output", required=True, help="Output JSONL path for generated completions")
    p.add_argument("--model_name", default="gpt2", help="Base model name or path (default: gpt2)")
    p.add_argument("--max_tokens", type=int, default=160, help="Max new tokens to generate per prompt")
    p.add_argument("--do_sample", action="store_true", help="Enable sampling (otherwise greedy)")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature when --do_sample is set")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus when --do_sample is set")
    return p.parse_args()


def clean_bleed(text: str) -> str:
    """Trim anything that looks like it bled into the next prompt block like "[Hole N]"."""
    parts = re.split(r"\[Hole\s*\d+\]", text, maxsplit=1)
    return parts[0].strip()


def run_inference(adapter_dir: str, jsonl_file: str, output_file: str, model_name: str, max_tokens: int, do_sample: bool, temperature: float, top_p: float):
    fixed_adapter = prepare_adapter_folder(adapter_dir)
    print("Using adapter folder:", fixed_adapter)

    # Use CPU for broad compatibility on Windows; dataset is small so this is fine
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base, fixed_adapter, local_files_only=True)
    model.to(device).eval()

    records = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            prompt = entry["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
                do_sample=do_sample,
            )
            if do_sample:
                gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
            outputs = model.generate(**inputs, **gen_kwargs)
            gen_ids = outputs[0, inputs["input_ids"].shape[-1]:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            clean = clean_bleed(raw)
            rec = {**entry, "raw_completion": raw, "completion": clean}
            records.append(rec)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
