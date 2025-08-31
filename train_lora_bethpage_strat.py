import os
import argparse
import sys

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
os.environ.setdefault("PYTORCH_SDP_KERNEL", "math")
# Quiet HF datasets progress bars to reduce terminal noise on Windows piping
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

# Prefer safe attention kernels to avoid architecture-specific crashes on newer GPUs
def _configure_sdp_kernels(force_cpu: bool):
    try:
        if not force_cpu and torch.cuda.is_available():
            from torch.backends.cuda import sdp_kernel
            # Disable Flash and Mem-efficient to avoid unsupported kernels; use math fallback
            sdp_kernel.enable_flash(False)
            sdp_kernel.enable_mem_efficient(False)
            sdp_kernel.enable_math(True)
    except Exception:
        # Best-effort; continue if backend flags are unavailable
        pass

# Define plan-specific hyperparameters (GPU-calibrated)
# Notes:
# - debug: 1 step with tiny model (wiring)
# - debug_training: 20 steps with tiny model (sanity + checkpoint)
# - 1_hour: ~2,560 steps on GPU
# - 8_hour: ~20,480 steps on GPU
PLAN_CONFIGS = {
    0: {"name": "debug", "max_steps": 1, "save": False, "save_steps": 1, "model_name": "sshleifer/tiny-gpt2"},
    1: {"name": "debug_training", "max_steps": 20, "save": True, "save_steps": 20, "model_name": "sshleifer/tiny-gpt2"},
    2: {"name": "1_hour", "max_steps": 2560, "save": True, "save_steps": 640, "model_name": "gpt2"},
    3: {"name": "8_hour", "max_steps": 20480, "save": True, "save_steps": 2560, "model_name": "gpt2"},
}



def train_model(
    plan,
    data_file="data/bethpage_black/train_multitask_case_fixed.jsonl",
    resume=False,
    log_path=None,
    force_cpu=False,
    max_steps_override: int | None = None,
    save_steps_override: int | None = None,
):
    # Configure SDP kernels before model init
    _configure_sdp_kernels(force_cpu)
    cfg = PLAN_CONFIGS[plan]
    max_steps = max_steps_override if max_steps_override is not None else cfg["max_steps"]
    save_steps = save_steps_override if save_steps_override is not None else cfg["save_steps"]
    do_save = cfg["save"]
    mode_name = cfg["name"]
    model_name = cfg.get("model_name", "gpt2")

    output_base = "outputs/bethpage-lora"
    ckpt_dir = os.path.join(output_base, f"checkpoint-{mode_name}")
    os.makedirs(ckpt_dir, exist_ok=True)

    ds_raw = load_dataset("json", data_files=data_file)["train"]

    # Filter out known negative examples (is_correct == False or use_for_training == False)
    def _filter_training_examples(example):
        is_correct = example.get("is_correct")
        use_for_training = example.get("use_for_training")
        
        # Exclude if explicitly marked as incorrect or not for training
        if is_correct is False or use_for_training is False:
            return False
        return True
    ds = ds_raw.filter(_filter_training_examples)
    if len(ds) == 0:
        print("Warning: filter removed all examples; using unfiltered dataset.")
        ds = ds_raw

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Utilities to lock output format: "Strategy: <N> yards"
    import re
    def extract_drive_from_prompt(prompt: str):
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
                except Exception:
                    pass
        return None

    def choose_cutoff(strategies, drive):
        # strategies: list of {cutoff_distance: int, ...}
        if not strategies:
            return None
        cutoffs = sorted([s.get("cutoff_distance") for s in strategies if isinstance(s.get("cutoff_distance"), int)])
        if not cutoffs:
            return None
        if drive is None:
            # If only one strategy, use it; else default to max (more aggressive)
            return cutoffs[0] if len(cutoffs) == 1 else cutoffs[-1]
        eligible = [c for c in cutoffs if c <= drive]
        return max(eligible) if eligible else cutoffs[0]

    def extract_cutoff_from_completion(text: str):
        m = re.search(r"(\d{2,4})\s*-?\s*yard", text, flags=re.IGNORECASE)
        return int(m.group(1)) if m else None

    def build_standard_completion(example):
        # Handle multi-task examples with different completion formats
        task_type = example.get("task_type")
        
        if task_type == "description_synthesis":
            # For description synthesis, use the completion as-is (it's already the synthesized analysis)
            return example.get("completion", "Strategic analysis required.")
        
        # For strategy selection tasks, use the existing logic
        # Prefer explicit target from dataset when available
        exp = example.get("expected_cutoff_yards")
        if isinstance(exp, int):
            return f"Strategy: {exp} yards"
        # Determine target cutoff
        drive = extract_drive_from_prompt(example.get("prompt", ""))
        cutoff = choose_cutoff(example.get("tee_shot_strategies", []), drive)
        if cutoff is None:
            # Fallback: try to derive from original completion if present
            cutoff = extract_cutoff_from_completion(example.get("completion", ""))
        if cutoff is None:
            # Final fallback: 300
            cutoff = 300
        return f"Strategy: {cutoff} yards"

    FORMAT_INSTR = ""  # Remove format instruction for multi-task learning (task prefix handles this)

    def tokenize_fn(example):
        # Unified context windows for multi-task learning
        # Use the larger dimensions to accommodate both task types
        prompt_max = 200    # Can handle description synthesis prompts
        completion_max = 50  # Can handle description synthesis completions
        total_max = prompt_max + completion_max  # 250 tokens total
            
        prompt_text = example["prompt"] + FORMAT_INSTR
        std_completion = build_standard_completion(example)
        enc_prompt = tokenizer(
            prompt_text, truncation=True, max_length=prompt_max)
        enc_completion = tokenizer(
            std_completion, truncation=True, max_length=completion_max)
        input_ids = enc_prompt["input_ids"] + enc_completion["input_ids"]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(enc_prompt["input_ids"]) + enc_completion["input_ids"]
        
        # Pad to unified length for multi-task learning
        input_ids = input_ids[:total_max] + [tokenizer.pad_token_id] * max(0, total_max - len(input_ids))
        attention_mask = attention_mask[:total_max] + [0] * max(0, total_max - len(attention_mask))
        labels = labels[:total_max] + [-100] * max(0, total_max - len(labels))
        if all(l == -100 for l in labels):
            if len(enc_completion["input_ids"]) > 0:
                labels[-len(enc_completion["input_ids"])] = enc_completion["input_ids"][0]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Ensure tokenized columns are present and drop raw fields
    original_columns = ds.column_names
    ds = ds.map(tokenize_fn, remove_columns=original_columns)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,              # Increased from 8 for better capacity
        lora_alpha=32,     # Kept the same for stable learning
        lora_dropout=0.1,  # Increased from 0.05 for better generalization
        target_modules=["c_attn", "c_proj", "c_fc"],  # Target key attention/projection layers
    )
    model = get_peft_model(model, peft_config)

    # Decide device usage and log environment
    def _cuda_usable() -> bool:
        if force_cpu or not torch.cuda.is_available():
            return False
        try:
            x = torch.tensor([1.0], device="cuda") * 2
            torch.cuda.synchronize()
            return True
        except Exception:
            return False

    use_cuda = _cuda_usable()
    # Enrich device summary and surface likely-compatibility issues (e.g., sm_120 on CUDA 11.x wheels)
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    dev_cap = None
    try:
        if torch.cuda.is_available():
            dev_cap = torch.cuda.get_device_capability(0)  # (major, minor)
    except Exception:
        dev_cap = None
    cuda_ver = getattr(torch.version, "cuda", None)
    device_summary = {
        "use_cuda": use_cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_ver,
        "device_name": dev_name,
        "device_capability": f"sm_{dev_cap[0]}{dev_cap[1]}" if dev_cap else None,
    }
    print(f"Device: {device_summary}")
    if log_path:
        try:
            with open(log_path, "a") as logf:
                logf.write(f"Device: {device_summary}\n")
        except Exception:
            pass

    # Emit a human-friendly warning if the GPU is newer than the installed CUDA runtime
    try:
        if torch.cuda.is_available() and dev_cap is not None and isinstance(cuda_ver, str):
            major_minor = dev_cap[0] * 10 + dev_cap[1]
            # Heuristic: cards with sm_100+ typically require CUDA 12.x wheels; warn if running CUDA 11.x
            if major_minor >= 100 and cuda_ver.startswith("11"):
                warn_msg = (
                    f"Warning: {dev_name} ({device_summary['device_capability']}) is likely unsupported by CUDA {cuda_ver} wheels. "
                    "Install a CUDA 12.x PyTorch build (e.g., cu124/cu126) to enable optimized kernels and normal speed."
                )
                print(warn_msg)
                if log_path:
                    try:
                        with open(log_path, "a") as logf:
                            logf.write(warn_msg + "\n")
                    except Exception:
                        pass
    except Exception:
        pass

    # Adjust micro-batch if we're on CPU to keep steps shorter
    per_device_bs = 8 if use_cuda else 2
    grad_accum = 8 if use_cuda else 2

    # Only resume if trainer_state.json exists in checkpoint dir
    trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
    resume_ckpt = ckpt_dir if os.path.isfile(trainer_state_path) else None

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        save_steps=save_steps,
        save_total_limit=1,
        learning_rate=5e-4,
        logging_steps=1,
        fp16=False,
        resume_from_checkpoint=resume_ckpt,
        remove_unused_columns=False,
        # Disable external trackers (e.g., trackio/wandb) to avoid gradio/pydub imports on Py 3.13
        report_to="none",
        # Force CPU when requested or when GPU is incompatible
        no_cuda=not use_cuda,
        # Keep the terminal clean when piping output
        disable_tqdm=True,
    )

    # Custom callback to record per-step loss
    from transformers import TrainerCallback
    class LossRecorder(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.losses = []
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.losses.append(logs["loss"])
        def on_init_end(self, args, state, control, **kwargs):
            pass

    loss_recorder = LossRecorder()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        callbacks=[loss_recorder],
    )

    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    metrics = train_result.metrics
    # Try to get grad_norm and learning_rate from trainer state
    grad_norm = getattr(trainer.state, "grad_norm", None)
    learning_rate = getattr(trainer.state, "learning_rate", None)
    per_step_losses = loss_recorder.losses
    success = metrics.get("train_loss") is not None
    if do_save:
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    if log_path:
        with open(log_path, "a") as logf:
            logf.write(f"{mode_name} success: {success}, train_loss: {metrics.get('train_loss', 'N/A')}, grad_norm: {grad_norm}, learning_rate: {learning_rate}, per_step_losses: {per_step_losses}\n")
    print(f"{mode_name} success: {success}, train_loss: {metrics.get('train_loss', 'N/A')}, grad_norm: {grad_norm}, learning_rate: {learning_rate}, per_step_losses: {per_step_losses}")
    return success, ckpt_dir, metrics, grad_norm, learning_rate, per_step_losses


def run_pipeline(start_mode, data_file, resume, log_path, force_cpu=False, max_steps_override: int | None = None, save_steps_override: int | None = None):
    # Clean log at start
    if os.path.exists(log_path):
        os.remove(log_path)

    def verify_metrics(metrics, grad_norm, learning_rate, per_step_losses, mode_name):
        # Basic checks
        train_loss = metrics.get("train_loss")
        results = []
        if train_loss is None or not (0 < train_loss < 100):
            msg = f"{mode_name}: train_loss abnormal: {train_loss}"
            results.append(msg)
        else:
            msg = f"{mode_name}: train_loss OK: {train_loss}"
            results.append(msg)
        if grad_norm is not None:
            if not (0 < grad_norm < 100):
                msg = f"{mode_name}: grad_norm abnormal: {grad_norm}"
                results.append(msg)
            else:
                msg = f"{mode_name}: grad_norm OK: {grad_norm}"
                results.append(msg)
        if learning_rate is not None:
            if not (0 < learning_rate < 1):
                msg = f"{mode_name}: learning_rate abnormal: {learning_rate}"
                results.append(msg)
            else:
                msg = f"{mode_name}: learning_rate OK: {learning_rate}"
                results.append(msg)
        # Loss should decrease over time (allow some noise)
        if per_step_losses and len(per_step_losses) > 2:
            first = per_step_losses[0]
            last = per_step_losses[-1]
            if last >= first:
                # In very short runs like quick_test, allow flat/slightly increasing loss without failing
                if mode_name == "quick_test":
                    msg = f"{mode_name}: loss did not decrease (first: {first}, last: {last}) [warning only]"
                    results.append(msg)
                else:
                    msg = f"{mode_name}: loss did not decrease (first: {first}, last: {last})"
                    results.append(msg)
            else:
                msg = f"{mode_name}: loss decreased (first: {first}, last: {last})"
                results.append(msg)
        # Print and log all results
        for msg in results:
            print(msg)
        if log_path:
            with open(log_path, "a") as logf:
                for msg in results:
                    logf.write(msg + "\n")
        # If any abnormal, raise AssertionError
        for msg in results:
            if "abnormal" in msg or ("did not decrease" in msg and mode_name != "quick_test"):
                raise AssertionError(msg)

    # Always run debug and debug_training once
    result_debug = train_model(0, data_file, resume, log_path, force_cpu)
    success_debug, ckpt_debug, metrics_debug, grad_norm_debug, lr_debug, losses_debug = result_debug
    if not success_debug:
        print("Debug mode failed. Stopping.")
        return
    result_debug_training = train_model(1, data_file, resume, log_path, force_cpu)
    success_debug_training, ckpt_debug_training, metrics_debug_training, grad_norm_debug_training, lr_debug_training, losses_debug_training = result_debug_training
    verify_metrics(metrics_debug_training, grad_norm_debug_training, lr_debug_training, losses_debug_training, "debug_training")
    if not success_debug_training:
        print("Debug training failed. Stopping.")
        return
    if start_mode == "short":
        result_1hr = train_model(2, data_file, resume, log_path, force_cpu, max_steps_override, save_steps_override)
        success_1hr, ckpt_1hr, metrics_1hr, grad_norm_1hr, lr_1hr, losses_1hr = result_1hr
        verify_metrics(metrics_1hr, grad_norm_1hr, lr_1hr, losses_1hr, "1_hour")
        if not success_1hr:
            print("1_hour mode failed. Stopping.")
            return
        print("Short pipeline complete.")
        return
    if start_mode == "long":
        result_8hr = train_model(3, data_file, resume, log_path, force_cpu, max_steps_override, save_steps_override)
        success_8hr, ckpt_8hr, metrics_8hr, grad_norm_8hr, lr_8hr, losses_8hr = result_8hr
        verify_metrics(metrics_8hr, grad_norm_8hr, lr_8hr, losses_8hr, "8_hour")
        if not success_8hr:
            print("8_hour mode failed. Stopping.")
            return
        print("Long pipeline complete.")
        return
    # If an unknown mode is passed, default to short
    result_1hr = train_model(2, data_file, resume, log_path, force_cpu, max_steps_override, save_steps_override)
    success_1hr, ckpt_1hr, metrics_1hr, grad_norm_1hr, lr_1hr, losses_1hr = result_1hr
    verify_metrics(metrics_1hr, grad_norm_1hr, lr_1hr, losses_1hr, "1_hour")
    print("Short pipeline complete.")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=str,
    choices=["short", "long"],
    default="short",
    help="Pipeline start mode: short (~1 hour), long (~8 hours). Debug phases always run first."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/bethpage_black/train_multitask_case_fixed.jsonl",
        help="Path to JSONL data file"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="outputs/bethpage-lora/train_log.txt",
        help="Path to log file"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (sets no_cuda=True). Use this on unsupported GPUs or to avoid CUDA kernel issues."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps for the selected main phase only (does not affect debug stages)."
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Override save_steps for the selected main phase only."
    )
    args = parser.parse_args()
    # Always resume from checkpoint by default
    run_pipeline(args.start, args.data_file, True, args.log_path, args.cpu, args.max_steps, args.save_steps)

if __name__ == "__main__":
    main()