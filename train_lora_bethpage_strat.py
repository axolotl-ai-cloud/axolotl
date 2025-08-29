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

# Define plan-specific hyperparameters
# TRAINING TIME ESTIMATES (based on observed 4.5 minutes per step):
# - debug (1 step): Pipeline connectivity test only (~5 minutes)
# - debug_training (20 steps): Sanity check - NOT meaningful validation (~90 minutes)
# - quick_test (7 steps): Basic training verification (~30 minutes) 
# - 1_hour (13 steps): Light training run (~1 hour)
# - 8_hour (107 steps): Full training run (~8 hours)
# - multitask (107 steps): Multi-task learning (~8 hours)
PLAN_CONFIGS = {
    0: {"name": "debug", "max_steps": 1, "save": False, "save_steps": 1},
    1: {"name": "debug_training", "max_steps": 20, "save": True, "save_steps": 20},  # SANITY CHECK ONLY
    2: {"name": "quick_test", "max_steps": 7, "save": True, "save_steps": 7},       # ~30 minutes
    3: {"name": "1_hour", "max_steps": 13, "save": True, "save_steps": 13},        # ~1 hour  
    4: {"name": "8_hour", "max_steps": 107, "save": True, "save_steps": 107},      # ~8 hours
    5: {"name": "8_hour_enhanced", "max_steps": 107, "save": True, "save_steps": 53},  # ~8 hours (save every ~4 hours)
    6: {"name": "multitask", "max_steps": 107, "save": True, "save_steps": 53},   # ~8 hours MULTI-TASK LEARNING
    7: {"name": "fixed_multitask", "max_steps": 60, "save": True, "save_steps": 30},  # ~4.5 hours FIXED DATASET
}



def train_model(plan, data_file="data/bethpage_black/train_multitask_fixed.jsonl", resume=False, log_path=None):
    cfg = PLAN_CONFIGS[plan]
    max_steps = cfg["max_steps"]
    save_steps = cfg["save_steps"]
    do_save = cfg["save"]
    mode_name = cfg["name"]

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

    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
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

    model = AutoModelForCausalLM.from_pretrained("gpt2")
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

    # Only resume if trainer_state.json exists in checkpoint dir
    trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
    resume_ckpt = ckpt_dir if os.path.isfile(trainer_state_path) else None

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        max_steps=max_steps,
        save_steps=save_steps,
        save_total_limit=1,
        learning_rate=5e-4,
        logging_steps=1,
        fp16=False,
        resume_from_checkpoint=resume_ckpt,
        remove_unused_columns=False,
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


def run_pipeline(start_mode, data_file, resume, log_path):
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
            if "abnormal" in msg or "did not decrease" in msg:
                raise AssertionError(msg)

    # Always run debug and debug_training once
    result_debug = train_model(0, data_file, resume, log_path)
    success_debug, ckpt_debug, metrics_debug, grad_norm_debug, lr_debug, losses_debug = result_debug
    if not success_debug:
        print("Debug mode failed. Stopping.")
        return
    result_debug_training = train_model(1, data_file, resume, log_path)
    success_debug_training, ckpt_debug_training, metrics_debug_training, grad_norm_debug_training, lr_debug_training, losses_debug_training = result_debug_training
    verify_metrics(metrics_debug_training, grad_norm_debug_training, lr_debug_training, losses_debug_training, "debug_training")
    if not success_debug_training:
        print("Debug training failed. Stopping.")
        return
    if start_mode == "test":
        # Run quick inference test with 20-step model
        print("Running inference test with 20-step model...")
        import subprocess
        try:
            # Run inference
            cmd_infer = [
                sys.executable, "fix_and_infer_lora_v3.py",
                "--adapter_dir", "outputs/bethpage-lora/checkpoint-debug_training",
                "--jsonl", data_file,
                "--output", "outputs/bethpage-lora/inference_debug_training.jsonl"
            ]
            subprocess.run(cmd_infer, check=True)
            
            # Run validation (use flags)
            cmd_check = [
                sys.executable, "check_inference_strategies.py",
                "--inference-file", "outputs/bethpage-lora/inference_debug_training.jsonl",
                "--data-file", data_file,
                "--log-file", "outputs/bethpage-lora/inference_strategy_check_log.txt"
            ]
            subprocess.run(cmd_check, check=True)
            print("20-step inference test completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"20-step inference test failed: {e}")
        
        print("Test pipeline complete.")
        return
    if start_mode == "quick":
        result_quick = train_model(2, data_file, resume, log_path)
        success_quick, ckpt_quick, metrics_quick, grad_norm_quick, lr_quick, losses_quick = result_quick
        verify_metrics(metrics_quick, grad_norm_quick, lr_quick, losses_quick, "quick_test")
        if not success_quick:
            print("Quick test mode failed. Stopping.")
            return
        print("Quick test pipeline complete.")
        return
    if start_mode == "short":
        result_1hr = train_model(3, data_file, resume, log_path)
        success_1hr, ckpt_1hr, metrics_1hr, grad_norm_1hr, lr_1hr, losses_1hr = result_1hr
        verify_metrics(metrics_1hr, grad_norm_1hr, lr_1hr, losses_1hr, "1_hour")
        if not success_1hr:
            print("1_hour mode failed. Stopping.")
            return
        print("Short pipeline complete.")
        return
    if start_mode == "long":
        result_8hr = train_model(4, data_file, resume, log_path)
        success_8hr, ckpt_8hr, metrics_8hr, grad_norm_8hr, lr_8hr, losses_8hr = result_8hr
        verify_metrics(metrics_8hr, grad_norm_8hr, lr_8hr, losses_8hr, "8_hour")
        if not success_8hr:
            print("8_hour mode failed. Stopping.")
            return
        print("Long pipeline complete.")
        return
    if start_mode == "enhanced":
        result_enhanced = train_model(5, data_file, resume, log_path)
        success_enhanced, ckpt_enhanced, metrics_enhanced, grad_norm_enhanced, lr_enhanced, losses_enhanced = result_enhanced
        verify_metrics(metrics_enhanced, grad_norm_enhanced, lr_enhanced, losses_enhanced, "8_hour_enhanced")
        if not success_enhanced:
            print("8_hour_enhanced mode failed. Stopping.")
            return
        print("Enhanced pipeline complete.")
        return
    if start_mode == "multitask":
        result_multitask = train_model(6, data_file, resume, log_path)
        success_multitask, ckpt_multitask, metrics_multitask, grad_norm_multitask, lr_multitask, losses_multitask = result_multitask
        verify_metrics(metrics_multitask, grad_norm_multitask, lr_multitask, losses_multitask, "multitask")
        if not success_multitask:
            print("Multitask mode failed. Stopping.")
            return
        print("Multi-task pipeline complete. Model can now handle both strategy selection AND description synthesis!")
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=str,
        choices=["test", "quick", "short", "long", "enhanced", "multitask"],
        default="multitask",
        help="Pipeline start mode: test, quick, short, long, enhanced, multitask (default: multitask)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/bethpage_black/train_multitask_perfect.jsonl",
        help="Path to JSONL data file (default: data/bethpage_black/train_multitask_perfect.jsonl)"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="outputs/bethpage-lora/train_log.txt",
        help="Path to log file"
    )
    args = parser.parse_args()
    # Always resume from checkpoint by default
    run_pipeline(args.start, args.data_file, True, args.log_path)

if __name__ == "__main__":
    main()