import argparse
import subprocess
import sys
import os

# map each plan to its max_steps (must match train script)
PLAN_MAX_STEPS = {
    0: 100,
    1: 2000,
    2: 10000,
    3: 3600,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=int,
        choices=[0, 1, 2, 3],
        required=True,
        help="Which Plan to run: 0 = smoke, 1 = fine-tune, 2 = full scale, 3 = 3600 steps",
    )
    args = parser.parse_args()
    plan = args.plan

    # use the same Python interpreter that launched this script
    python_exe = sys.executable

    data_file = "data/bethpage_black/basics.jsonl"
    output_base = "outputs/bethpage-lora"

    # 1. TRAINING STEP
    train_cmd = [
        python_exe,
        "train_lora_bethpage_strat.py",
        "--plan", str(plan),
        "--data-file", data_file,
    ]
    # only plans 1 & 2 resume from previous checkpoint
    if plan in (1, 2):
        train_cmd.append("--resume")

    print("→ Running training:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    # 2. INFERENCE STEP
    max_steps = PLAN_MAX_STEPS[plan]
    ckpt_dir = os.path.join(output_base, f"checkpoint-{max_steps}")

    infer_cmd = [
        python_exe,
        "fix_and_infer_lora_v3.py",
        "--adapter_dir", ckpt_dir,
        "--jsonl", data_file,
        "--model_name", "gpt2",
        "--max_tokens", "60",
        "--output", "outputs/inference.jsonl",
    ]

    print("→ Running inference:", " ".join(infer_cmd))
    subprocess.run(infer_cmd, check=True)


if __name__ == "__main__":
    main()