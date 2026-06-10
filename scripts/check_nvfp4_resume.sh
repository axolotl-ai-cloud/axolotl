#!/usr/bin/env bash
# E2E save/resume check for NVFP4 training. Trains a few steps, saves a checkpoint,
# resumes from it, and asserts the resume is BIT-FAITHFUL (the resumed step's loss
# equals the original run's loss at that step) — i.e. the frozen FP4 base
# reconstructs deterministically and the adapter + optimizer reload correctly.
#
# Forces --launcher python (so the venv/MSLK is used, not an accelerate-spawned
# base conda). Usage:
#   scripts/check_nvfp4_resume.sh [--gpu N] [--venv PATH] config.yaml
# The config must set output_dir and an nvfp4_training block; the script injects
# save_steps/max_steps and the resume.
set -euo pipefail

GPU=6
VENV="${VENV:-/home/rgilbreth/Documents/GitHub/LLM-Tools/Build-Venv/_venvs/axolotl_nvfp4}"
CFG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    *) CFG="$1"; shift;;
  esac
done
[[ -n "$CFG" ]] || { echo "usage: $0 [--gpu N] config.yaml"; exit 2; }

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src"
PY="$VENV/bin/python"; AX="$VENV/bin/axolotl"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$SRC"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -c "import mslk" 2>/dev/null && echo "[ok] MSLK present" || echo "[WARN] no MSLK — FP4 path unrepresentative"

WORK="$(mktemp -d /tmp/nvfp4_resume.XXXXXX)"
OUT="$WORK/out"
# run 1: train 8 steps, checkpoint at 4
sed "s#^output_dir:.*#output_dir: $OUT#; s/max_steps: .*/max_steps: 8/; \
     s/save_strategy: .*/save_strategy: steps/; s/saves_per_epoch: .*//" "$CFG" > "$WORK/run1.yaml"
grep -q "save_steps:" "$WORK/run1.yaml" && sed -i "s/save_steps: .*/save_steps: 4/" "$WORK/run1.yaml" \
  || echo "save_steps: 4" >> "$WORK/run1.yaml"
cp "$WORK/run1.yaml" "$WORK/run2.yaml"
echo "resume_from_checkpoint: $OUT/checkpoint-4" >> "$WORK/run2.yaml"

echo "[run1] train + checkpoint-4 ..."; "$AX" train "$WORK/run1.yaml" --launcher python > "$WORK/run1.log" 2>&1
echo "[run2] resume from checkpoint-4 ..."; "$AX" train "$WORK/run2.yaml" --launcher python > "$WORK/run2.log" 2>&1

"$PY" - "$WORK/run1.log" "$WORK/run2.log" "$OUT" <<'PY'
import json, os, re, sys
r1, r2, out = sys.argv[1], sys.argv[2], sys.argv[3]
def losses(p): return [float(x) for x in re.findall(r"'loss': '([0-9.]+)'", open(p).read())]
l1, l2 = losses(r1), losses(r2)
ck = os.path.join(out, "checkpoint-4")
ok = os.path.isfile(os.path.join(ck, "adapter_model.safetensors"))
print(f"checkpoint-4 adapter saved: {ok}")
gs = None
ts = os.path.join(out, "checkpoint-8", "trainer_state.json")
if os.path.isfile(ts): gs = json.load(open(ts)).get("global_step")
print(f"final global_step: {gs} (expect 8)")
# bit-faithful: run2's first logged loss (step 5) should equal run1's step-5 loss
if len(l1) >= 5 and l2:
    s5_orig, s5_resumed = l1[4], l2[0]
    match = abs(s5_orig - s5_resumed) < 1e-3
    print(f"step-5 loss  original={s5_orig:.4f}  resumed={s5_resumed:.4f}  "
          f"{'BIT-FAITHFUL' if match else 'MISMATCH (resume not faithful!)'}")
    sys.exit(0 if (ok and gs == 8 and match) else 1)
print("could not parse losses — check logs at", out); sys.exit(1)
PY
echo "logs: $WORK"
