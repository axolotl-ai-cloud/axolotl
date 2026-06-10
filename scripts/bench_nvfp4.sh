#!/usr/bin/env bash
# Reusable NVFP4 training benchmark that always lands a TRUSTWORTHY number.
#
# Three traps this script exists to avoid:
#   1. `axolotl train` defaults to the `accelerate` launcher, which spawns a NEW
#      python (often base conda, no MSLK) -> the FP4 quant silently uses the slow
#      torchao path and numbers are meaningless. We force `--launcher python`.
#   2. tqdm's instantaneous `it/s` is a noisy EMA that updates several times per
#      step -> medians of it wobble 5-20% and can't resolve close calls. We instead
#      take MARGINAL step time from `train_runtime` at two run lengths
#      (wall_N2 - wall_N1)/(N2 - N1), which cancels the one-time compile/warmup cost
#      and is measured by the trainer, not scraped from a progress bar.
#   3. A co-running job (another bench, a llama.cpp server) steals SMs/bandwidth and
#      silently inflates timings. We REFUSE to measure unless the target GPU is
#      exclusively idle (<2.5 GiB used) before each run, and abort if it isn't.
# Also: a fast-but-diverging run is worse than useless -> we flag DIVERGED
# for non-finite loss/grad_norm or loss > 5.
#
# Usage:
#   scripts/bench_nvfp4.sh [--gpu N] [--steps N2] [--short N1] [--venv PATH] cfg.yaml ...
#   N2 (default 140) long run, N1 (default 40) short run; marginal = (wall2-wall1)/(N2-N1).
set -euo pipefail

GPU=6                # 5090 (sm_120). NEVER the RTX 6000 Blackwell (idx 0,3) for timing.
N2=140
N1=40
VENV="${VENV:-/home/rgilbreth/Documents/GitHub/LLM-Tools/Build-Venv/_venvs/axolotl_nvfp4}"
CFGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --steps) N2="$2"; shift 2;;
    --short) N1="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    *) CFGS+=("$1"); shift;;
  esac
done
[[ ${#CFGS[@]} -gt 0 ]] || { echo "usage: $0 [--gpu N] [--steps N2] cfg.yaml ..."; exit 2; }

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src"
PY="$VENV/bin/python"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$SRC"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gpu_mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
           | sed -n "$((GPU+1))p"; }
require_idle(){  # abort rather than time under contention
  local m; m=$(gpu_mem)
  if [[ -z "$m" || "$m" -gt 2500 ]]; then
    echo "[ABORT] gpu $GPU not exclusively idle (${m:-?} MiB used) before $1 — contention would taint timing."
    exit 3
  fi
}

"$PY" -c "import mslk" 2>/dev/null && echo "[ok] MSLK present ($("$PY" -c 'import torch;print(torch.__version__)'))" \
  || echo "[WARN] no MSLK in $VENV — FP4 path unrepresentative."
echo "[ok] launcher=python gpu=$GPU(PCI_BUS_ID) N1=$N1 N2=$N2 src=$SRC"

LOGDIR="$(mktemp -d /tmp/bench_nvfp4.XXXXXX)"
runwall(){ # $1=cfg $2=steps -> echoes train_runtime (or empty on fail); writes $3 log
  local cfg="$1" steps="$2" log="$3"
  sed "s/^max_steps:.*/max_steps: $steps/" "$cfg" > "$LOGDIR/run.yaml"
  # if the config had no max_steps line, sed leaves it absent -> the run would do
  # a FULL EPOCH and the marginal math is wrong. Force it.
  grep -q "^max_steps:" "$LOGDIR/run.yaml" || echo "max_steps: $steps" >> "$LOGDIR/run.yaml"
  "$PY" -m axolotl.cli.main train "$LOGDIR/run.yaml" --launcher python > "$log" 2>&1 || true
  grep -oE "train_runtime': '([0-9.]+)'" "$log" | grep -oE "[0-9.]+" | tail -1 || true
}

printf '\n%-26s %12s %10s %10s %9s\n' CONFIG "s/step(marg)" "warmup s" "loss" "act GiB"
for cfg in "${CFGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  # Per-config compile warm-up: each config compiles its OWN inductor/triton graph.
  # Without this the short run compiles cold and the long run reuses the warm disk
  # cache -> the marginal subtraction is corrupted (can go negative). Warm THIS
  # config first (throwaway) so both measured runs hit the warm cache.
  require_idle "$name:warm"
  runwall "$cfg" 8 "$LOGDIR/$name.warm.log" >/dev/null || true  # OOM here must not kill the driver (set -e)
  require_idle "$name:short"
  w1=$(runwall "$cfg" "$N1" "$LOGDIR/$name.short.log")
  require_idle "$name:long"
  w2=$(runwall "$cfg" "$N2" "$LOGDIR/$name.long.log")
"$PY" - "$name" "$LOGDIR/$name.long.log" "${w1:-0}" "${w2:-0}" "$N1" "$N2" <<'PY'
import math, re, sys
name, log, w1, w2, n1, n2 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
s = open(log).read()

def floats_for(field):
    vals = []
    for raw in re.findall(rf"'{field}': '([^']+)'", s):
        try:
            vals.append(float(raw))
        except ValueError:
            pass
    return vals

losses = floats_for("loss")
grad_norms = floats_for("grad_norm")
act = [float(x) for x in re.findall(r"max_active \(GiB\)': '([0-9.]+)'", s)]
failed = ("OutOfMemoryError" in s or "Traceback (most recent call last)" in s) and "'train_loss'" not in s
bad_loss = any((not math.isfinite(l)) or l > 5.0 for l in losses) if losses else False
bad_grad = any(not math.isfinite(g) for g in grad_norms) if grad_norms else False
diverged = bad_loss or bad_grad
if failed or w1 == 0 or w2 == 0:
    err = next((ln.strip() for ln in s.splitlines() if "OutOfMemory" in ln), "run failed")
    print(f"{name:<26} {'FAILED':>12}  {err[:50]}")
elif diverged:
    loss_msg = f"loss up to {max(losses):.1f}" if losses else "no loss"
    grad_msg = "non-finite grad_norm" if bad_grad else "finite grad_norm"
    print(f"{name:<26} {'DIVERGED':>12}  ({loss_msg}; {grad_msg})")
else:
    marg = (w2 - w1) / (n2 - n1)
    warm = w1 - n1 * marg
    print(f"{name:<26} {marg:>12.4f} {warm:>10.1f} {losses[-1] if losses else 0:>10.3f} {max(act) if act else 0:>9.2f}")
PY
done
echo; echo "logs: $LOGDIR  (method: marginal train_runtime delta, exclusive-GPU guarded)"
