#!/usr/bin/env bash
# Phase 2 driver: quality runs + throughput/memory probes, sequentially on one GPU.
# All runs log to wandb project sonic_nvfp4; raw logs land in $LOGDIR.
#
#   nohup bash benchmarks/sonicmoe_nvfp4/run_phase2.sh > /workspace/phase2.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export PATH=/workspace/axolotl-venv/bin:$PATH
LOGDIR=${LOGDIR:-/workspace/phase2_logs}
mkdir -p "$LOGDIR"
CFG=benchmarks/sonicmoe_nvfp4/train_tiny_lora.yaml
CFG_SMOE=benchmarks/sonicmoe_nvfp4/train_tiny_lora_scattermoe.yaml

echo "commit: $(git rev-parse HEAD)"

run() {
  local name=$1
  shift
  echo "=== START $name: $(date -u +%H:%M:%S) ==="
  "$@" >"$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo "=== END $name rc=$rc $(date -u +%H:%M:%S) ==="
  grep -o "{'train_runtime[^}]*}" "$LOGDIR/$name.log" | tail -1
  grep -o "{'loss[^}]*}" "$LOGDIR/$name.log" | tail -1 | cut -c1-160
}

# --- quality: 200 steps @ seq 2048 with a shared held-out eval split ---
QUAL="--max-steps 200 --val-set-size 0.03 --eval-steps 50"

run lora200_fp4_cute axolotl train "$CFG" $QUAL \
  --wandb-name lora200_qwen3-30b_fp4_cute \
  --output-dir ./outputs/p2-lora200-fp4

AXOLOTL_SONICMOE_NVFP4_BACKEND=dequant \
  run lora200_dequant axolotl train "$CFG" $QUAL \
  --wandb-name lora200_qwen3-30b_dequant \
  --output-dir ./outputs/p2-lora200-dq

# --- throughput/memory probes: 12 steps, sweep context length and tokens/step ---
bench() {
  local backend=$1 seq=$2 mbs=$3
  local name="bench_${backend}_seq${seq}_mbs${mbs}"
  local cfg=$CFG
  [ "$backend" = marlin ] && cfg=$CFG_SMOE
  if [ "$backend" = dequant ]; then
    AXOLOTL_SONICMOE_NVFP4_BACKEND=dequant \
      run "$name" axolotl train "$cfg" \
      --max-steps 12 --sequence-len "$seq" --micro-batch-size "$mbs" \
      --wandb-name "$name" \
      --output-dir "./outputs/p2-$name"
  else
    run "$name" axolotl train "$cfg" \
      --max-steps 12 --sequence-len "$seq" --micro-batch-size "$mbs" \
      --wandb-name "$name" \
      --output-dir "./outputs/p2-$name"
  fi
}

for seq in 2048 8192 16384; do
  bench fp4_cute "$seq" 1
  bench dequant "$seq" 1
done
# larger grouped M at fixed context
bench fp4_cute 2048 4
bench dequant 2048 4
# incumbent scattermoe marlin W4A16
bench marlin 2048 1
bench marlin 8192 1

echo "ALL DONE $(date -u +%H:%M:%S)"
