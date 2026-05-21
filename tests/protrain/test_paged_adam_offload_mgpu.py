"""Multi-GPU regression: QLoRA + paged_adamw_8bit + Mode C at seq=2048 crashed DDP broadcast on shape-preserving placeholders."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _pick_free_port() -> int:
    """Bind to port 0 so the OS hands back a free port and MASTER_PORT collisions are impossible."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _nvidia_smi_gpu_indices() -> list[int]:
    """List GPU indices via nvidia-smi to bypass the pytest host's CUDA_VISIBLE_DEVICES masking."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", errors="replace")
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return []
    indices: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            indices.append(int(line))
        except ValueError:
            continue
    return indices


# Precheck must verify these specific indices since count-based gating would still let launches fail late.
_REQUIRED_GPU_INDICES = (1, 4, 5, 7)


def _repo_root() -> Path:
    """Resolve the worktree root (parent of ``src/axolotl``)."""
    here = Path(__file__).resolve()
    # tests/protrain/test_paged_adam_offload_mgpu.py -> tests/protrain -> tests -> repo
    return here.parents[2]


# Every key in this YAML is part of the regression contract; do not edit without re-validating the failure repro.
_REPRODUCER_YAML = textwrap.dedent(
    """\
    base_model: NousResearch/Meta-Llama-3-8B-Instruct
    model_type: LlamaForCausalLM

    load_in_8bit: false
    load_in_4bit: true
    strict: false

    datasets:
      - path: tatsu-lab/alpaca
        type: alpaca
    val_set_size: 0.0
    output_dir: {output_dir}

    sequence_len: 2048
    sample_packing: false
    pad_to_sequence_len: true

    adapter: qlora
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    lora_target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - up_proj
      - down_proj
      - gate_proj

    plugins:
      - axolotl.integrations.protrain.ProTrainPlugin

    protrain_auto_memory: true
    protrain_auto_mode: false
    protrain_force_all_persistent: false
    protrain_zero3_shard: true
    protrain_n_persist_override: 0
    protrain_n_buffer_override: 12
    protrain_n_swap_override: 0
    protrain_n_checkpoint_override: 32

    gradient_accumulation_steps: 1
    micro_batch_size: 1
    max_steps: 5
    optimizer: paged_adamw_8bit
    lr_scheduler: cosine
    learning_rate: 0.0002

    bf16: true
    fp16: false
    tf32: false

    gradient_checkpointing: false

    flash_attention: false
    xformers_attention: false

    lora_mlp_kernel: false
    lora_qkv_kernel: false
    lora_o_kernel: false

    logging_steps: 1
    save_steps: 100
    save_first_step: false
    save_total_limit: 1

    warmup_steps: 1
    weight_decay: 0.0

    peft_autocast_adapter_dtype: false
    """
)


def _launch_axolotl(yaml_path: Path, log_path: Path, repo_root: Path) -> int:
    """Run accelerate launch of axolotl.cli.train; pins GPUs 1,4,5,7 with a 720s timeout for cold-cache hook install."""
    env = os.environ.copy()
    env["DS_SKIP_CUDA_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_root / "src")
    env["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("MASTER_PORT", str(_pick_free_port()))

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "4",
        "--mixed_precision",
        "bf16",
        "-m",
        "axolotl.cli.train",
        str(yaml_path),
    ]
    with log_path.open("w") as f:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=720,
        )
    return proc.returncode


def _require_real_multigpu() -> None:
    """Skip helper for the multi-GPU subprocess test."""
    visible = _nvidia_smi_gpu_indices()
    missing = [i for i in _REQUIRED_GPU_INDICES if i not in visible]
    if missing:
        pytest.skip(
            f"4-bit + paged_adamw_8bit + Mode C multi-GPU regression requires "
            f"GPU indices {list(_REQUIRED_GPU_INDICES)} (hard-coded in "
            f"``_launch_axolotl``); nvidia-smi reports {visible}, "
            f"missing {missing}"
        )
    try:
        import accelerate  # noqa: F401
    except ImportError:
        pytest.skip("accelerate not installed; required for multi-GPU launch")


@pytest.mark.slow
@pytest.mark.gpu
def test_paged_adam_offload_mgpu_no_ddp_broadcast_crash(tmp_path: Path) -> None:
    """4x3090 QLoRA + paged_adamw_8bit + Mode C at seq=2048 trains 5 steps without the DDP broadcast crash on expand placeholders."""
    _require_real_multigpu()

    repo_root = _repo_root()
    workdir = tmp_path
    output_dir = workdir / "protrain_paged_qlora_mgpu_out"

    yaml_path = workdir / "ext_b1_qlora_paged_seq2048_mgpu.yml"
    yaml_path.write_text(_REPRODUCER_YAML.format(output_dir=str(output_dir)))

    log_path = workdir / "ext_b1_qlora_paged_seq2048_mgpu.log"
    rc = _launch_axolotl(yaml_path, log_path, repo_root)
    log_text = log_path.read_text()
    log_tail = log_text[-3000:]

    assert rc == 0, (
        f"paged_adamw_8bit + Mode C multi-GPU subprocess exited {rc} "
        f"(expected 0); tail:\n{log_tail}"
    )
    assert "Traceback" not in log_text, (
        f"unexpected Traceback in the captured log; tail:\n{log_tail}"
    )
    # DDP init_sync bypass must engage when the chunk-managed marker is present, else broadcast over expand placeholders crashes.
    assert "patched-injection of init_sync=False" in log_text, (
        f"DDP init_sync bypass did NOT fire on this YAML's path. tail:\n{log_tail}"
    )
    # Chunk-managed param-name registration is the secondary defence; keep pinning it so it cannot silently empty out.
    assert "registered" in log_text and "chunk-managed param names" in log_text, (
        f"chunk-managed param-name registration log line missing. tail:\n{log_tail}"
    )
    # Sanity: 5 steps of training means at least 5 per-step loss lines.
    assert log_text.count("'loss':") >= 5, (
        f"expected >= 5 per-step loss log lines for max_steps=5, got "
        f"{log_text.count(chr(0x27) + 'loss' + chr(0x27) + ':')}; "
        f"tail:\n{log_tail}"
    )
