"""Multi-GPU regression: bnb 4-bit + paged_adamw_8bit + Mode C at seq=2048.

This pins the failure pattern surfaced by Coverage audit Block B
(`ProTrain/m0_artifacts/ext_b1_qlora_paged_seq2048_mgpu.log`) where
DDP construction-time ``_sync_module_states._broadcast_coalesced``
raised ``RuntimeError: unsupported operation: more than one element
of the written-to tensor refers to a single memory location`` on
every rank, before training step 0. The failure was specific to the
QLoRA (load_in_4bit=true) + paged_adamw_8bit + Mode C
(zero3_shard=true, force_all_persistent=false, non-persistent
overrides) + seq=2048 + 4-rank intersection.

The Block B audit log was captured 75 minutes BEFORE M6C-fix-8
(commit ``17ffb8d1``) landed; the patch monkey-patches
``DistributedDataParallel.__init__`` to auto-inject
``init_sync=False`` whenever the wrapped module carries the
``_protrain_ddp_skip_init_sync`` marker (set in
``api/model_wrapper.py`` only on the multi-GPU sharded
``_shape_preserving`` path). On 4×3090 re-test under the current tip
(``rerun_1778547187.log``) the same YAML now trains 5 steps cleanly
with M6C-fix-8 firing the ``patched-injection of init_sync=False``
log line and ``materialize_offload`` registering 731/731
chunk-managed param names into
``model._ddp_params_and_buffers_to_ignore``. This test re-runs the
exact reproducer YAML to lock that behaviour.

The launch helper mirrors ``test_cross_mode_resume.py``'s
``_launch_axolotl``: GPUs 1,4,5,7 via ``CUDA_VISIBLE_DEVICES`` +
``PCI_BUS_ID``, the only stable 4-GPU set on the reference rig
(GPUs 0/3/6 are Blackwell/RTX 5090 cards that fail the P2P check;
the user's live training also pins 0/3 on the same hardware).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _pick_free_port() -> int:
    """Bind to port 0 so the OS hands back a free port. Mirrors the
    helper in :mod:`test_cross_mode_resume` to avoid MASTER_PORT
    collisions on a busy box."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _nvidia_smi_gpu_count() -> int:
    """Return the number of GPUs reported by ``nvidia-smi``.

    Uses the subprocess-level invocation rather than torch so the
    pytest host process's ``CUDA_VISIBLE_DEVICES`` masking does not
    under-report visibility.
    """
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
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


def _repo_root() -> Path:
    """Resolve the worktree root (parent of ``src/axolotl``)."""
    here = Path(__file__).resolve()
    # tests/protrain/test_paged_adam_offload_mgpu.py -> tests/protrain -> tests -> repo
    return here.parents[2]


# Reproducer YAML: identical to
# ``ProTrain/m0_artifacts/ext_b1_qlora_paged_seq2048_mgpu.yml`` modulo
# ``output_dir`` (kept ``{output_dir}``-templated so the test fixture
# can land it under ``tmp_path``). Keep this string in lockstep with
# the audit YAML — every key here is part of the regression contract.
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
    """Run a single ``accelerate launch`` of ``axolotl.cli.train``.

    Returns the subprocess exit code. Pins GPUs 1,4,5,7 + 720 s
    timeout (the audit's re-run on the same hardware completed in
    ~5–6 minutes wall-clock; 720 s leaves slack for slow hook
    install on cold caches).
    """
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
    if _nvidia_smi_gpu_count() < 4:
        pytest.skip(
            f"4-bit + paged_adamw_8bit + Mode C multi-GPU regression requires "
            f">= 4 GPUs; nvidia-smi reports {_nvidia_smi_gpu_count()}"
        )
    try:
        import accelerate  # noqa: F401
    except ImportError:
        pytest.skip("accelerate not installed; required for multi-GPU launch")


@pytest.mark.slow
@pytest.mark.gpu
def test_paged_adam_offload_mgpu_no_ddp_broadcast_crash(tmp_path: Path) -> None:
    """4×3090 QLoRA + paged_adamw_8bit + Mode C at seq=2048 trains 5 steps.

    Coverage audit Block B captured the failure mode this pin
    regresses against:

      RuntimeError: unsupported operation: more than one element of
      the written-to tensor refers to a single memory location.
      Please clone() the tensor before performing the operation.

    The crash happened in
    ``DistributedDataParallel.__init__ → _sync_module_states →
    _broadcast_coalesced`` BEFORE step 0, on the chunk-managed
    shape-preserving expand placeholders that M6C-fix-7 introduced
    to close the autograd shape-capture race. M6C-fix-8 closes the
    DDP broadcast hazard by patching ``DDP.__init__`` to auto-inject
    ``init_sync=False`` whenever the wrapped module carries the
    ``_protrain_ddp_skip_init_sync`` marker (set in
    ``api/model_wrapper.py`` only on the multi-GPU sharded
    ``_shape_preserving`` path).

    Acceptance:

    * subprocess exits 0,
    * no ``Traceback`` in the captured log,
    * the M6C-fix-8 ``patched-injection of init_sync=False``
      diagnostic appears (proves the bypass actually engaged on
      this YAML's path — guards against a future refactor that
      silently relaxes the gate),
    * the ``_ddp_params_and_buffers_to_ignore`` registration log
      records >= 1 chunk-managed name per rank (defends against a
      future regression where the registration silently empties out
      due to a name-resolution drift between the chunk manager and
      ``model.named_parameters()``),
    * >= 5 per-step loss log lines (the configured ``max_steps``).
    """
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
    # The M6C-fix-8 bypass MUST engage for this config — that's the
    # whole point of the regression. The patched-injection log line
    # fires at DDP construction time when the marker is detected.
    assert "patched-injection of init_sync=False" in log_text, (
        f"M6C-fix-8 DDP init_sync bypass did NOT fire on this YAML's "
        f"path — the bug is likely back. tail:\n{log_tail}"
    )
    # The ``_ddp_params_and_buffers_to_ignore`` registration log line
    # records the count of chunk-managed names per rank; pre-M6C-fix-8
    # this was the only defence and it was insufficient on the
    # sharded path. Today it's the SECOND line of defence (with the
    # init_sync bypass) — keep pinning it so the second defence
    # doesn't quietly disappear.
    assert "registered" in log_text and "chunk-managed param names" in log_text, (
        f"M6C-fix-8 chunk-managed param-name registration log line is "
        f"missing — the second line of defence has regressed. "
        f"tail:\n{log_tail}"
    )
    # Sanity: 5 steps of training means at least 5 per-step loss lines.
    assert log_text.count("'loss':") >= 5, (
        f"expected >= 5 per-step loss log lines for max_steps=5, got "
        f"{log_text.count(chr(0x27) + 'loss' + chr(0x27) + ':')}; "
        f"tail:\n{log_tail}"
    )
