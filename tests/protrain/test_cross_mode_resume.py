"""Cross-mode (Mode A ↔ Mode C) checkpoint resume smoke test (M6C).

ProTrain has multiple operating modes:

* Mode A: all chunks persistent on GPU (``force_all_persistent=True``).
* Mode C: chunks sharded with offload (``zero3_shard=True``).

Different modes have different chunk layouts and optimizer-state shapes.
This module exercises whether a checkpoint saved in one mode loads cleanly
in the other.

Two layers of coverage:

* **Single-process (synthetic) round-trip** — :func:`test_cross_mode_resume_a_to_c`
  and :func:`test_cross_mode_resume_c_to_a`. Tiny Llama-arch LM, no CLI.
  Pins the state-dict round-trip + re-wrap invariant. Note: under
  ``world_size <= 1`` the wrapper auto-coerces ``zero3_shard`` to
  ``False`` (see ``model_wrapper.py:1019-1023``), so these tests
  exercise Mode A → Mode A with a different ``force_all_persistent``
  setting — i.e., the round-trip path runs but the *sharded layout*
  property the spec targets is NOT exercised. The next layer adds it.

* **Real multi-GPU subprocess** — :func:`test_real_multigpu_cross_mode_resume_a_to_c`
  and :func:`test_real_multigpu_cross_mode_resume_c_to_a`. Llama-3-8B +
  LoRA on 4×3090 via ``accelerate launch`` (subprocess). With
  ``world_size > 1`` the auto-coercion no longer fires and Mode C
  actually engages chunk sharding. These tests are marked ``slow`` +
  ``gpu`` and auto-skip when ``nvidia-smi`` reports < 4 GPUs.

  Empirical state on the 4×3090 rig (commit ``91e0912e``): both
  directions originally FAILED with structural bugs (see
  ``ProTrain/m6c_real_multigpu_report.md``):

  * A→C originally failed at HF Trainer's ``_load_from_checkpoint``
    with ``size mismatch ... shape in current model is torch.Size([0])``
    on every offloaded LoRA tensor. **M6C-fix-1 closes this gap** —
    the resume hook (``plugin.py:_install_resume_hook``)
    restore_to_gpu's the offloaded chunks, lets HF copy the loaded
    weights into full-shape ``param.data`` slots, then re-runs
    ``materialize_offload`` and rebuilds the optimizer adapter.
  * **M6C-fix-7** closed the autograd shape-capture race window at
    forward construction time via the shape-preserving expand
    placeholder (``chunk/manager.py::_shape_preserving_placeholder``;
    pinned by ``test_param_data_shape_preservation.py``). The 4×3090
    multi-GPU verification leg then surfaced a follow-on DDP
    ``_sync_module_states`` shared-storage hazard at construction
    time (the expand placeholder is shape-preserving but not write-
    safe; DDP's init-time broadcast tries to write it).
  * **M6C-fix-8** closes that follow-on by auto-injecting
    ``init_sync=False`` at DDP construction whenever the wrapped
    module carries the ProTrain marker (set in
    ``api/model_wrapper.py`` only on the multi-GPU sharded path).
    Architectural rationale: every rank already agrees on init state
    via ``materialize_offload``'s deterministic partition, so the
    construction-time broadcast is redundant for replicated params
    and INCORRECT for sharded params (broadcasting one rank's bytes
    over all ranks would corrupt per-rank shards). The
    ``_ddp_params_and_buffers_to_ignore`` registration also stays in
    place so the backward-pass allreduce skips chunk-managed params.
    Both ``test_real_multigpu_*`` tests now PASS on the 4×3090 rig.

Substitution rationale (single-process tests): real LLaMA-3-8B + CLI
subprocess invocations were the post-crash unsafe path at the time the
synthetic tests were written; the tested invariant (state-dict
round-trip across modes) is architecture-independent. The multi-GPU
subprocess tests below are now also exercised because the P2P fix in
commit ``91e0912e`` made 4×3090 launches stable.
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def _build_tiny_llama_lora():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        vocab_size=512,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        use_cache=False,
    )
    model = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg), cfg


def _wrap(
    model, cfg, *, force_all_persistent: bool, zero3_shard: bool, bs: int, seq: int
):
    import torch

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )
    wrapped = protrain_model_wrapper(
        model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=4 * (1 << 30),
        force_all_persistent=force_all_persistent,
        zero3_shard=zero3_shard,
    )
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)
    return wrapped, optim


def _train(wrapped, optim, *, n_iters, input_ids, labels) -> list[float]:
    losses: list[float] = []
    for i in range(n_iters):
        out = wrapped.module(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_value = float(loss.detach())
        assert math.isfinite(loss_value), f"iter {i}: non-finite loss {loss_value}"
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss_value)
    return losses


def _resume(wrapped, optim, model_state, optim_state):
    """Best-effort cross-mode load. Tolerates partial layouts: if Mode A's
    optimizer state cannot be remapped to Mode C's sharded layout (or
    vice versa), the load_state_dict is allowed to skip the optimizer
    state — we only require it not to crash, and that subsequent training
    still produces finite losses (the optimizer cold-starts, which is the
    documented limitation per phase2.md M6C bail criterion).
    """
    underlying = getattr(wrapped, "module", wrapped)
    try:
        # Allow strict=False because LoRA-PEFT state dicts contain only
        # trainable params; PEFT's load_state_dict accepts strict-False.
        load = getattr(underlying, "load_state_dict", None)
        if load is not None:
            load(model_state, strict=False)
    except Exception as exc:
        pytest.fail(f"cross-mode model state_dict load crashed: {exc}")

    if optim_state is not None and hasattr(optim, "load_state_dict"):
        try:
            optim.load_state_dict(optim_state)
        except Exception as exc:  # noqa: BLE001
            # Documented limitation: cross-mode optimizer-state remap may
            # not be implemented. We don't fail the test on this — we
            # log it and let training cold-start the optimizer.
            print(
                f"\n[cross-mode-resume] optimizer state load failed (cold-start): {exc}"
            )


def _make_inputs(cfg, *, bs: int, seq: int):
    import torch

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    input_ids = torch.randint(
        0, cfg.vocab_size, (bs, seq), device=device, dtype=torch.long
    )
    labels = input_ids.clone()
    return input_ids, labels


def test_cross_mode_resume_a_to_c() -> None:
    """Mode A → Mode C: train, save, re-wrap in Mode C, resume, assert finite training."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain cross-mode resume smoke requires CUDA.")

    model, cfg = _build_tiny_llama_lora()
    device = torch.device("cuda:0")
    model = model.to(device)

    bs, seq = 1, 32
    input_ids, labels = _make_inputs(cfg, bs=bs, seq=seq)

    # Mode A: train + capture state.
    wrapped_a, optim_a = _wrap(
        model, cfg, force_all_persistent=True, zero3_shard=False, bs=bs, seq=seq
    )
    losses_a = _train(wrapped_a, optim_a, n_iters=3, input_ids=input_ids, labels=labels)
    underlying_a = getattr(wrapped_a, "module", wrapped_a)
    model_state = {k: v.detach().clone() for k, v in underlying_a.state_dict().items()}
    optim_state = optim_a.state_dict() if hasattr(optim_a, "state_dict") else None

    # Mode C: re-wrap fresh from same model object, load state, train more.
    wrapped_c, optim_c = _wrap(
        model, cfg, force_all_persistent=False, zero3_shard=True, bs=bs, seq=seq
    )
    _resume(wrapped_c, optim_c, model_state, optim_state)
    losses_c = _train(wrapped_c, optim_c, n_iters=3, input_ids=input_ids, labels=labels)

    print(f"\nA→C resume: losses_a={losses_a} losses_c={losses_c}")

    # Acceptance: no crash above; losses are finite; Mode C losses are
    # not catastrophically larger than the last Mode A loss (allow 5x as
    # a generous bound — the optimizer may have cold-started).
    assert all(math.isfinite(v) for v in losses_c), (
        f"non-finite Mode C loss: {losses_c}"
    )
    assert losses_c[0] < 5.0 * losses_a[-1] + 1.0, (
        f"Mode C loss diverged after A→C resume: a-end={losses_a[-1]} "
        f"c-start={losses_c[0]} (>5x is treated as catastrophic divergence)"
    )


def test_cross_mode_resume_c_to_a() -> None:
    """Mode C → Mode A: symmetric. Train Mode C, save, resume in Mode A."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain cross-mode resume smoke requires CUDA.")

    model, cfg = _build_tiny_llama_lora()
    device = torch.device("cuda:0")
    model = model.to(device)

    bs, seq = 1, 32
    input_ids, labels = _make_inputs(cfg, bs=bs, seq=seq)

    wrapped_c, optim_c = _wrap(
        model, cfg, force_all_persistent=False, zero3_shard=True, bs=bs, seq=seq
    )
    losses_c = _train(wrapped_c, optim_c, n_iters=3, input_ids=input_ids, labels=labels)
    underlying_c = getattr(wrapped_c, "module", wrapped_c)
    model_state = {k: v.detach().clone() for k, v in underlying_c.state_dict().items()}
    optim_state = optim_c.state_dict() if hasattr(optim_c, "state_dict") else None

    wrapped_a, optim_a = _wrap(
        model, cfg, force_all_persistent=True, zero3_shard=False, bs=bs, seq=seq
    )
    _resume(wrapped_a, optim_a, model_state, optim_state)
    losses_a = _train(wrapped_a, optim_a, n_iters=3, input_ids=input_ids, labels=labels)

    print(f"\nC→A resume: losses_c={losses_c} losses_a={losses_a}")

    assert all(math.isfinite(v) for v in losses_a), (
        f"non-finite Mode A loss: {losses_a}"
    )
    assert losses_a[0] < 5.0 * losses_c[-1] + 1.0, (
        f"Mode A loss diverged after C→A resume: c-end={losses_c[-1]} "
        f"a-start={losses_a[0]} (>5x is treated as catastrophic divergence)"
    )


# =============================================================================
# Real multi-GPU subprocess-based cross-mode resume tests (M6C audit close).
#
# The single-process tests above silently degrade Mode C → Mode A under
# ``world_size <= 1`` (see module docstring for the auto-coercion at
# ``model_wrapper.py:1019-1023``). The two ``test_real_multigpu_*`` tests
# below close that gap by invoking ``accelerate launch --num_processes 4``
# in a subprocess with a real Llama-3-8B + LoRA workload, so the
# ``world_size > 1`` branch runs and Mode C actually engages chunk
# sharding (``zero3_shard=True (requested=True)`` in the log).
#
# Originally on commit ``91e0912e`` (4×3090 rig, GPUs 1/4/5/7, ProTrain
# Phase 2 branch) both directions FAILED — see the report at
# ``ProTrain/m6c_real_multigpu_report.md``. The M6C-fix-{1..8} chain
# closes the path: M6C-fix-1 the cross-mode resume monkey-patch in
# ``plugin.py:_install_resume_hook`` (load_from_checkpoint shape-
# mismatch); M6C-fix-{2..6} the per-LoRA-container gather hook
# coverage in profiler/on_demand.py and runtime/hooks.py; M6C-fix-7
# the architectural shape-preserving expand placeholder in
# ``chunk/manager.py::_shape_preserving_placeholder`` (autograd
# shape-capture race window); M6C-fix-8 the DDP init_sync=False
# auto-injection in ``api/model_wrapper.py`` (DDP construction-time
# broadcast hazard on the expand placeholder). Both
# ``test_real_multigpu_*`` tests now PASS on the 4×3090 rig (xfail
# markers removed in the M6C-fix-8 commit).
# =============================================================================


def _pick_free_port() -> int:
    """Bind to port 0 so the OS hands back a free port. Mirrors the
    helper in :mod:`test_multi_gpu_7b` to avoid MASTER_PORT collisions
    on a busy box."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _nvidia_smi_gpu_indices() -> list[int]:
    """Return the list of GPU indices reported by ``nvidia-smi``.

    Uses the subprocess-level invocation rather than torch so that the
    pytest host process's CUDA_VISIBLE_DEVICES masking does not under-
    report visibility.
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


def _nvidia_smi_gpu_count() -> int:
    """Return the number of GPUs reported by ``nvidia-smi``.

    Thin wrapper over :func:`_nvidia_smi_gpu_indices` for callers that
    only need the count.
    """
    return len(_nvidia_smi_gpu_indices())


# Indices ``_launch_axolotl`` pins via ``CUDA_VISIBLE_DEVICES``. The
# corresponding precheck must verify these specific indices actually
# exist on the host — a count-based >=4 check passes on any 4-GPU box
# but launch fails late if e.g. GPU 7 isn't present. Kept in sync with
# the env in ``_launch_axolotl``.
_REQUIRED_GPU_INDICES = (1, 4, 5, 7)


_MODE_A_YAML = textwrap.dedent(
    """\
    base_model: NousResearch/Meta-Llama-3-8B-Instruct
    model_type: LlamaForCausalLM
    load_in_8bit: false
    load_in_4bit: false
    strict: false
    datasets:
      - path: tatsu-lab/alpaca
        type: alpaca
    val_set_size: 0.0
    output_dir: {output_dir}
    {resume_line}
    sequence_len: 256
    sample_packing: false
    pad_to_sequence_len: false
    adapter: lora
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
    protrain_force_all_persistent: true
    protrain_zero3_shard: false
    gradient_accumulation_steps: 1
    micro_batch_size: 1
    max_steps: {max_steps}
    optimizer: adamw_torch
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
    save_steps: {save_steps}
    save_first_step: false
    save_total_limit: 2
    warmup_steps: 2
    weight_decay: 0.0
    """
)


_MODE_C_YAML = textwrap.dedent(
    """\
    base_model: NousResearch/Meta-Llama-3-8B-Instruct
    model_type: LlamaForCausalLM
    load_in_8bit: false
    load_in_4bit: false
    strict: false
    datasets:
      - path: tatsu-lab/alpaca
        type: alpaca
    val_set_size: 0.0
    output_dir: {output_dir}
    {resume_line}
    sequence_len: 256
    sample_packing: false
    pad_to_sequence_len: false
    adapter: lora
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
    protrain_n_buffer_override: 8
    protrain_n_swap_override: 0
    protrain_n_checkpoint_override: 0
    protrain_n_offload_override: 32
    gradient_accumulation_steps: 1
    micro_batch_size: 1
    max_steps: {max_steps}
    optimizer: adamw_torch
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
    save_steps: {save_steps}
    save_first_step: false
    save_total_limit: 2
    warmup_steps: 2
    weight_decay: 0.0
    """
)


def _launch_axolotl(yaml_path: Path, log_path: Path, repo_root: Path) -> int:
    """Run a single ``accelerate launch`` of ``axolotl.cli.train``.

    Returns the subprocess exit code. Uses GPUs 1,4,5,7 via
    CUDA_VISIBLE_DEVICES + PCI_BUS_ID, the only stable 4-GPU set on
    this rig (GPUs 0/3/6 are heterogeneous Blackwell/RTX 5090 cards
    that fail the P2P check). PYTHONPATH is forced to the worktree
    ``src/`` so accelerate doesn't pick up a different axolotl install.
    """
    env = os.environ.copy()
    env["DS_SKIP_CUDA_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_root / "src")
    env["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Pick a free port; prevents EADDRINUSE if other torch.distributed
    # processes are already bound (e.g. concurrent tests on the same
    # rig). Accelerate forwards MASTER_PORT into the child group.
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
            timeout=720,  # per-launch budget; multi-GPU bring-up takes ~1 min
        )
    return proc.returncode


def _require_real_multigpu() -> None:
    """Skip helper for the multi-GPU subprocess tests."""
    visible = _nvidia_smi_gpu_indices()
    missing = [i for i in _REQUIRED_GPU_INDICES if i not in visible]
    if missing:
        pytest.skip(
            f"real multi-GPU cross-mode resume requires GPU indices "
            f"{list(_REQUIRED_GPU_INDICES)} (hard-coded in "
            f"``_launch_axolotl``); nvidia-smi reports {visible}, "
            f"missing {missing}"
        )
    # accelerate must be importable in the *child* invocation; check it
    # in the parent first so we get a clean skip rather than a child-
    # subprocess crash.
    try:
        import accelerate  # noqa: F401
    except ImportError:
        pytest.skip("accelerate not installed; required for multi-GPU launch")


def _repo_root() -> Path:
    """Resolve the worktree root (parent of ``src/axolotl``)."""
    here = Path(__file__).resolve()
    # tests/protrain/test_cross_mode_resume.py -> tests/protrain -> tests -> repo
    return here.parents[2]


@pytest.mark.slow
@pytest.mark.gpu
def test_real_multigpu_cross_mode_resume_a_to_c(tmp_path: Path) -> None:
    """4×3090 cross-mode A→C: train+save Mode A, resume in Mode C.

    Two subprocess launches, sequentially. Phase 1 trains 5 steps in
    Mode A and writes ``checkpoint-5/`` under ``modeA_ckpt/``. Phase 2
    sets ``resume_from_checkpoint`` to that path, forces Mode C
    (``protrain_zero3_shard: true`` + non-persistent overrides), and
    asks for max_steps=10 (so 5 more steps after resume).

    Acceptance: both phases exit 0; Phase 2's stdout shows loss values
    for steps 6..10 with no Traceback.

    Status (M6C-fix-8): PASSING. The full M6C chain (fixes 1..8) closed
    the multi-GPU plain-LoRA Mode C cross-mode resume path. M6C-fix-7
    architecturally closed the autograd shape-capture race window via
    the shape-preserving expand placeholder; M6C-fix-8 closed the
    follow-on DDP ``_sync_module_states`` shared-storage hazard by
    auto-injecting ``init_sync=False`` on the chunk-managed model
    (every rank already agreed on init state via
    ``materialize_offload``'s deterministic partition, so the
    construction-time broadcast was redundant; the module-level
    ``_ddp_params_and_buffers_to_ignore`` registration also stays in
    place so the backward-pass allreduce skips chunk-managed params,
    matching ProTrain's reduce_scatter contract). See
    ``api/model_wrapper.py``'s M6C-fix-8 block and
    ``tests/protrain/test_param_data_shape_preservation.py`` for the
    full architectural invariant + 8 unit tests.
    """
    _require_real_multigpu()

    repo_root = _repo_root()
    workdir = tmp_path
    modeA_ckpt_dir = workdir / "modeA_ckpt"
    modeC_resumed_dir = workdir / "modeC_resumed"

    # ---- Phase 1: Mode A train + save ------------------------------------
    yaml_a = workdir / "modeA_save.yml"
    yaml_a.write_text(
        _MODE_A_YAML.format(
            output_dir=str(modeA_ckpt_dir),
            resume_line="",
            max_steps=5,
            save_steps=5,
        )
    )
    log_a = workdir / "modeA_save.log"
    rc_a = _launch_axolotl(yaml_a, log_a, repo_root)
    assert rc_a == 0, (
        f"Mode A train+save subprocess exited {rc_a}; tail:\n"
        f"{log_a.read_text()[-3000:]}"
    )
    assert (modeA_ckpt_dir / "checkpoint-5").is_dir(), (
        f"Mode A did not produce checkpoint-5/ under {modeA_ckpt_dir}; "
        f"contents: {list(modeA_ckpt_dir.iterdir()) if modeA_ckpt_dir.exists() else 'NONE'}"
    )

    # ---- Phase 2: Mode C resume from Mode A's checkpoint -----------------
    yaml_c = workdir / "modeC_resume.yml"
    yaml_c.write_text(
        _MODE_C_YAML.format(
            output_dir=str(modeC_resumed_dir),
            resume_line=f"resume_from_checkpoint: {modeA_ckpt_dir / 'checkpoint-5'}",
            max_steps=10,
            save_steps=10,
        )
    )
    log_c = workdir / "modeC_resume.log"
    rc_c = _launch_axolotl(yaml_c, log_c, repo_root)
    log_c_text = log_c.read_text()
    assert rc_c == 0, (
        f"Mode C resume subprocess exited {rc_c}; tail:\n{log_c_text[-3000:]}"
    )
    assert "Traceback" not in log_c_text, (
        f"Mode C resume produced a Traceback; tail:\n{log_c_text[-3000:]}"
    )
    # Sanity: the per-step loss line format Axolotl emits contains
    # ``'loss':``. Five resumed steps should leave at least 5 such lines
    # (one per training_step log). Anything less means the loop didn't
    # enter the resumed range.
    assert log_c_text.count("'loss':") >= 5, (
        f"Mode C resume did not produce >= 5 step-loss lines; tail:\n"
        f"{log_c_text[-3000:]}"
    )


@pytest.mark.slow
@pytest.mark.gpu
def test_real_multigpu_cross_mode_resume_c_to_a(tmp_path: Path) -> None:
    """4×3090 cross-mode C→A: train+save Mode C, resume in Mode A.

    Symmetric to A→C. Two subprocess launches, sequentially. Phase 1
    forces Mode C (sharded chunks, non-persistent) and trains 5 steps;
    Phase 2 resumes in Mode A.

    Acceptance: both phases exit 0; Phase 2's stdout shows 5 resumed
    step losses with no Traceback.

    Status (M6C-fix-8): PASSING. See A→C test docstring for the full
    M6C chain close. Phase 1 (Mode C train) exercises the same DDP
    init_sync bypass as the A→C Phase 2 (Mode C resume); Phase 2 here
    (Mode A resume) goes through the standard DDP path (no shape-
    preserving placeholders engaged in Mode A — the bypass marker is
    not set on the model so DDP's __init__ runs the normal init_sync
    broadcast, correct for the all-persistent path).
    """
    _require_real_multigpu()

    repo_root = _repo_root()
    workdir = tmp_path
    modeC_ckpt_dir = workdir / "modeC_ckpt"
    modeA_resumed_dir = workdir / "modeA_resumed"

    # ---- Phase 1: Mode C train + save ------------------------------------
    yaml_c = workdir / "modeC_save.yml"
    yaml_c.write_text(
        _MODE_C_YAML.format(
            output_dir=str(modeC_ckpt_dir),
            resume_line="",
            max_steps=5,
            save_steps=5,
        )
    )
    log_c = workdir / "modeC_save.log"
    rc_c = _launch_axolotl(yaml_c, log_c, repo_root)
    assert rc_c == 0, (
        f"Mode C train+save subprocess exited {rc_c}; tail:\n"
        f"{log_c.read_text()[-3000:]}"
    )
    assert (modeC_ckpt_dir / "checkpoint-5").is_dir(), (
        f"Mode C did not produce checkpoint-5/ under {modeC_ckpt_dir}"
    )

    # ---- Phase 2: Mode A resume from Mode C's checkpoint -----------------
    yaml_a = workdir / "modeA_resume.yml"
    yaml_a.write_text(
        _MODE_A_YAML.format(
            output_dir=str(modeA_resumed_dir),
            resume_line=f"resume_from_checkpoint: {modeC_ckpt_dir / 'checkpoint-5'}",
            max_steps=10,
            save_steps=10,
        )
    )
    log_a = workdir / "modeA_resume.log"
    rc_a = _launch_axolotl(yaml_a, log_a, repo_root)
    log_a_text = log_a.read_text()
    assert rc_a == 0, (
        f"Mode A resume subprocess exited {rc_a}; tail:\n{log_a_text[-3000:]}"
    )
    assert "Traceback" not in log_a_text, (
        f"Mode A resume produced a Traceback; tail:\n{log_a_text[-3000:]}"
    )
    assert log_a_text.count("'loss':") >= 5, (
        f"Mode A resume did not produce >= 5 step-loss lines; tail:\n"
        f"{log_a_text[-3000:]}"
    )
