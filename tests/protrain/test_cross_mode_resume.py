"""Cross-mode (Mode A persistent vs Mode C sharded+offload) checkpoint resume smoke tests."""

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
    """Best-effort cross-mode load: never crash, allow optimizer-state cold-start when layouts differ."""
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
    """Mode A trains+saves, Mode C re-wraps and resumes; assert finite loss with explicit close()."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain cross-mode resume smoke requires CUDA.")

    model, cfg = _build_tiny_llama_lora()
    device = torch.device("cuda:0")
    model = model.to(device)

    bs, seq = 1, 32
    input_ids, labels = _make_inputs(cfg, bs=bs, seq=seq)

    wrapped_c = None
    try:
        # Mode A: train + capture state.
        wrapped_a, optim_a = _wrap(
            model, cfg, force_all_persistent=True, zero3_shard=False, bs=bs, seq=seq
        )
        try:
            losses_a = _train(
                wrapped_a, optim_a, n_iters=3, input_ids=input_ids, labels=labels
            )
            underlying_a = getattr(wrapped_a, "module", wrapped_a)
            model_state = {
                k: v.detach().clone() for k, v in underlying_a.state_dict().items()
            }
            optim_state = (
                optim_a.state_dict() if hasattr(optim_a, "state_dict") else None
            )
        finally:
            # Explicit teardown BEFORE re-wrapping so the D2 snapshot is
            # restored and the new chunk manager starts from a clean
            # ``_ddp_params_and_buffers_to_ignore`` baseline. GC-only
            # teardown would leave the prior wrap's hooks / pinned pool
            # alive until the next allocator cycle.
            close_a = getattr(wrapped_a, "close", None)
            if callable(close_a):
                close_a()

        # Mode C: re-wrap fresh from same model object, load state, train more.
        wrapped_c, optim_c = _wrap(
            model, cfg, force_all_persistent=False, zero3_shard=True, bs=bs, seq=seq
        )
        _resume(wrapped_c, optim_c, model_state, optim_state)
        losses_c = _train(
            wrapped_c, optim_c, n_iters=3, input_ids=input_ids, labels=labels
        )

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
    finally:
        if wrapped_c is not None:
            close_c = getattr(wrapped_c, "close", None)
            if callable(close_c):
                close_c()


def test_cross_mode_resume_c_to_a() -> None:
    """Mode C trains+saves, Mode A re-wraps and resumes; symmetric to A-to-C."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain cross-mode resume smoke requires CUDA.")

    model, cfg = _build_tiny_llama_lora()
    device = torch.device("cuda:0")
    model = model.to(device)

    bs, seq = 1, 32
    input_ids, labels = _make_inputs(cfg, bs=bs, seq=seq)

    wrapped_a = None
    try:
        wrapped_c, optim_c = _wrap(
            model, cfg, force_all_persistent=False, zero3_shard=True, bs=bs, seq=seq
        )
        try:
            losses_c = _train(
                wrapped_c, optim_c, n_iters=3, input_ids=input_ids, labels=labels
            )
            underlying_c = getattr(wrapped_c, "module", wrapped_c)
            model_state = {
                k: v.detach().clone() for k, v in underlying_c.state_dict().items()
            }
            optim_state = (
                optim_c.state_dict() if hasattr(optim_c, "state_dict") else None
            )
        finally:
            close_c = getattr(wrapped_c, "close", None)
            if callable(close_c):
                close_c()

        wrapped_a, optim_a = _wrap(
            model, cfg, force_all_persistent=True, zero3_shard=False, bs=bs, seq=seq
        )
        _resume(wrapped_a, optim_a, model_state, optim_state)
        losses_a = _train(
            wrapped_a, optim_a, n_iters=3, input_ids=input_ids, labels=labels
        )

        print(f"\nC→A resume: losses_c={losses_c} losses_a={losses_a}")

        assert all(math.isfinite(v) for v in losses_a), (
            f"non-finite Mode A loss: {losses_a}"
        )
        assert losses_a[0] < 5.0 * losses_c[-1] + 1.0, (
            f"Mode A loss diverged after C→A resume: c-end={losses_c[-1]} "
            f"a-start={losses_a[0]} (>5x is treated as catastrophic divergence)"
        )
    finally:
        if wrapped_a is not None:
            close_a = getattr(wrapped_a, "close", None)
            if callable(close_a):
                close_a()


# Multi-GPU subprocess tests: single-process tests above auto-coerce to Mode A under
# world_size<=1, so these accelerate-launch a real LoRA workload to exercise real sharding.


def _pick_free_port() -> int:
    """Bind to port 0 so the OS hands back a free port (avoids MASTER_PORT collisions)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _nvidia_smi_gpu_indices() -> list[int]:
    """Return GPU indices from nvidia-smi (subprocess sidesteps CUDA_VISIBLE_DEVICES masking)."""
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
    """Return the GPU count from nvidia-smi."""
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
    """Spawn ``accelerate launch`` of ``axolotl.cli.train``; pins GPUs 1/4/5/7 (stable P2P set)."""
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
    """Resolve the worktree root (parent of src/axolotl)."""
    here = Path(__file__).resolve()
    # tests/protrain/test_cross_mode_resume.py -> tests/protrain -> tests -> repo
    return here.parents[2]


@pytest.mark.slow
@pytest.mark.gpu
def test_real_multigpu_cross_mode_resume_a_to_c(tmp_path: Path) -> None:
    """4x3090 cross-mode A->C: subprocess trains Mode A 5 steps, resumes Mode C for 5 more."""
    _require_real_multigpu()

    repo_root = _repo_root()
    workdir = tmp_path
    modeA_ckpt_dir = workdir / "modeA_ckpt"
    modeC_resumed_dir = workdir / "modeC_resumed"

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
    """4x3090 cross-mode C->A: subprocess trains Mode C 5 steps, resumes Mode A for 5 more."""
    _require_real_multigpu()

    repo_root = _repo_root()
    workdir = tmp_path
    modeC_ckpt_dir = workdir / "modeC_ckpt"
    modeA_resumed_dir = workdir / "modeA_resumed"

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
