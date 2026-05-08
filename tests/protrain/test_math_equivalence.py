"""Math-equivalence check: ProTrain vs vanilla AdamW.

The integration history of the ProTrain stack is long and dominated by
*runtime* fixes (gather/release lifecycle, CPU-Adam ↔ D2H race closes,
peak-cost calibration tweaks). Those changes are easy to verify on the
performance and feasibility axes, but the central correctness claim
— *"ProTrain trains exactly the same model as vanilla AdamW, just
with a chunk-pack + offload runtime around it"* — is implicit in the
design and has no end-to-end regression test outside the DeepSpeed
external-baseline test (which deliberately does NOT compare loss
trajectories tightly because DeepSpeed's master-precision and
gradient-scaling pipelines differ from ProTrain's).

This test closes that gap with a paired, small-model, fixed-input
loop:

* **Vanilla baseline** — fresh-init Llama in bf16 with a
  reference AdamW optimizer that mirrors what ProTrain's adapters
  actually run (DeepSpeedCPUAdam ``adamw_mode=True`` +
  ``fp32_optimizer_states=True`` for offloaded chunks; Apex FusedAdam
  in adamw mode for persistent chunks — both keep fp32 moments and
  apply the standard decoupled-decay AdamW update). The reference
  loop here implements that math directly so it's also robust on rigs
  where Apex FusedAdam is unavailable and the GPU adapter falls back
  to ``torch.optim.AdamW`` (which keeps moments in the param dtype —
  a slightly different regime).

* **ProTrain run** — SAME initial weights (loaded via
  ``state_dict``), SAME inputs, SAME hyperparams, SAME number of
  iters. Wrapped in two configurations:

  1. ``force_all_persistent=True`` — exercises the GPU FusedAdam
     adapter path and the chunk-pack data layout, but with no
     CPU-master / offload / recompute machinery.
  2. Explicit override path with ``n_offload>0`` and
     ``n_checkpoint>0`` — exercises the CPU-master path, the
     gather/release lifecycle, the per-param post-accumulate-grad
     hook, and the recompute-block code path.

Acceptance bars:

* Iter-0 forward loss agrees within 1e-3 relative — no parameter
  update has happened yet, so any larger gap is a forward-path bug
  (block-wrapper reordering, dropout RNG mismatch, wrong dtype on the
  recompute path).
* Iter-1 .. iter-N losses agree within 1% relative. This admits bf16
  reduction-order noise in linear / softmax / RMSNorm kernels and
  small numerical differences from the Adam moment update path
  (DeepSpeedCPUAdam's bias-correction is bit-for-bit different from
  pure-Python reference math in the last bf16 ulp).
* Final per-trainable-param state agrees within 1% relative L2 error
  per tensor.
* No NaN / Inf in either trajectory.

Sized to run in well under 5 minutes on a single 24 GB 3090: a 4-layer
Llama (~80 M params bf16), batch=2, seq=64, 6 iterations, lr=1e-3.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torch import nn


def _mark(stage: str) -> None:
    """Emit a progress marker that survives pytest output buffering."""

    line = f"[protrain-math-equiv] {stage}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    sys.stderr.write(line)
    sys.stderr.flush()


def _ref_adamw_step(
    params: list["nn.Parameter"],
    state: dict,
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    step: int,
) -> None:
    """Reference AdamW update: bf16 params, fp32 moments, fp32 update math.

    Mirrors DeepSpeedCPUAdam(adamw_mode=True, fp32_optimizer_states=True)
    and Apex FusedAdam in adamw mode + fp32 moments — both apply the
    standard decoupled-weight-decay AdamW update with bias correction,
    keep moments in fp32, and round the final write back to the param's
    native dtype. This explicit reference avoids the ambiguity around
    PyTorch's ``torch.optim.AdamW`` keeping moments in the param dtype
    when fed bf16 params (a slightly different regime than the fused
    kernels ProTrain ships with).
    """

    import torch

    beta1, beta2 = betas
    bias_corr1 = 1.0 - beta1**step
    bias_corr2 = 1.0 - beta2**step
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad.detach().float()
        st = state.setdefault(id(p), {})
        if "m" not in st:
            st["m"] = torch.zeros_like(grad, dtype=torch.float32)
            st["v"] = torch.zeros_like(grad, dtype=torch.float32)
        m = st["m"]
        v = st["v"]
        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        m_hat = m / bias_corr1
        v_hat = v / bias_corr2
        p_data_f32 = p.data.float()
        # AdamW: decoupled weight decay scales the param BEFORE the
        # gradient step; both DeepSpeedCPUAdam and Apex FusedAdam use
        # this formulation in adamw_mode (vs. legacy Adam's coupled
        # version which folds wd into the gradient term).
        p_data_f32.mul_(1.0 - lr * weight_decay).addcdiv_(
            m_hat, v_hat.sqrt() + eps, value=-lr
        )
        p.data.copy_(p_data_f32.to(p.data.dtype))


def _normalize_keys(sd: dict) -> dict:
    """Strip the ``.block.`` infix that block wrappers introduce.

    OffloadedBlock / CheckpointedBlock / SwappedBlock all carry the
    wrapped module as ``self.block``, so a transformer block at
    ``model.layers.3`` becomes ``model.layers.3.block.<...>`` in the
    wrapped state_dict. Normalize so the comparison can match by
    semantic name.
    """

    out = {}
    for k, v in sd.items():
        out[k.replace(".block.", ".")] = v
    return out


def _build_init_state(seed: int):
    """Return ``(cfg, init_state)`` — the canonical CPU init state-dict.

    Both the vanilla and ProTrain runs load this state-dict so they
    start from byte-identical weights regardless of any RNG state
    consumed by the wrapper's profiler invocation.
    """

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2816,
        vocab_size=8000,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        torch_dtype="float32",
        use_cache=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return cfg, init_state


def _run_vanilla(
    cfg,
    init_state,
    input_ids,
    labels,
    *,
    n_iters: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
):
    """Build a fresh model, load init_state, run reference AdamW for n_iters."""

    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM(cfg)
    model.load_state_dict(init_state)
    model = model.bfloat16().to("cuda")
    state: dict = {}
    losses: list[float] = []
    for it in range(n_iters):
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_val = float(loss.detach().cpu())
        if not (loss_val == loss_val) or loss_val in (
            float("inf"),
            float("-inf"),
        ):
            raise AssertionError(f"vanilla iter {it}: non-finite loss ({loss_val})")
        losses.append(loss_val)
        model.zero_grad()
        loss.backward()
        _ref_adamw_step(
            list(model.parameters()),
            state,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            step=it + 1,
        )
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return losses, final_state


def _run_protrain(
    cfg,
    init_state,
    input_ids,
    labels,
    *,
    n_iters: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    force_persist: bool = False,
    overrides: dict | None = None,
):
    """Build a fresh model, load init_state, wrap with ProTrain, run for n_iters."""

    import torch
    from transformers import LlamaForCausalLM

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    model = LlamaForCausalLM(cfg)
    model.load_state_dict(init_state)
    model = model.bfloat16().to("cuda")

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )

    bs, seq = input_ids.shape
    kwargs: dict = dict(
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
    )
    if force_persist:
        kwargs["force_all_persistent"] = True
    if overrides is not None:
        kwargs.update(overrides)

    wrapped = protrain_model_wrapper(model, **kwargs)
    optim = protrain_optimizer_wrapper(
        wrapped, lr=lr, betas=betas, eps=eps, weight_decay=0.0
    )

    losses: list[float] = []
    for it in range(n_iters):
        out = wrapped.module(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_val = float(loss.detach().cpu())
        if not (loss_val == loss_val) or loss_val in (
            float("inf"),
            float("-inf"),
        ):
            raise AssertionError(f"protrain iter {it}: non-finite loss ({loss_val})")
        losses.append(loss_val)
        optim.zero_grad()
        loss.backward()
        optim.step()
    final_state = {
        k: v.detach().cpu().clone() for k, v in wrapped.module.state_dict().items()
    }
    return losses, final_state


def _compare(
    losses_v: list[float],
    losses_p: list[float],
    sd_v: dict,
    sd_p: dict,
    *,
    label: str,
    iter0_tol: float = 1e-3,
    iter_tol: float = 0.01,
    param_tol: float = 0.01,
) -> None:
    """Apply the math-equivalence acceptance bars and raise on regression."""

    assert len(losses_v) == len(losses_p), (
        f"[{label}] iter count mismatch: vanilla={len(losses_v)} "
        f"protrain={len(losses_p)}"
    )

    iter0_rel = abs(losses_v[0] - losses_p[0]) / max(abs(losses_v[0]), 1e-9)
    _mark(
        f"[{label}] iter 0 forward: vanilla={losses_v[0]:.6f} "
        f"protrain={losses_p[0]:.6f} rel={iter0_rel * 100:.4f}%"
    )
    assert iter0_rel <= iter0_tol, (
        f"[{label}] iter-0 forward divergence — vanilla={losses_v[0]:.6f} "
        f"protrain={losses_p[0]:.6f} rel={iter0_rel * 100:.4f}% "
        f"(tol={iter0_tol * 100:.4f}%). Forward path is changed by the wrapper "
        "(block reordering, dropout RNG mismatch, recompute dtype)."
    )

    for i, (lv, lp) in enumerate(zip(losses_v, losses_p, strict=True)):
        rel = abs(lv - lp) / max(abs(lv), 1e-9)
        _mark(
            f"[{label}] iter {i}: vanilla={lv:.6f} protrain={lp:.6f} "
            f"rel={rel * 100:.4f}%"
        )
        assert rel <= iter_tol, (
            f"[{label}] iter-{i} loss divergence — vanilla={lv:.6f} "
            f"protrain={lp:.6f} rel={rel * 100:.4f}% (tol={iter_tol * 100:.2f}%). "
            "Optimizer / backward / gather-release math regression."
        )

    sd_v_n = _normalize_keys(sd_v)
    sd_p_n = _normalize_keys(sd_p)
    missing = [k for k in sd_v_n if k not in sd_p_n]
    assert not missing, (
        f"[{label}] params missing from protrain state_dict: {missing[:5]}"
    )
    worst_name = ""
    worst_rel = 0.0
    for k in sd_v_n:
        v = sd_v_n[k].float()
        p = sd_p_n[k].float()
        if v.shape != p.shape:
            raise AssertionError(
                f"[{label}] param shape mismatch on {k}: v={v.shape} p={p.shape}"
            )
        diff = (v - p).norm().item()
        denom = max(v.norm().item(), 1e-9)
        rel = diff / denom
        if rel > worst_rel:
            worst_rel = rel
            worst_name = k
    _mark(f"[{label}] worst-param relerr: {worst_name} = {worst_rel * 100:.4f}%")
    assert worst_rel <= param_tol, (
        f"[{label}] final-param divergence — worst tensor {worst_name} "
        f"rel-err={worst_rel * 100:.4f}% (tol={param_tol * 100:.2f}%). "
        "Either the optimizer applied a different update or the gather/release "
        "lifecycle dropped a step."
    )


# Hyperparameters shared across the two configs. Kept as module-level
# constants so the configurations are easy to tweak in lockstep.
_N_ITERS = 6
_LR = 1e-3
_BETAS = (0.9, 0.999)
_EPS = 1e-8
_BATCH = 2
_SEQ = 64
_INIT_SEED = 42


@pytest.mark.slow
@pytest.mark.gpu
def test_math_equivalence_force_all_persistent() -> None:
    """ProTrain ≡ vanilla AdamW when nothing is offloaded.

    Exercises:
      * Block-wrap (Checkpointed for activations only when n_checkpoint>0;
        force_all_persistent picks n_checkpoint=N_block by default, see
        the model_wrapper docstring).
      * Chunk-pack data layout — params live inside the unified pool but
        stay GPU-resident for the duration.
      * GPU FusedAdam adapter (or torch.optim.AdamW fallback when Apex is
        unavailable; the reference loop accommodates both via fp32 moments).
    """

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    _mark("starting force_all_persistent variant")
    cfg, init_state = _build_init_state(_INIT_SEED)
    torch.manual_seed(7)
    input_ids = torch.randint(
        0, cfg.vocab_size, (_BATCH, _SEQ), device="cuda", dtype=torch.long
    )
    labels = input_ids.clone()

    _mark("running vanilla reference loop")
    losses_v, sd_v = _run_vanilla(
        cfg,
        init_state,
        input_ids,
        labels,
        n_iters=_N_ITERS,
        lr=_LR,
        betas=_BETAS,
        eps=_EPS,
    )
    _mark(f"vanilla losses: {[f'{x:.5f}' for x in losses_v]}")

    torch.cuda.empty_cache()

    _mark("running protrain force_all_persistent")
    losses_p, sd_p = _run_protrain(
        cfg,
        init_state,
        input_ids,
        labels,
        n_iters=_N_ITERS,
        lr=_LR,
        betas=_BETAS,
        eps=_EPS,
        force_persist=True,
    )
    _mark(f"protrain-persist losses: {[f'{x:.5f}' for x in losses_p]}")

    _compare(losses_v, losses_p, sd_v, sd_p, label="force_all_persistent")


@pytest.mark.slow
@pytest.mark.gpu
def test_math_equivalence_offload_and_checkpoint() -> None:
    """ProTrain ≡ vanilla AdamW when offload + recompute are both engaged.

    Picks a deterministic layout via the explicit-knob override path so
    every run exercises the same code paths regardless of host
    capacity:

      * ``n_persist=1`` (chunks): only the embedding/lm_head super-chunk
        stays GPU-resident; transformer-block chunks all live on CPU.
      * ``n_offload=2`` (blocks): the saved-tensors-hook re-gather path
        is active for half the blocks.
      * ``n_checkpoint=2`` (blocks): the recompute path is active for
        the other half.
      * ``n_swap=0``: SWAP is exercised by other tests; deliberately
        excluded here to keep the bisect surface small.
      * ``n_buffer=1``: minimum prefetch buffer.

    A failure here that does NOT also fail the persistent variant
    points at the CPU-Adam / gather-release / per-param grad-offload
    machinery; see the test docstrings on each adapter for the
    bisection trail.
    """

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    # CPU FusedAdam is the workhorse on this path — without it the
    # offloaded chunks never get an optimizer step (see the
    # ``CpuFusedAdamAdapter`` failure-mode comment in optim.py). On rigs
    # without DeepSpeed the construction raises and the wrapper translates
    # it into a RuntimeError; skip cleanly so this test isn't flaky on
    # systems without the kernel installed.
    pytest.importorskip("deepspeed")

    _mark("starting offload+ckpt variant")
    cfg, init_state = _build_init_state(_INIT_SEED)
    torch.manual_seed(7)
    input_ids = torch.randint(
        0, cfg.vocab_size, (_BATCH, _SEQ), device="cuda", dtype=torch.long
    )
    labels = input_ids.clone()

    _mark("running vanilla reference loop")
    losses_v, sd_v = _run_vanilla(
        cfg,
        init_state,
        input_ids,
        labels,
        n_iters=_N_ITERS,
        lr=_LR,
        betas=_BETAS,
        eps=_EPS,
    )
    _mark(f"vanilla losses: {[f'{x:.5f}' for x in losses_v]}")

    torch.cuda.empty_cache()

    n_block = cfg.num_hidden_layers
    overrides = dict(
        n_persist_override=1,
        n_buffer_override=1,
        n_swap_override=0,
        n_checkpoint_override=2,
        n_offload_override=n_block - 2,  # = 2 for a 4-block model
    )

    _mark(f"running protrain offload+ckpt overrides={overrides}")
    losses_p, sd_p = _run_protrain(
        cfg,
        init_state,
        input_ids,
        labels,
        n_iters=_N_ITERS,
        lr=_LR,
        betas=_BETAS,
        eps=_EPS,
        overrides=overrides,
    )
    _mark(f"protrain-offload+ckpt losses: {[f'{x:.5f}' for x in losses_p]}")

    _compare(losses_v, losses_p, sd_v, sd_p, label="offload+ckpt")
