"""Activation-aware ternary fit — the harness behind `init: ternary_fit_calibrated`.

Sequential, one decoder layer at a time (the GPTQ arrangement), so only a single
block's activations are ever resident:

1. run the embedding over the calibration sequences and capture the inputs of the
   first decoder layer, together with the kwargs its forward needs to be replayed,
2. for the layer being solved, accumulate the second-moment matrix `S = XᵀX` per
   distinct linear input (q/k/v share one, gate/up share one),
3. seed the codes with the data-free fit, then refit the scale against `S` in closed
   form (AGA) — the activation-weighted objective `‖(W - sT)X‖²` rather than the
   plain `‖W - sT‖²`, which is where most of the calibrated gain comes from,
4. walk the columns in `BLOCK_SIZE` blocks, quantizing each and compensating the
   residual error into the not-yet-quantized columns through the inverse Hessian
   (the GPTQ loop),
5. run the solved layer over the captured inputs to produce the next layer's inputs,
   so every layer is fit against the activations the *quantized* prefix produces.

Calibration data should be drawn from the healing corpus: a domain mismatch between
the calibration and training distributions is worth more perplexity than the whole
method buys.

The whole harness runs at λ = 0, which is not a shortcut: an unsolved layer still
holds FP latents and behaves as FP, while a solved one already holds `codes · s16`,
so the raw-latent forward *is* the quantized prefix the next layer must be fit
against.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .. import quant
from ..args import WeightScaleMode, resolve_ternary_config
from ..modules import SUBLN_ATTR, TernaryLinear, iter_ternary_modules
from . import fit_module_latent, iter_swapped, write_latent
from .ternary_fit import RIDGE_LAMBDA, fit_ternary, solve_2x2_ridge, solve_scale

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..swap import SwapManifest

LOG = get_logger(__name__)

# columns quantized per error-compensation block
BLOCK_SIZE: int = 128

# fraction of mean(diag(S)) added to the diagonal before the Cholesky; without it a
# rank-deficient S (fewer calibration tokens than input features) has no inverse
DAMPING: float = 0.01

# kwargs the decoder layer is replayed without: a cache would be filled twice
_SKIP_KWARGS: frozenset[str] = frozenset(
    {"past_key_value", "past_key_values", "use_cache"}
)

_CHOLESKY_RETRIES: int = 6
_DAMP_FLOOR: float = 1e-8
_TINY: float = 1e-30


class _StopForward(Exception):
    """Raised by the capture hook to abandon the forward once layer 0 is reached."""


class _Catcher(nn.Module):
    """Stands in for the first decoder layer and records what reaches it."""

    def __init__(self, layer: nn.Module, store: dict[str, Any]) -> None:
        """Wrap `layer`, writing captured inputs and kwargs into `store`."""
        super().__init__()
        self.layer = layer
        self._store = store

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Record `hidden_states` and the replay kwargs, then stop the forward."""
        self._store["inputs"].append(hidden_states.detach())
        self._store["kwargs"] = {
            key: value for key, value in kwargs.items() if key not in _SKIP_KWARGS
        }
        raise _StopForward


@quant.pinned_fp32_precision()
def calibrate_model_latents(
    model: "PreTrainedModel", manifest: "SwapManifest", cfg: DictDefault
) -> None:
    """Fit every swapped latent against calibration activations, layer by layer, in place.

    Args:
        model: The converted model, still whole and unwrapped.
        manifest: The swap manifest for `model`.
        cfg: The axolotl config; `ternary.init_calibration` supplies the corpus.

    Raises:
        ValueError: If the architecture exposes no decoder-layer list to walk, or the
            calibration dataset cannot be resolved.
    """
    calibration = resolve_ternary_config(cfg).init_calibration
    layers = decoder_layers(model)
    pending = {
        id(module): (entry, module)
        for entry, module in iter_swapped(model, manifest)
        if not module.is_baked()
    }
    if not pending:
        return

    inputs, kwargs = collect_layer_inputs(
        model,
        cfg,
        calibration.num_sequences,
        calibration.sequence_len,
        calibration.dataset,
    )
    solved = 0
    with _calibration_mode(model), torch.no_grad():
        for layer in layers:
            members = {id(child) for child in layer.modules()}
            targets = [item for key, item in pending.items() if key in members]
            if targets:
                grams = _layer_grams(
                    [module for _, module in targets], layer, inputs, kwargs
                )
                for entry, module in targets:
                    gram = grams.get(id(module))
                    if gram is None:
                        LOG.warning(
                            f"ternary: {entry.name} was not called by its own decoder "
                            "layer over the calibration sequences, so it has no "
                            "activations to fit against; falling back to the data-free "
                            "fit for it"
                        )
                        fit_module_latent(module, entry)
                    else:
                        codes, scale = gptq_compensate(
                            module.weight.detach(),
                            gram,
                            scale_mode=entry.weight_scale,  # type: ignore[arg-type]
                            group_size=entry.group_size,
                        )
                        write_latent(module, codes, scale)
                    pending.pop(id(module))
                    solved += 1
            # rewritten in place: a second list would double the resident activations
            for index, hidden in enumerate(inputs):
                inputs[index] = _run_layer(layer, hidden, kwargs)

    for entry, module in pending.values():
        LOG.warning(
            f"ternary: {entry.name} is not inside a decoder layer; falling back to the "
            "data-free fit for it"
        )
        fit_module_latent(module, entry)
    LOG.info(
        f"ternary: calibrated {solved} latents over {len(inputs)} sequences of "
        f"{calibration.sequence_len} tokens"
    )


def collect_layer_inputs(
    model: "PreTrainedModel",
    cfg: DictDefault,
    num_sequences: int,
    sequence_len: int,
    dataset: str | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Capture the hidden states entering the first decoder layer.

    Args:
        model: The model to run the embedding (and any pre-block modules) of.
        cfg: The axolotl config, used to build the calibration dataloader.
        num_sequences: How many sequences to capture.
        sequence_len: Tokens per sequence; longer samples are truncated.
        dataset: Dataset path to calibrate on; `None` uses the training dataset.
        input_ids: Pre-tokenized calibration tokens, chopped into `sequence_len`
            chunks instead of loading a dataset.

    Returns:
        `(inputs, forward_kwargs)` — one `(batch, sequence_len, hidden)` tensor per
        captured batch, and the kwargs (attention mask, position embeddings, cache
        position) every decoder layer needs to be replayed on them.

    Raises:
        ValueError: If the corpus yields no full-length sequence.
    """
    layers = decoder_layers(model)
    store: dict[str, Any] = {"inputs": [], "kwargs": {}}
    original = layers[0]
    layers[0] = _Catcher(original, store)
    device = next(model.parameters()).device
    try:
        with _calibration_mode(model), torch.no_grad():
            for batch in _token_batches(
                cfg, num_sequences, sequence_len, dataset, input_ids
            ):
                try:
                    model(input_ids=batch.to(device), use_cache=False)
                except _StopForward:
                    continue
    finally:
        layers[0] = original

    if not store["inputs"]:
        raise ValueError(
            "ternary: the calibration corpus yielded no sequence of "
            f"{sequence_len} tokens; lower ternary.init_calibration.sequence_len or "
            "point init_calibration.dataset at a larger corpus"
        )
    return store["inputs"], store["kwargs"]


def accumulate_gram(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return `S = Σ XᵀX` in fp32 over the calibration activations of one linear.

    Args:
        inputs: Activations of shape `(..., in_features)`; every leading dimension
            is a sample.

    Returns:
        fp32 `(in_features, in_features)` second-moment matrix.

    Raises:
        ValueError: If `inputs` is empty.
    """
    if len(inputs) == 0:
        raise ValueError("accumulate_gram needs at least one activation tensor")
    gram: torch.Tensor | None = None
    for tensor in inputs:
        flat = tensor.detach().reshape(-1, tensor.shape[-1]).to(torch.float32)
        outer = flat.T @ flat
        gram = outer if gram is None else gram + outer
    return gram  # type: ignore[return-value]


def aga_refit(
    weight: torch.Tensor,
    codes: torch.Tensor,
    gram: torch.Tensor,
    scale_mode: WeightScaleMode = "absmean",
    group_size: int | None = None,
    damping: float = DAMPING,
) -> torch.Tensor:
    """Refit the scale against the activation-weighted objective, in closed form.

    Minimizes `‖(W - sT)X‖²` for fixed codes: per row `s = (wᵢ S tᵢᵀ) / (tᵢ S tᵢᵀ)`,
    summed over rows for a per-tensor scale, restricted to the columns of a block for
    a per-group one.

    `S` is damped the way `gptq_compensate` damps it before the Cholesky, which is not
    cosmetic: fewer calibration tokens than input features (or a low-rank activation
    subspace) leaves whole directions unconstrained, and the objective then says
    nothing about the scale along them. The damping adds back `λ‖W - sT‖²`, so an
    unconstrained direction falls to the data-free answer instead of an arbitrary one.

    Args:
        weight: Latent weight of shape `(out_features, in_features)`.
        codes: Codes from the data-free fit, shaped like `weight` (or `(2, ...)` for
            `dual`).
        gram: `S = XᵀX` for this linear's input.
        scale_mode: Which unit shares a scale.
        group_size: Block size for `scale_mode: group`.
        damping: Fraction of `mean(diag(S))` added to the diagonal before solving.

    Returns:
        fp32 scales in the same layout `fit_ternary` returns for `scale_mode`. Under
        `dual` the pair stays matched to the planes positionally, and a row whose
        solve is unusable is taken from the data-free fit *whole*: substituting one
        component would put one plane's scale on the other plane's weights.

    Raises:
        ValueError: If `gram` does not match the weight's input dimension.
    """
    dense = weight.detach().to(torch.float32)
    second_moment = _damped(_validated_gram(dense, gram), damping)
    planes = codes.detach().to(torch.float32)
    fallback = solve_scale(dense, codes, scale_mode, group_size)

    if scale_mode == "dual":
        low, high = planes[0], planes[1]
        low_s, high_s, weight_s = (
            low @ second_moment,
            high @ second_moment,
            dense @ second_moment,
        )
        scales = solve_2x2_ridge(
            (low_s * low).sum(-1),
            (low_s * high).sum(-1),
            (high_s * high).sum(-1),
            (weight_s * low).sum(-1),
            (weight_s * high).sum(-1),
            RIDGE_LAMBDA,
        )
        return _guarded_rows(scales, fallback)

    if scale_mode == "group":
        assert group_size is not None
        scales = torch.empty_like(fallback)
        for index in range(fallback.shape[-1]):
            block = slice(index * group_size, (index + 1) * group_size)
            sub = second_moment[block, block]
            weight_block, code_block = dense[:, block], planes[:, block]
            scales[:, index] = _ratio(
                ((weight_block @ sub) * code_block).sum(-1),
                ((code_block @ sub) * code_block).sum(-1),
            )
        return _guarded(scales, fallback)

    numerator = ((dense @ second_moment) * planes).sum(-1)
    denominator = ((planes @ second_moment) * planes).sum(-1)
    if scale_mode == "learnable_row":
        return _guarded(_ratio(numerator, denominator).unsqueeze(-1), fallback)
    return _guarded(_ratio(numerator.sum(), denominator.sum()), fallback)


def gptq_compensate(
    weight: torch.Tensor,
    gram: torch.Tensor,
    scale_mode: WeightScaleMode = "absmean",
    group_size: int | None = None,
    block_size: int = BLOCK_SIZE,
    damping: float = DAMPING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `weight` column-block by column-block, compensating the error as it goes.

    Args:
        weight: Latent weight of shape `(out_features, in_features)`.
        gram: `S = XᵀX` for this linear's input.
        scale_mode: Which unit shares a scale.
        group_size: Block size for `scale_mode: group`.
        block_size: Columns quantized before the residual is pushed forward.
        damping: Fraction of `mean(diag(S))` added to the diagonal for stability.

    Returns:
        `(codes, scales)` in the same layout `fit_ternary` returns.
    """
    dense = weight.detach().to(torch.float32)
    second_moment = _validated_gram(dense, gram)
    codes, _ = fit_ternary(dense, scale_mode, group_size)
    scale = aga_refit(dense, codes, second_moment, scale_mode, group_size, damping)
    if scale_mode == "dual":
        # the five-value grid has no single-column rounding rule to compensate against
        return codes, scale

    inverse = _inverse_cholesky(second_moment, damping)
    work = dense.clone()
    fitted = torch.zeros_like(dense)
    for start in range(0, dense.shape[1], block_size):
        stop = min(start + block_size, dense.shape[1])
        block = work[:, start:stop].clone()
        if _refits_block(scale_mode, group_size, start, stop):
            assert group_size is not None
            block_codes, _ = fit_ternary(block, "group", group_size)
            scale[:, start // group_size : stop // group_size] = aga_refit(
                block,
                block_codes,
                second_moment[start:stop, start:stop],
                "group",
                group_size,
                damping,
            )
        chol = inverse[start:stop, start:stop]
        errors = torch.zeros_like(block)
        for offset in range(stop - start):
            column = block[:, offset : offset + 1]
            column_scale = _column_scale(scale, start + offset, scale_mode, group_size)
            column_codes = quant.ternary_codes(column, column_scale)
            quantized = column_codes.to(torch.float32) * quant.f16_round_scale(
                column_scale
            )
            error = (column - quantized) / chol[offset, offset]
            block[:, offset + 1 :] -= error * chol[offset, offset + 1 :]
            errors[:, offset : offset + 1] = error
            fitted[:, start + offset] = column_codes.reshape(-1)
        work[:, stop:] -= errors @ inverse[start:stop, stop:]
    return fitted.to(torch.int8), scale


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the `ModuleList` of decoder blocks the sequential solve walks.

    Raises:
        ValueError: If no block list holding ternary modules can be found.
    """
    for _, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        if any(isinstance(child, TernaryLinear) for child in module.modules()):
            return module
    raise ValueError(
        "ternary: init: ternary_fit_calibrated walks the decoder blocks one at a time, "
        f"but {type(model).__name__} exposes no ModuleList of blocks holding ternary "
        "modules; use init: ternary_fit"
    )


def _token_batches(
    cfg: DictDefault,
    num_sequences: int,
    sequence_len: int,
    dataset: str | None,
    input_ids: torch.Tensor | None,
) -> Iterator[torch.Tensor]:
    """Yield `(1, sequence_len)` token batches from `input_ids` or the configured corpus."""
    stream = (
        _dataset_tokens(cfg, dataset)
        if input_ids is None
        else iter([input_ids.reshape(-1).tolist()])
    )
    buffer: list[int] = []
    emitted = 0
    for tokens in stream:
        buffer.extend(tokens)
        while len(buffer) >= sequence_len and emitted < num_sequences:
            chunk, buffer = buffer[:sequence_len], buffer[sequence_len:]
            emitted += 1
            yield torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
        if emitted >= num_sequences:
            return


def _dataset_tokens(cfg: DictDefault, dataset: str | None) -> Iterator[list[int]]:
    """Yield the tokenized `input_ids` of the calibration corpus, example by example."""
    from axolotl.loaders import load_tokenizer
    from axolotl.utils.data.sft import prepare_datasets

    calibration_cfg = cfg
    if dataset is not None:
        calibration_cfg = DictDefault(
            {
                **cfg,
                "datasets": [{"path": dataset, "type": "completion"}],
                "test_datasets": None,
                "val_set_size": 0,
                # `prepare_datasets` hands either of these the whole job and never
                # looks at `datasets`, which would drop the requested corpus silently
                "pretraining_dataset": None,
                "streaming": False,
            }
        )
    tokenizer = load_tokenizer(calibration_cfg)
    train_dataset = prepare_datasets(calibration_cfg, tokenizer)[0]
    for example in train_dataset:
        tokens = example.get("input_ids")
        if tokens is not None:
            yield list(tokens)


@contextmanager
def _calibration_mode(model: nn.Module) -> Iterator[None]:
    """Put `model` in eval mode with every ternary module at λ = 0, then restore both."""
    training = model.training
    lambdas = [(module, module.lambda_) for _, module in iter_ternary_modules(model)]
    model.eval()
    for module, _ in lambdas:
        module.set_lambda(0.0)
    try:
        yield
    finally:
        for module, lambda_ in lambdas:
            module.set_lambda(lambda_)
        model.train(training)


def _layer_grams(
    modules: Sequence[TernaryLinear],
    layer: nn.Module,
    inputs: Sequence[torch.Tensor],
    kwargs: dict[str, Any],
) -> dict[int, torch.Tensor]:
    """Accumulate `S = XᵀX` for every module in `layer`, sharing one per distinct input."""
    grams: dict[int, torch.Tensor] = {}
    owners: dict[int, int] = {}
    live: dict[int, int] = {}
    holds: list[torch.Tensor] = []
    done: set[int] = set()

    def _hook(module: nn.Module, args: tuple) -> None:
        activation = args[0]
        norm = getattr(module, SUBLN_ATTR, None)
        if norm is not None:
            activation = norm(activation)
            owner = id(module)
        else:
            owner = live.get(id(activation), id(module))
            if id(activation) not in live:
                live[id(activation)] = owner
                # holding a reference keeps the id unique for the whole batch
                holds.append(activation)
        owners[id(module)] = owner
        if owner in done:
            return
        done.add(owner)
        flat = activation.detach().reshape(-1, activation.shape[-1]).to(torch.float32)
        outer = flat.T @ flat
        grams[owner] = outer if owner not in grams else grams[owner] + outer

    handles = [module.register_forward_pre_hook(_hook) for module in modules]
    try:
        for hidden in inputs:
            live.clear()
            holds.clear()
            done.clear()
            _run_layer(layer, hidden, kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return {key: grams[owner] for key, owner in owners.items()}


def _run_layer(
    layer: nn.Module, hidden: torch.Tensor, kwargs: dict[str, Any]
) -> torch.Tensor:
    output = layer(hidden, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def _validated_gram(weight: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    in_features = weight.shape[-1]
    if gram.shape != (in_features, in_features):
        raise ValueError(
            f"gram must be ({in_features}, {in_features}) for a weight with "
            f"{in_features} input features, got {tuple(gram.shape)}"
        )
    return gram.detach().to(torch.float32)


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / torch.where(
        denominator > 0, denominator, torch.full_like(denominator, _TINY)
    )


def _guarded(scales: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Keep the data-free scale wherever the activation-weighted solve is unusable."""
    usable = torch.isfinite(scales) & (scales > 0)
    return torch.where(usable, scales, fallback).clamp_min(quant.SCALE_EPS)


def _guarded_rows(scales: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """`_guarded`, all-or-nothing per row — for scale layouts whose entries are coupled."""
    usable = (torch.isfinite(scales) & (scales > 0)).all(-1, keepdim=True)
    return torch.where(usable, scales, fallback).clamp_min(quant.SCALE_EPS)


def _damped(gram: torch.Tensor, damping: float) -> torch.Tensor:
    """Return `S + λ·mean(diag(S))·I`, the matrix every solve here actually uses."""
    penalty = float(
        (damping * gram.diagonal().mean()).clamp_min(_DAMP_FLOOR)  # type: ignore[arg-type]
    )
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return gram + penalty * identity


def _refits_block(
    scale_mode: str, group_size: int | None, start: int, stop: int
) -> bool:
    return (
        scale_mode == "group"
        and group_size is not None
        and start % group_size == 0
        and (stop - start) % group_size == 0
    )


def _column_scale(
    scale: torch.Tensor, column: int, scale_mode: str, group_size: int | None
) -> torch.Tensor:
    if scale_mode == "group":
        assert group_size is not None
        index = column // group_size
        return scale[:, index : index + 1]
    return scale


def _inverse_cholesky(gram: torch.Tensor, damping: float) -> torch.Tensor:
    """Return the upper Cholesky factor of `(S + λI)⁻¹` — GPTQ's compensation matrix.

    Raises:
        ValueError: If the damped Gram matrix stays indefinite after every retry.
    """
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    penalty = float(
        (damping * gram.diagonal().mean()).clamp_min(_DAMP_FLOOR)  # type: ignore[arg-type]
    )
    for _ in range(_CHOLESKY_RETRIES):
        try:
            factor = torch.linalg.cholesky(gram + penalty * identity)
            return torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
        except RuntimeError:
            penalty *= 10.0
    raise ValueError(
        "ternary: the calibration Gram matrix is not positive definite even after "
        "damping; calibrate on more sequences or raise the damping"
    )


__all__ = [
    "BLOCK_SIZE",
    "DAMPING",
    "accumulate_gram",
    "aga_refit",
    "calibrate_model_latents",
    "collect_layer_inputs",
    "decoder_layers",
    "gptq_compensate",
]
