"""Damage map: what quantization costs before a single training step is spent.

Eval-only passes over a small split, each toggling one part of the quantizer on a
freshly swapped (untrained) model:

- `weights` — ternary weights, full-precision activations,
- `activations` — FP weights, per-token int8 activations,
- `both` — the configured quantizer,
- one row per module family (`q_proj`, `down_proj`, …) — that family ternary, the
  rest full precision.

The shape of the table decides the recipe: activation-dominant damage argues for
smoothing, weight-dominant damage for a fitted init and a longer λ warmup, and a
single outlier family argues for keeping it full precision.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

    from ..modules import TernaryLinear

LOG = get_logger(__name__)

DEFAULT_SEQUENCES: int = 64
DEFAULT_SEQUENCE_LEN: int = 1024

# scopes measured before the per-family rows
GLOBAL_SCOPES: tuple[str, ...] = ("weights", "activations", "both")

FAMILY_SCOPE_PREFIX: str = "family:"

# record fields a raw-text corpus is read from, in order
TEXT_FIELDS: tuple[str, ...] = ("text", "content", "raw_content")

# perplexity above this is reported as infinite rather than overflowing `exp`
_MAX_NLL: float = 700.0


@dataclass(frozen=True)
class DamageRow:
    """One eval-only pass: what it quantized and what that cost.

    Attributes:
        scope: `weights`, `activations`, `both`, or `family:<name>`.
        perplexity: Perplexity of that pass on the split.
        delta: `perplexity` minus the full-precision baseline.
    """

    scope: str
    perplexity: float
    delta: float


@dataclass
class DamageReport:
    """The full table plus the baseline it is measured against."""

    baseline_perplexity: float
    rows: list[DamageRow] = field(default_factory=list)
    sequences: int = DEFAULT_SEQUENCES
    sequence_len: int = DEFAULT_SEQUENCE_LEN

    def to_dict(self) -> dict:
        """Return a JSON-serializable view of the report."""
        return {
            "baseline_perplexity": self.baseline_perplexity,
            "sequences": self.sequences,
            "sequence_len": self.sequence_len,
            "rows": [
                {
                    "scope": row.scope,
                    "perplexity": row.perplexity,
                    "delta": row.delta,
                }
                for row in self.rows
            ],
        }

    def to_table(self) -> str:
        """Render the report as a fixed-width text table for the CLI."""
        width = max([len("baseline"), *(len(row.scope) for row in self.rows)])
        header = f"{'scope':<{width}}  {'perplexity':>12}  {'delta':>12}"
        lines = [
            f"{self.sequences} sequences x {self.sequence_len} tokens",
            "",
            header,
            "-" * len(header),
            f"{'baseline':<{width}}  {self.baseline_perplexity:>12.4f}  {'':>12}",
        ]
        lines.extend(
            f"{row.scope:<{width}}  {row.perplexity:>12.4f}  {row.delta:>+12.4f}"
            for row in self.rows
        )
        return "\n".join(lines)


def damage_map(
    cfg: DictDefault,
    num_sequences: int = DEFAULT_SEQUENCES,
    sequence_len: int = DEFAULT_SEQUENCE_LEN,
    dataset: str | None = None,
) -> DamageReport:
    """Measure the perplexity cost of each part of the quantizer, without training.

    Args:
        cfg: The axolotl config; the model, the ternary block and the dataset are
            read from it exactly as a training run would.
        num_sequences: Sequences evaluated per pass.
        sequence_len: Tokens per sequence.
        dataset: Dataset path to evaluate on; `None` uses the training dataset.

    Returns:
        The report, one row per pass, deltas relative to the FP baseline.

    Raises:
        ValueError: If no dataset can be resolved, the corpus is too short, or the
            config swaps no linear at all.
    """
    from ..modules import iter_ternary_modules
    from ..swap import convert_model

    model = load_eval_model(cfg)
    batches = collect_eval_batches(cfg, num_sequences, sequence_len, dataset)

    convert_model(model, _swap_cfg(cfg))
    modules = dict(iter_ternary_modules(model))
    if not modules:
        raise ValueError(
            "the damage map needs at least one swapped linear; ternary.target_modules "
            "matched nothing in this model"
        )
    configured_bits = {name: module.activation_bits for name, module in modules.items()}

    with _scope(modules, "baseline", configured_bits):
        baseline = perplexity(model, batches)
    rows: list[DamageRow] = []
    for scope in (*GLOBAL_SCOPES, *_family_scopes(modules)):
        with _scope(modules, scope, configured_bits):
            value = perplexity(model, batches)
        rows.append(DamageRow(scope=scope, perplexity=value, delta=value - baseline))
        LOG.info(
            f"ternary damage map: {scope} ppl {value:.4f} ({value - baseline:+.4f})"
        )

    return DamageReport(
        baseline_perplexity=baseline,
        rows=rows,
        sequences=sum(int(batch.shape[0]) for batch in batches),
        sequence_len=sequence_len,
    )


def load_eval_model(cfg: DictDefault) -> "PreTrainedModel":
    """Load the unquantized `base_model` for the eval passes, on the best local device."""
    import torch
    from transformers import AutoModelForCausalLM

    base_model = cfg.get("base_model")
    if not base_model:
        raise ValueError("the damage map needs `base_model:` in the config")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=bool(cfg.get("trust_remote_code")),
    )
    model.eval()
    return model


def collect_eval_batches(
    cfg: DictDefault,
    num_sequences: int,
    sequence_len: int,
    dataset: str | None = None,
) -> list["torch.Tensor"]:
    """Tokenize a raw-text split into fixed-length evaluation sequences.

    Args:
        cfg: The axolotl config; its tokenizer and (unless `dataset` overrides it)
            its first configured dataset are used.
        num_sequences: How many sequences to return.
        sequence_len: Tokens per sequence.
        dataset: Dataset path to evaluate on; `None` uses the training dataset.

    Returns:
        One `(1, sequence_len)` int64 tensor per sequence.

    Raises:
        ValueError: If no dataset is configured, or it yields too little text.
    """
    import torch
    from datasets import load_dataset

    from axolotl.loaders import load_tokenizer

    path, name, split = _resolve_dataset(cfg, dataset)
    tokenizer = load_tokenizer(cfg)
    stream = load_dataset(path, name, split=split, streaming=True)

    needed = num_sequences * sequence_len
    ids: list[int] = []
    fields: set[str] = set()
    for record in stream:
        text = _record_text(record)
        if not text:
            if hasattr(record, "keys"):
                fields.update(record.keys())
            continue
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        if len(ids) >= needed:
            break

    usable = min(num_sequences, len(ids) // sequence_len)
    if not usable and not ids:
        raise ValueError(
            f"no record of {path} carries any of the raw-text fields {list(TEXT_FIELDS)} "
            f"(it has {sorted(fields)}); the damage map tokenizes a raw-text corpus "
            "rather than applying a prompt strategy, so point it at one with "
            "`axolotl ternary damage-map --dataset`"
        )
    if not usable:
        raise ValueError(
            f"{path} yielded {len(ids)} tokens, too few for one {sequence_len}-token "
            "sequence; lower ternary damage-map --sequence-len or pick another dataset"
        )
    if usable < num_sequences:
        LOG.warning(
            f"ternary damage map: {path} yielded {usable} of {num_sequences} requested "
            "sequences"
        )
    tokens = torch.tensor(ids[: usable * sequence_len], dtype=torch.long)
    return list(tokens.reshape(usable, sequence_len).split(1))


def perplexity(model: "PreTrainedModel", batches: list["torch.Tensor"]) -> float:
    """Return token-weighted perplexity of `model` over `batches` of input ids."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            ids = batch.to(device)
            logits = model(input_ids=ids).logits.float()
            labels = ids[:, 1:].reshape(-1)
            total_nll += float(
                F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    labels,
                    reduction="sum",
                )
            )
            total_tokens += labels.numel()
    model.train(was_training)
    if not total_tokens:
        raise ValueError("the damage map was handed no evaluation tokens")
    mean_nll = total_nll / total_tokens
    return math.inf if mean_nll > _MAX_NLL else math.exp(mean_nll)


@contextmanager
def _scope(
    modules: dict[str, "TernaryLinear"],
    scope: str,
    configured_bits: dict[str, int | None],
) -> Iterator[None]:
    """Configure every swapped module for one pass, then put it back."""
    handles = []
    weights_only = scope == "weights" or scope.startswith(FAMILY_SCOPE_PREFIX)
    try:
        for name, module in modules.items():
            module.activation_bits = None if weights_only else configured_bits[name]
            module.set_lambda(_module_lambda(name, scope))
        if scope == "activations":
            # λ=0 short-circuits the module's own activation quantizer, so the pass
            # measures it from the outside on otherwise untouched FP linears
            hook = _act_quant_hook()
            handles = [
                module.register_forward_pre_hook(hook) for module in modules.values()
            ]
        yield
    finally:
        for handle in handles:
            handle.remove()
        for name, module in modules.items():
            module.activation_bits = configured_bits[name]
            module.set_lambda(0.0)


def _act_quant_hook():
    from .. import quant

    def hook(_module, args):
        return (quant.act_quant(args[0], 1.0),) + tuple(args[1:])

    return hook


def _module_lambda(name: str, scope: str) -> float:
    if scope in ("baseline", "activations"):
        return 0.0
    if scope.startswith(FAMILY_SCOPE_PREFIX):
        from ..swap import module_family

        return float(module_family(name) == scope[len(FAMILY_SCOPE_PREFIX) :])
    return 1.0


def _family_scopes(modules: dict[str, "TernaryLinear"]) -> list[str]:
    from ..swap import module_family

    families = dict.fromkeys(module_family(name) for name in modules)
    return [f"{FAMILY_SCOPE_PREFIX}{family}" for family in families]


def _swap_cfg(cfg: DictDefault) -> DictDefault:
    """Return `cfg` with everything that is not function-preserving at λ=0 turned off.

    The baseline pass is the swapped model at λ=0, so the swap has to be the identity
    there or every delta in the table is measured against an already-damaged model. A
    fitting `init` rewrites the latents to their ternary reconstruction and `subln`
    inserts a norm that rescales its input at unit gain — both would make the table
    report that the quantizer costs nothing.
    """
    # the damage map is a diagnostic, not a run: nothing of it belongs in output_dir
    data = dict(cfg)
    data["output_dir"] = None
    ternary = dict(data.get("ternary") or {})
    neutralized = [
        key
        for key, neutral in (("init", "absmean"), ("subln", False))
        if ternary.get(key) not in (None, neutral)
    ]
    if neutralized:
        LOG.info(
            f"ternary damage map: ignoring {neutralized} for these passes; the table "
            "measures the quantizer against the untouched FP weights, and neither is "
            "function-preserving at lambda=0"
        )
    ternary["init"] = "absmean"
    ternary["subln"] = False
    ternary.pop("init_calibration", None)
    data["ternary"] = ternary
    return DictDefault(data)


def _resolve_dataset(
    cfg: DictDefault, dataset: str | None
) -> tuple[str, str | None, str]:
    if dataset:
        return dataset, None, "train"
    for key in ("pretraining_dataset", "datasets"):
        entries = cfg.get(key) or []
        for entry in entries:
            path = entry.get("path") if hasattr(entry, "get") else None
            if path:
                return path, entry.get("name"), entry.get("split") or "train"
    raise ValueError(
        "the damage map needs a corpus: configure `datasets:`/`pretraining_dataset:` "
        "or pass `axolotl ternary damage-map --dataset`"
    )


def _record_text(record: Any) -> str | None:
    for field_name in TEXT_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


__all__ = [
    "DEFAULT_SEQUENCES",
    "DEFAULT_SEQUENCE_LEN",
    "GLOBAL_SCOPES",
    "DamageReport",
    "DamageRow",
    "collect_eval_batches",
    "damage_map",
    "load_eval_model",
    "perplexity",
]
