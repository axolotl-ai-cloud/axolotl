"""How async GRPO packs a recurrent model's rows: prefix-shared trees, or chains.

TRL packs each rank's rollouts into a prefix trie. The ``tree-recurrence`` package
makes transformers' recurrent mixers follow it (Gated DeltaNet, Mamba2, LFM2's
short convs) through hooks ``packing_support`` installs. Axolotl adds:

- the KDA families of its in-tree Kimi-Linear and Bailing models, whose own
  forwards follow trees;
- one chain per sample for the recurrent models the package does not hook but
  axolotl's packed-sequence boundary patches cover (Mamba, Falcon-Mamba, Falcon-H1,
  Qwen4-exp): their mixers reset where each sample's position_ids restart.
"""

from __future__ import annotations

from tree_recurrence import Support, TreeFamily, families, install, register_family

from axolotl.monkeypatch.models.recurrent_support import PACKING_PATCHED
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

for _name, _model_types, _mixer in (
    ("kimi_delta_attention", ("kimi_linear",), "KimiDeltaAttention"),
    (
        "bailing_kimi_delta_attention",
        ("bailing_hybrid",),
        "BailingMoeV3KimiDeltaAttention",
    ),
):
    register_family(
        TreeFamily(_name, _model_types, (_mixer,), layer_types=("linear_attention",))
    )


def hooked_model_types() -> frozenset[str]:
    """Model types whose mixers follow trees through the package's hooks.

    Their stock forwards carry the trie to the hooked kernels, so axolotl's packing
    patches, which replace those forwards, stay off for async GRPO.
    """
    return frozenset(
        model_type
        for family in families()
        if family.install is not None
        for model_type in family.model_types
    )


def _model_types(model) -> set[str]:
    config = model.config
    return {config.model_type, config.get_text_config().model_type}


def packing_support(model) -> Support:
    """Make ``model``'s recurrent mixers follow trees, and say how rows may pack.

    Idempotent: the PatchManager calls it once the model is loaded, and the trainer
    again for the answer.
    """
    support = install(model)
    if support.mode != "none":
        return support
    covered = {t for family in families() for t in family.model_types}
    chained = PACKING_PATCHED - covered
    if _model_types(model) & chained:
        return Support(
            "chains",
            "axolotl's packed-sequence boundary patch keeps whole samples apart",
        )
    raise ValueError(
        f"Async GRPO packs rollouts into rows, but {model.config.model_type} "
        f"cannot keep its packed samples apart: {support.reason}. Train it with "
        "synchronous GRPO (without `trl.async_prefetch`)."
    )
