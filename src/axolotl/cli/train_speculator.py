"""CLI to train an EAGLE-3 speculator (draft model) via TorchSpec."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PLUGIN = "axolotl.integrations.torchspec.TorchSpecPlugin"


def _load_speculator_cfg(config: Union[Path, str], **kwargs) -> DictDefault:
    """Load + validate the axolotl config with the TorchSpec plugin registered.

    The plugin must be present in ``plugins:`` before ``load_cfg`` validates, or
    the ``speculator:`` block is silently dropped by the pydantic schema. We
    inject it here so users don't have to remember to list it.
    """
    with open(config, encoding="utf-8") as file:
        raw = DictDefault(yaml.safe_load(file))

    plugins = list(raw.get("plugins") or [])
    if _PLUGIN not in plugins:
        plugins.append(_PLUGIN)
    raw["plugins"] = plugins

    return load_cfg(raw, **kwargs)


def do_train_speculator(
    config: Union[Path, str],
    dry_run: bool = False,
    extra_overrides: list[str] | None = None,
    **kwargs,
) -> None:
    """Translate the axolotl config to TorchSpec args and launch training.

    Args:
        config: path to the axolotl config YAML.
        dry_run: print the translated TorchSpec config and exit (no Ray/GPUs).
        extra_overrides: OmegaConf dotlist overrides forwarded to TorchSpec, e.g.
            ``["training.num_train_steps=10"]``.
        kwargs: flat overrides applied to the axolotl config (CLI passthrough).
    """
    from axolotl.integrations.torchspec.translate import build_overrides

    cfg = _load_speculator_cfg(config, **kwargs)

    if cfg.get("speculator") is None:
        raise ValueError(
            "No `speculator:` block found after validation. Ensure the config "
            "includes a `speculator:` section (see "
            "examples/speculators/qwen3-8b-eagle3.yaml)."
        )

    if dry_run:
        import json

        # Stay side-effect-free: show the pure config mapping only. The dataset
        # standardization bridge and full flat-args resolution run at real launch.
        LOG.info(
            "TorchSpec config overrides (dry-run, no dataset I/O):\n%s",
            json.dumps(build_overrides(cfg), indent=2, default=str),
        )
        spec = cfg.get("speculator")
        prepare = getattr(spec, "prepare_dataset", None)
        if prepare is None and isinstance(spec, dict):
            prepare = spec.get("prepare_dataset", True)
        if prepare:
            LOG.info(
                "speculator.prepare_dataset=true: at launch, `datasets` are "
                "standardized to <output_dir>/torchspec_data/train.jsonl and "
                "train_data_path is repointed there."
            )
        return

    from axolotl.integrations.torchspec.translate import build_torchspec_args

    args = build_torchspec_args(cfg, extra_overrides=extra_overrides)

    from torchspec.train_entry import train_async_no_generation

    LOG.info(
        "Launching TorchSpec EAGLE-3 training for target=%s "
        "(%s inference + %s training GPUs).",
        args.target_model_path,
        args.inference_num_gpus,
        args.training_num_gpus_per_node * args.training_num_nodes,
    )
    train_async_no_generation(args)
