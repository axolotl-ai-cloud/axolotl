"""CLI to run training on a model."""

import gc
import os
from pathlib import Path
from typing import Any, Union

import fire

from axolotl.cli.args import TrainerCliArgs
from axolotl.utils import make_lazy_getattr
from axolotl.utils.dict import DictDefault

_LAZY_IMPORTS = {
    "Accelerator": "accelerate",
    "HfArgumentParser": "transformers.hf_argparser",
    "gpu_capabilities": "axolotl.cli.config",
    "load_cfg": "axolotl.cli.config",
    "load_datasets": "axolotl.common.datasets",
    "load_preference_datasets": "axolotl.common.datasets",
    "normalize_config": "axolotl.utils.config",
    "plugin_set_cfg": "axolotl.cli.config",
    "prepare_optim_env": "axolotl.utils.trainer",
    "prepare_plugins": "axolotl.cli.config",
    "resolve_dtype": "axolotl.utils.config",
    "train": "axolotl.train",
    "validate_config": "axolotl.utils.config",
}

__getattr__ = make_lazy_getattr(_LAZY_IMPORTS, __name__, globals())


def _lazy_attr(name: str) -> Any:
    return globals().get(name) or __getattr__(name)


def do_train(cfg: DictDefault, cli_args: TrainerCliArgs):
    """
    Trains a `transformers` model by first loading the dataset(s) specified in the
    `axolotl` config, and then calling `axolotl.train.train`. Also runs the plugin
    manager's `post_train_unload` once training completes.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Training-specific CLI arguments.
    """
    from axolotl.cli.checks import check_accelerate_default_config, check_user_token

    check_accelerate_default_config()
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        check_user_token()

    load_datasets_fn: Any = globals().get("load_datasets")
    load_preference_datasets_fn: Any = globals().get("load_preference_datasets")
    train_fn: Any = globals().get("train")
    if load_datasets_fn is None or load_preference_datasets_fn is None:
        from axolotl.common import datasets as datasets_module

        load_datasets_fn = load_datasets_fn or datasets_module.load_datasets
        load_preference_datasets_fn = (
            load_preference_datasets_fn or datasets_module.load_preference_datasets
        )
    if train_fn is None:
        from axolotl.train import train as train_fn

    dataset_meta = None
    if cfg.get("plugins"):
        from axolotl.integrations.base import PluginManager

        plugin_manager = PluginManager.get_instance()
        dataset_meta = plugin_manager.load_datasets(cfg, preprocess=False)

    if dataset_meta is None:
        if cfg.rl:
            dataset_meta = load_preference_datasets_fn(cfg=cfg, cli_args=cli_args)
        else:
            dataset_meta = load_datasets_fn(cfg=cfg, cli_args=cli_args)

    model, tokenizer, trainer = train_fn(cfg=cfg, dataset_meta=dataset_meta)

    del model, tokenizer, trainer

    gc.collect()

    if cfg.get("plugins"):
        from axolotl.integrations.base import PluginManager

        plugin_manager = PluginManager.get_instance()
        plugin_manager.post_train_unload(cfg)


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    """
    Parses `axolotl` config, CLI args, and calls `do_train`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """
    parser_cls: Any = globals().get("HfArgumentParser")
    load_cfg_fn: Any = globals().get("load_cfg")
    if parser_cls is None:
        from transformers.hf_argparser import HfArgumentParser

        parser_cls = HfArgumentParser
    if load_cfg_fn is None:
        from axolotl.cli.config import load_cfg

        load_cfg_fn = load_cfg

    parsed_cfg = load_cfg_fn(config, **kwargs)
    parser = parser_cls(TrainerCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if parsed_cfg.use_ray:
        from ray.train import RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer

        train_loop_config = {"cfg": parsed_cfg.to_dict(), "cli_args": parsed_cli_args}
        trainer = TorchTrainer(
            ray_train_func,
            train_loop_config=train_loop_config,
            scaling_config=ScalingConfig(
                num_workers=parsed_cfg.ray_num_workers,
                resources_per_worker=parsed_cfg.resources_per_worker.to_dict(),
                use_gpu=True,
            ),
            run_config=RunConfig(
                name=parsed_cfg.ray_run_name,
                storage_path=Path(parsed_cfg.output_dir).absolute().as_posix(),
            ),
        )

        trainer.fit()
        return

    do_train(parsed_cfg, parsed_cli_args)


def ray_train_func(kwargs: dict):
    """Ray Train entrypoint executed on each GPU worker.

    Re-validates the config against worker-local GPU capabilities (deferred from
    the driver), then runs the standard training pipeline.
    """
    # cast `cfg` back to DictDefault (ray tune deepcopy has issues with DictDefault so needed it to be dict)
    # also renormalize the config now that TorchTrainer has spawned distributed workers
    cfg = DictDefault(kwargs["cfg"])

    # Plugins must be registered before `validate_config` so the plugin-extended
    # pydantic schema is in scope on this worker; otherwise plugin-specific cfg
    # fields are silently dropped by `model_dump(exclude_none=True)`.
    if cfg.get("plugins"):
        prepare_plugins_fn: Any = _lazy_attr("prepare_plugins")
        prepare_plugins_fn(cfg)

    # GPU capability detection was deferred from the driver; run the checks now
    # that we are on a worker that actually has the training device attached.
    accelerator_cls: Any = _lazy_attr("Accelerator")
    gpu_capabilities_fn: Any = _lazy_attr("gpu_capabilities")
    normalize_config_fn: Any = _lazy_attr("normalize_config")
    prepare_optim_env_fn: Any = _lazy_attr("prepare_optim_env")
    resolve_dtype_fn: Any = _lazy_attr("resolve_dtype")
    validate_config_fn: Any = _lazy_attr("validate_config")

    capabilities, env_capabilities = gpu_capabilities_fn()
    cfg = validate_config_fn(
        cfg,
        capabilities=capabilities,
        env_capabilities=env_capabilities,
    )

    # Derive here (not in controller normalize_config) so the worker's
    # validate_config above doesn't see both set and trip check_gas_bsz.
    cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
        cfg.batch_size // cfg.micro_batch_size
    )
    cfg.batch_size = (
        cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps
    )

    prepare_optim_env_fn(cfg)
    normalize_config_fn(cfg)
    resolve_dtype_fn(cfg)

    # ray serializing objects gets rid of frozen attribute - HF expects dict not DefaultDict
    if cfg.deepspeed and hasattr(cfg.deepspeed, "to_dict"):
        cfg.deepspeed = cfg.deepspeed.to_dict()

    # initialize accelerator before model instantiation
    accelerator_cls(gradient_accumulation_steps=cfg.gradient_accumulation_steps)

    # Bind the post-validation cfg to the plugin manager.
    if cfg.get("plugins"):
        plugin_set_cfg_fn: Any = _lazy_attr("plugin_set_cfg")
        plugin_set_cfg_fn(cfg)

    kwargs["cfg"] = cfg

    do_train(**kwargs)


if __name__ == "__main__":
    fire.Fire(do_cli)
