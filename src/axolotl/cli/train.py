"""CLI to run training on a model."""

import gc
import os
import queue
from pathlib import Path
from typing import Union

import fire
from accelerate import Accelerator
from transformers.hf_argparser import HfArgumentParser

from axolotl.cli.args import TrainerCliArgs
from axolotl.cli.checks import check_accelerate_default_config, check_user_token
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets, load_preference_datasets
from axolotl.integrations.base import PluginManager
from axolotl.train import train
from axolotl.utils.config import normalize_config, resolve_dtype
from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import prepare_optim_env


def do_train(cfg: DictDefault, cli_args: TrainerCliArgs):
    """
    Trains a `transformers` model by first loading the dataset(s) specified in the
    `axolotl` config, and then calling `axolotl.train.train`. Also runs the plugin
    manager's `post_train_unload` once training completes.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Training-specific CLI arguments.
    """
    check_accelerate_default_config()
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        check_user_token()

    # Start TUI early (before data loading) so it captures preprocessing events
    tui_renderer = None
    tui_queue: queue.Queue | None = None
    is_rank_0 = int(os.getenv("LOCAL_RANK", "0")) == 0
    if is_rank_0:
        from axolotl.train import _is_tui_enabled

        if _is_tui_enabled(cfg):
            import queue as _queue

            from axolotl.train import _get_tui_config
            from axolotl.tui.config import TUIConfig
            from axolotl.tui.renderer import TUIRenderer

            tui_config_dict = _get_tui_config(cfg)
            tui_config = (
                TUIConfig(**tui_config_dict)
                if isinstance(tui_config_dict, dict)
                else tui_config_dict
            )
            tui_queue = _queue.Queue(maxsize=4096)
            tui_renderer = TUIRenderer(config=tui_config, metric_queue=tui_queue)

            # Send initial run info
            model_name = cfg.base_model or ""
            training_mode = str(cfg.rl) if cfg.rl else "sft"
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            try:
                tui_queue.put_nowait(
                    {
                        "type": "run_info",
                        "model_name": model_name,
                        "training_mode": training_mode,
                        "world_size": world_size,
                    }
                )
            except _queue.Full:
                pass

            tui_renderer.start()

            # Attach logging handler early
            import logging

            from axolotl.tui.callback import _TUILogHandler

            _early_log_handler = _TUILogHandler(
                tui_queue, min_level=tui_config.log_level
            )
            _early_log_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            # Attach to BOTH root and axolotl loggers because axolotl logger
            # has propagate=False so root handler never sees axolotl.* messages
            root_logger = logging.getLogger()
            root_logger.addHandler(_early_log_handler)
            axolotl_logger = logging.getLogger("axolotl")
            axolotl_logger.addHandler(_early_log_handler)

            # Stash refs on cfg so train() can reuse the renderer
            cfg._tui_renderer = tui_renderer
            cfg._tui_queue = tui_queue
            cfg._tui_early_log_handler = _early_log_handler

    try:
        plugin_manager = PluginManager.get_instance()
        dataset_meta = plugin_manager.load_datasets(cfg, preprocess=False)
        if not dataset_meta:
            if cfg.rl:
                dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)
            else:
                dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

        del model, tokenizer, trainer

        gc.collect()

        plugin_manager = PluginManager.get_instance()
        plugin_manager.post_train_unload(cfg)
    finally:
        # If the TUI renderer started early but train() didn't get to stop it
        # (e.g., error during data loading), clean up here
        if tui_renderer is not None and not tui_renderer._stop_event.is_set():
            try:
                if tui_queue is not None:
                    tui_queue.put_nowait({"type": "done"})
            except queue.Full:
                pass
            tui_renderer.stop()
        # Remove early log handler from both root and axolotl loggers
        if hasattr(cfg, "_tui_early_log_handler"):
            import logging

            logging.getLogger().removeHandler(cfg._tui_early_log_handler)
            logging.getLogger("axolotl").removeHandler(cfg._tui_early_log_handler)


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    """
    Parses `axolotl` config, CLI args, and calls `do_train`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser(TrainerCliArgs)
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
        return trainer.fit()
    return do_train(parsed_cfg, parsed_cli_args)


def ray_train_func(kwargs: dict):
    # cast `cfg` back to DictDefault (ray tune deepcopy has issues with DictDefault so needed it to be dict)
    # also renormalize the config now that TorchTrainer has spawned distributed workers
    cfg = DictDefault(kwargs["cfg"])
    prepare_optim_env(cfg)
    normalize_config(cfg)

    # now that we are on the worker node, we can check `is_torch_bf16_gpu_available` to resolve dtype
    resolve_dtype(cfg)

    # ray serializing objects gets rid of frozen attribute - HF expects dict not DefaultDict
    if cfg.deepspeed and hasattr(cfg.deepspeed, "to_dict"):
        cfg.deepspeed = cfg.deepspeed.to_dict()

    # initialize accelerator before model instantiation
    Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)

    # Register plugins in Ray workers
    if cfg.get("plugins"):
        from axolotl.cli.config import plugin_set_cfg, prepare_plugins

        prepare_plugins(cfg)
        plugin_set_cfg(cfg)

    kwargs["cfg"] = cfg

    do_train(**kwargs)


if __name__ == "__main__":
    fire.Fire(do_cli)
