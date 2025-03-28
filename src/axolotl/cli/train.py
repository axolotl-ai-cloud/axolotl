"""CLI to run training on a model."""

import logging
import os
from pathlib import Path
from typing import Union

import fire
from accelerate import Accelerator
from dotenv import load_dotenv
from transformers.hf_argparser import HfArgumentParser

from axolotl.cli.args import TrainerCliArgs
from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.checks import check_accelerate_default_config, check_user_token
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets, load_preference_datasets
from axolotl.integrations.base import PluginManager
from axolotl.train import train
from axolotl.utils import set_pytorch_cuda_alloc_conf
from axolotl.utils.config import normalize_config, resolve_dtype
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


def do_train(cfg: DictDefault, cli_args: TrainerCliArgs):
    """
    Trains a `transformers` model by first loading the dataset(s) specified in the
    `axolotl` config, and then calling `axolotl.train.train`. Also runs the plugin
    manager's `post_train_unload` once training completes.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Training-specific CLI arguments.
    """
    # Enable expandable segments for cuda allocation to improve VRAM usage
    set_pytorch_cuda_alloc_conf()

    print_axolotl_text_art()
    check_accelerate_default_config()
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        check_user_token()

    if cfg.rl:
        dataset_meta = load_preference_datasets(cfg=cfg, cli_args=cli_args)
    else:
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
    del model, tokenizer, trainer

    plugin_manager = PluginManager.get_instance()
    plugin_manager.post_train_unload(cfg)


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    """
    Parses `axolotl` config, CLI args, and calls `do_train`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """
    # pylint: disable=duplicate-code
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
    normalize_config(cfg)

    # now that we are on the worker node, we can check `is_torch_bf16_gpu_available` to resolve dtype
    resolve_dtype(cfg)

    # ray serializing objects gets rid of frozen attribute - HF expects dict not DefaultDict
    if cfg.deepspeed:
        cfg.deepspeed = cfg.deepspeed.to_dict()

    # initialize accelerator before model instantiation
    Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)

    kwargs["cfg"] = cfg

    do_train(**kwargs)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
