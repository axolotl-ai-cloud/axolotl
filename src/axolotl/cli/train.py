"""
CLI to run training on a model
"""
import logging
from pathlib import Path
from typing import Union

import fire
from dotenv import load_dotenv
from transformers.hf_argparser import HfArgumentParser
from accelerate import Accelerator

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    load_rl_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.integrations.base import PluginManager
from axolotl.train import train
from axolotl.utils.dict import DictDefault
from axolotl.utils.config import normalize_config

LOG = logging.getLogger("axolotl.cli.train")


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    train_loop_config = {"cfg": parsed_cfg.to_dict(), "cli_args": parsed_cli_args}
    trainer = TorchTrainer(
        ray_train_func,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(
            num_workers=4,
            resources_per_worker={"GPU": 1},
            use_gpu=True,
        ),
    )
    trainer.fit()
    # return do_train(parsed_cfg, parsed_cli_args)

def initialize_accelerator(config: DictDefault) -> Accelerator:
    # Initialize accelerator (needed for logging)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    return accelerator

def ray_train_func(kwargs: dict):
    # cast `cfg` back to DictDefault (ray tune deepcopy has issues with DictDefault so needed it to be dict)
    # also renormalize the config now that TorchTrainer has spawned distributed workers
   
    kwargs["cfg"] = DictDefault(kwargs["cfg"])
    normalize_config(kwargs["cfg"])
    config = kwargs["cfg"]

    config["use_ray"] = True

    # ray serializing objects gets rid of frozen attribute?
    config.deepspeed = config.deepspeed.to_dict()
    # initialize accelerator before model instantiation
    accelerator = initialize_accelerator(config)
    config["accelerator"] = accelerator

    do_train(**kwargs)
    
def do_train(cfg, cli_args) -> None:
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()
    
    if cfg.rl:  # and cfg.rl != "orpo":
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)
    else:
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    model, tokenizer = train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
    plugin_manager = PluginManager.get_instance()

    del model
    del tokenizer

    plugin_manager.post_train_unload(cfg)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
