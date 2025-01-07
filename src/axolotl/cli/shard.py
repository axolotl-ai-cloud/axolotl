"""CLI to shard a trained model into 10GiB chunks."""

import logging
from pathlib import Path
from typing import Union

import fire
from dotenv import load_dotenv

from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.config import load_cfg
from axolotl.common.cli import load_model_and_tokenizer
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


def shard(*, cfg: DictDefault):
    model, _ = load_model_and_tokenizer(cfg=cfg)
    safe_serialization = cfg.save_safetensors is True
    LOG.debug("Re-saving model w/ sharding")
    model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    shard(cfg=parsed_cfg)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
