"""test for train checkpoint utils"""

import os

from axolotl.utils.dict import DictDefault
from axolotl.utils.train import determine_last_checkpoint


def test_determine_last_checkpoint(temp_dir):
    cfg = DictDefault(
        output_dir=temp_dir,
    )
    for cpt_idx in [1, 9, 10, 20]:
        os.makedirs(
            os.path.join(cfg.output_dir, f"checkpoint-{cpt_idx}"), exist_ok=True
        )

    last_checkpoint = determine_last_checkpoint(cfg, update=False)
    assert last_checkpoint == os.path.join(cfg.output_dir, "checkpoint-20")

    cfg.resume_from_checkpoint = None
    cfg.auto_resume_from_checkpoints = True
    determine_last_checkpoint(cfg, update=True)
    assert cfg.resume_from_checkpoint == os.path.join(cfg.output_dir, "checkpoint-20")
