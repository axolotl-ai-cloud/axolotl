"""Configuration routing for native full-sequence varlen diffusion."""

from types import SimpleNamespace

from axolotl.integrations.diffusion.lm.trainer import AxolotlDiffusionTrainer
from axolotl.utils.dict import DictDefault


def test_native_nemotron_varlen_reaches_full_sequence_backend():
    trainer = AxolotlDiffusionTrainer.__new__(AxolotlDiffusionTrainer)
    trainer.axolotl_cfg = DictDefault(
        {
            "attn_implementation": "varlen",
            "sample_packing": True,
            "diffusion": {"from_causal_lm": False},
        }
    )
    trainer.model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="nemotron_labs_diffusion",
            mask_token_id=100,
        )
    )
    trainer._special_token_ids = set()

    backend = trainer._full_sequence_backend()

    assert backend.attention_backend == "varlen"
    assert backend.sample_packing is True
