"""EOS sync between tokenizer, model config and generation config."""

from types import SimpleNamespace

from transformers import GenerationConfig

from axolotl.loaders.model import ModelLoader
from axolotl.utils.dict import DictDefault


class _Tokenizer:
    bos_token_id = None
    eos_token_id = 2
    _vocab = {"<|im_end|>": [7], "<|multi|>": [8, 9]}

    def encode(self, text, add_special_tokens=True):
        return self._vocab[text]


def _loader(model_eos, generation_eos, **cfg):
    loader = ModelLoader.__new__(ModelLoader)
    loader.cfg = DictDefault(cfg)
    loader.tokenizer = _Tokenizer()
    loader.model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=model_eos),
        generation_config=GenerationConfig(eos_token_id=generation_eos),
    )
    return loader


def test_generation_config_gains_eos_and_single_token_eot():
    loader = _loader(2, [5, 6], eot_tokens=["<|im_end|>", "<|multi|>", "<|im_end|>"])

    loader._adjust_model_config()

    assert loader.model.generation_config.eos_token_id == [5, 6, 2, 7]


def test_list_model_config_eos_keeps_ids():
    loader = _loader([5, 6], None)

    loader._adjust_model_config()

    assert loader.model.config.eos_token_id == [5, 6, 2]
    assert loader.model.generation_config.eos_token_id == 2
