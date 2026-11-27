"""pytest tests for the `asr` config schema."""

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def test_asr_block_defaults_with_untyped_dataset(min_base_cfg):
    assert validate_config(min_base_cfg).asr is None

    cfg = min_base_cfg | DictDefault(
        skip_prepare_dataset=True,
        datasets=[{"path": "hf-internal-testing/librispeech_asr_dummy"}],
        asr={"language": "en"},
    )
    asr = validate_config(cfg).asr
    assert (asr.audio_column, asr.text_column, asr.language, asr.punctuation) == (
        "audio",
        "text",
        "en",
        True,
    )
    assert asr.language_column is None
