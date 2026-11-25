"""An explicit tf32: true is dropped, with a warning, when there is no CUDA GPU."""

from unittest.mock import patch

from axolotl.utils.config import resolve_dtype
from axolotl.utils.dict import DictDefault


def test_tf32_is_disabled_without_cuda():
    cfg = DictDefault(tf32=True, bf16=False, fp16=False)
    with patch("axolotl.utils.config.torch.cuda.is_available", return_value=False):
        resolve_dtype(cfg)
    assert cfg.tf32 is False
