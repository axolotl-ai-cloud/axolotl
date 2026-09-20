"""Model loading must not require torchao, which pyproject excludes on macOS and aarch64."""

import sys

from axolotl.loaders.model import ModelLoader


def test_quantized_init_guard_tolerates_missing_torchao(monkeypatch):
    for name in list(sys.modules):
        if name == "torchao" or name.startswith("torchao."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torchao", None)
    monkeypatch.delitem(sys.modules, "axolotl.utils.quantization", raising=False)

    ModelLoader._patch_quantized_init()
