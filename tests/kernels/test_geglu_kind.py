"""``_geglu_kind`` must pick the GEGLU kernel whose GELU matches the model's act_fn."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

pytest.importorskip("bitsandbytes", reason="axolotl.kernels.lora imports bitsandbytes")

from axolotl.kernels.lora import _geglu_kind  # noqa: E402

_ACT_FNS = {name: ACT2FN[name] for name in sorted(ACT2FN) if "gelu" in name}
_ACT_FNS["nn.GELU"] = nn.GELU()
_ACT_FNS["nn.GELU_tanh"] = nn.GELU(approximate="tanh")


@pytest.mark.parametrize("act_fn", _ACT_FNS.values(), ids=_ACT_FNS.keys())
def test_geglu_kind_matches_activation(act_fn):
    # past +-10 so ClippedGELU (gelu_10) differs from plain GELU
    x = torch.linspace(-12, 12, 2001, dtype=torch.float64)
    y = act_fn(x)
    matching = [
        kind
        for kind, approximate in (("geglu", "none"), ("geglu_tanh", "tanh"))
        if torch.allclose(y, F.gelu(x, approximate=approximate), rtol=0, atol=1e-9)
    ]
    assert _geglu_kind(act_fn) == (matching[0] if matching else None)


def test_geglu_kind_without_act_fn_is_tanh():
    # Liger's GEGLU MLPs have no act_fn and always compute tanh GELU
    assert _geglu_kind(None) == "geglu_tanh"
