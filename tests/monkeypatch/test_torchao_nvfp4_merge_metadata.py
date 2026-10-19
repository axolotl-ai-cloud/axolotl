"""Native adapter quantizer identity must be checked by the real merge consumer."""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torchao")

from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor  # noqa: E402
from torchao.prototype.safetensors.safetensors_support import (  # noqa: E402
    flatten_tensor_state_dict,
    unflatten_tensor_state_dict,
)

from axolotl.cli.utils.native_nvfp4_merge import merge_native_nvfp4_shard  # noqa: E402
from axolotl.monkeypatch.torchao_nvfp4_merge_metadata import (  # noqa: E402
    build_native_merge_aware_metadata,
    validate_native_merge_aware_header,
    validate_native_merge_aware_target,
)


def _weight():
    return NVFP4Tensor.to_nvfp4(
        torch.randn(32, 32, dtype=torch.bfloat16),
        per_tensor_scale=torch.tensor(1.25),
    )


def test_native_metadata_uses_native_merge_dispatch():
    from axolotl.cli.utils.lora_merge import _resolve_nvfp4_scale_mode

    weight = _weight()
    metadata = build_native_merge_aware_metadata(
        {"model.q.weight": weight}, start_step=4
    )
    assert validate_native_merge_aware_header(metadata)
    assert validate_native_merge_aware_target(metadata, "model.q.weight", weight)
    assert _resolve_nvfp4_scale_mode({"nvfp4_merge_aware": metadata}) == "reuse"
    assert metadata["start_step"] == 4


@pytest.mark.parametrize("mutation", ["encoder", "version", "targets"])
def test_native_metadata_mismatch_warns_instead_of_aborting(mutation):
    from axolotl.cli.utils.lora_merge import _resolve_nvfp4_scale_mode

    metadata = build_native_merge_aware_metadata({"model.q.weight": _weight()})
    metadata[mutation] = "mismatched"
    with patch(
        "axolotl.monkeypatch.torchao_nvfp4_merge_metadata.LOG.warning"
    ) as warning:
        assert _resolve_nvfp4_scale_mode({"nvfp4_merge_aware": metadata}) == "reuse"
    assert "NVFP4 MERGE WARNING" in warning.call_args.args[0]


@pytest.mark.parametrize("mismatch", [False, True])
def test_native_writer_checks_recipe_and_preserves_native_output(mismatch):
    weight = _weight()
    metadata = build_native_merge_aware_metadata({"model.q.weight": weight})
    if mismatch:
        weight.per_tensor_scale = weight.per_tensor_scale + 0.5
    tensors, header = flatten_tensor_state_dict({"model.q.weight": weight})
    with patch(
        "axolotl.monkeypatch.torchao_nvfp4_merge_metadata.LOG.warning"
    ) as warning:
        output, result_header, count = merge_native_nvfp4_shard(
            tensors,
            header,
            lambda dense, _: (dense, True),
            merge_aware_metadata=metadata,
        )
    assert count == 1
    result, _ = unflatten_tensor_state_dict(output, result_header)
    assert type(result["model.q.weight"]).__name__ == "NVFP4Tensor"
    assert torch.equal(
        result["model.q.weight"].per_tensor_scale, weight.per_tensor_scale
    )
    assert warning.call_count == int(mismatch)
