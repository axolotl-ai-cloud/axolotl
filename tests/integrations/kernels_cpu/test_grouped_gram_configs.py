"""The grouped-Gram autotune config is sized to the device's shared memory on first use,
not at import (the query initializes CUDA)."""

import pytest

triton = pytest.importorskip("triton")

from axolotl.integrations.kernels.libs.scattermoe_lora.kernels import (  # noqa: E402
    grouped_gram,
)


@pytest.mark.parametrize(
    "smem_optin, block_wide, num_stages",
    [(228 * 1024, 128, 3), (99 * 1024, 64, 2)],
)
def test_config_sized_on_first_read(monkeypatch, smem_optin, block_wide, num_stages):
    queries = []

    def fake_optin():
        queries.append(smem_optin)
        return smem_optin

    monkeypatch.setattr(grouped_gram, "_device_shared_mem_optin", fake_optin)
    tuner = triton.autotune(
        configs=grouped_gram._DeviceSizedConfigs(),
        key=["M_BUCKET", "WIDE", "RANK_IS_I"],
    )(grouped_gram._grouped_gram_kernel.fn)
    assert queries == []

    assert len(tuner.configs) == 1
    (config,) = tuner.configs
    assert config.kwargs == {"BLOCK_WIDE": block_wide, "BLOCK_M": 128}
    assert config.num_stages == num_stages
    assert tuner.configs[0] is config
    assert queries == [smem_optin]
