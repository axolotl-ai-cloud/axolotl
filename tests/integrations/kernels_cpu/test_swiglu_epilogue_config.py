"""``moe_swiglu_epilogue`` config wiring into the scattermoe runtime."""

import pytest

pytest.importorskip("triton")  # the scattermoe_lora package imports its Triton kernels


def test_config_wiring(monkeypatch):
    from axolotl.integrations.kernels.args import KernelsArgs
    from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import (
        RUNTIME,
        configure_scattermoe_runtime,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.swiglu_epilogue import (
        swiglu_epilogue_enabled,
    )

    assert KernelsArgs.model_validate({"moe_swiglu_epilogue": True}).moe_swiglu_epilogue
    assert KernelsArgs.model_validate({}).moe_swiglu_epilogue is None

    class _C(dict):
        get = dict.get

    monkeypatch.delenv("AXOLOTL_SWIGLU_EPILOGUE", raising=False)
    configure_scattermoe_runtime(_C({"moe_swiglu_epilogue": True}))
    assert RUNTIME.swiglu_epilogue is True
    configure_scattermoe_runtime(_C({}))
    assert RUNTIME.swiglu_epilogue is False
    monkeypatch.setenv("AXOLOTL_SWIGLU_EPILOGUE", "1")
    configure_scattermoe_runtime(_C({}))
    assert RUNTIME.swiglu_epilogue is True
    configure_scattermoe_runtime(_C({"moe_swiglu_epilogue": False}))
    assert RUNTIME.swiglu_epilogue is False
    assert not swiglu_epilogue_enabled()
    RUNTIME.reset()
