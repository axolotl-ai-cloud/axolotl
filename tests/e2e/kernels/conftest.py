"""Shared fixtures for the GPU kernel tests."""

import pytest


@pytest.fixture(scope="session")
def sonicmoe_setup():
    """The kernels plugin's ``use_sonicmoe`` setup, with ``quack_autotune: false``.

    The plugin shims cutlass-dsl for the quack vendored in the hub kernel and makes quack compile
    in-process; without that, sweeps on B200 fault. No sweep keeps CI to one compile per GEMM key.
    """
    from transformers.integrations.sonicmoe import (
        is_sonicmoe_loadable,
        load_sonicmoe_kernel,
    )

    from axolotl.integrations.kernels.plugin import _check_sonicmoe_gpu_compat
    from axolotl.integrations.kernels.quack_autotune_policy import (
        install_quack_key_policy,
    )

    if not is_sonicmoe_loadable():
        pytest.skip("sonic-moe kernel not loadable")
    _check_sonicmoe_gpu_compat()
    load_sonicmoe_kernel()
    install_quack_key_policy("pow2", tune=False)
