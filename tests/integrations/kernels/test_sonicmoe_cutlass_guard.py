"""The SonicMoE cutlass DSL guard follows quack's own declared requirement."""

import importlib.metadata

import pytest

from axolotl.integrations.kernels.plugin import _check_sonicmoe_cutlass_dsl


def _install(monkeypatch, versions: dict, quack_requires: list[str]):
    def version(name):
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    def requires(name):
        if name == "quack-kernels" and name in versions:
            return quack_requires
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib.metadata, "requires", requires)


def test_cutlass_within_quack_range_passes(monkeypatch):
    _install(
        monkeypatch,
        {
            "nvidia-cutlass-dsl": "4.8.0",
            "quack-kernels": "0.6.5",
            "apache-tvm-ffi": "0.1.12",
        },
        ["nvidia-cutlass-dsl>=4.7", 'nvidia-cutlass-dsl[cu13]>=4.7; extra == "cu13"'],
    )
    _check_sonicmoe_cutlass_dsl()


@pytest.mark.parametrize(
    "cutlass, quack, spec",
    [
        ("4.6.0", "0.6.5", "nvidia-cutlass-dsl>=4.7"),
        ("4.8.0", "0.6.1", "nvidia-cutlass-dsl==4.6.0"),
    ],
)
def test_cutlass_outside_quack_range_raises(monkeypatch, cutlass, quack, spec):
    _install(
        monkeypatch,
        {"nvidia-cutlass-dsl": cutlass, "quack-kernels": quack},
        [spec],
    )
    with pytest.raises(RuntimeError, match=f"quack-kernels {quack}"):
        _check_sonicmoe_cutlass_dsl()


def test_missing_quack_skips_cutlass_check(monkeypatch):
    _install(monkeypatch, {"nvidia-cutlass-dsl": "4.6.0"}, [])
    _check_sonicmoe_cutlass_dsl()
