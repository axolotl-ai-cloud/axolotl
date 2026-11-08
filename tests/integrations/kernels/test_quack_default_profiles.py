# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for the built-in SonicMoE B200 and H100 quack autotune profiles."""

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import axolotl.integrations.kernels.autotune_profiles as autotune_profiles
from axolotl.integrations.kernels.autotune_profiles import (
    _install_quack,
    _validate_quack_match,
)
from axolotl.integrations.kernels.plugin import _autotune_profiles
from axolotl.integrations.kernels.quack_autotune_policy import heuristic_fields
from axolotl.integrations.kernels.quack_default_profiles import (
    B200_SONICMOE_PROFILES,
    H100_SONICMOE_PROFILES,
    H200_SONICMOE_PROFILES,
    SONICMOE_DEFAULT_PROFILES,
    default_quack_profiles,
)
from axolotl.utils.dict import DictDefault

README = (
    Path(__file__).resolve().parents[3] / "src/axolotl/integrations/kernels/README.md"
)
KERNELS = ("gemm_gated_tuned", "gemm_dgated_tuned", "gemm_tuned")


@pytest.mark.parametrize(
    "gpu, shipped", [("B200", B200_SONICMOE_PROFILES), ("H100", H100_SONICMOE_PROFILES)]
)
def test_readme_shows_the_shipped_profiles(gpu, shipped):
    text = README.read_text()
    section = text[text.index(f"### Built-in profiles for SonicMoE on {gpu}") :]
    block = re.search(r"```yaml\n(.*?)```", section, re.S).group(1)
    assert yaml.safe_load(block)["autotune_profiles"] == shipped


@pytest.mark.parametrize("profile", SONICMOE_DEFAULT_PROFILES)
def test_profiles_are_valid_and_grouped_only(profile):
    _validate_quack_match(profile)
    assert profile["backend"] == "quack" and profile["gpu"] in ("B200", "H100", "H200")
    assert {"cu_seqlens_m.dim0", "cu_seqlens_k.dim0"} & set(profile["match"])
    assert 1 <= len(profile["configs"]) <= 5


@pytest.mark.parametrize(
    "user, dropped",
    [
        ([], set()),
        ([{"kernel": "gemm_tuned", "backend": "quack", "allow": {}}], {"gemm_tuned"}),
        ([{"kernel": "gemm_*", "backend": "quack", "allow": {}}], set(KERNELS)),
        (
            [
                {
                    "kernel": "*.quack.gemm_interface.gemm_dgated_tuned",
                    "backend": "quack",
                }
            ],
            {"gemm_dgated_tuned"},
        ),
        # a Triton entry with the same name is a different autotuner
        ([{"kernel": "gemm_tuned", "configs": [{"BLOCK_M": 64}]}], set()),
    ],
)
def test_a_profiled_kernel_keeps_only_the_configs_entries(user, dropped):
    kept = {p["kernel"] for p in default_quack_profiles(user)}
    assert kept == set(KERNELS) - dropped


def test_h200_reuses_the_h100_entries():
    assert [
        {**p, "gpu": "H100"} for p in H200_SONICMOE_PROFILES
    ] == H100_SONICMOE_PROFILES
    for h200, h100 in zip(H200_SONICMOE_PROFILES, H100_SONICMOE_PROFILES, strict=True):
        assert h200["match"] is not h100["match"]
        assert h200["configs"] is not h100["configs"]


@pytest.mark.parametrize(
    "device, gpu",
    [
        ("NVIDIA B200", "B200"),
        ("NVIDIA GB200", "B200"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA H100 PCIe", "H100"),
        ("NVIDIA H100 NVL", "H100"),
        ("NVIDIA H200", "H200"),
        ("NVIDIA H200 NVL", "H200"),
        ("NVIDIA GH200 480GB", "H200"),
        ("NVIDIA H800", None),
        ("NVIDIA H20", None),
    ],
)
def test_each_gpu_gets_one_built_in_set(monkeypatch, device, gpu):
    tuner = Autotuner("gemm_gated_tuned", sm90=True)
    monkeypatch.setattr(autotune_profiles, "_gpu_name", lambda: device)
    monkeypatch.setattr(autotune_profiles, "_autotuners", lambda: [])
    monkeypatch.setattr(
        autotune_profiles,
        "sonicmoe_autotuners",
        lambda: [("fake_quack.gemm_interface", None, tuner, True)],
    )
    autotune_profiles.apply_autotune_profiles(default_quack_profiles([]), quiet=True)
    applied = getattr(tuner, "_axolotl_quack_profiles", None) or []
    assert [p["gpu"] for p, _ in applied] == ([gpu] if gpu else [])


def test_defaults_are_copies():
    default_quack_profiles([])[0]["configs"].clear()
    assert B200_SONICMOE_PROFILES[0]["configs"]


@pytest.mark.parametrize(
    "cfg, with_defaults",
    [
        ({"use_sonicmoe": True}, True),
        ({"use_sonicmoe": True, "quack_autotune": True}, True),
        ({"use_sonicmoe": True, "quack_autotune": False}, False),
        ({"use_sonicmoe": True, "quack_autotune_default_profiles": False}, False),
        ({"use_scattermoe": True}, False),
    ],
)
def test_plugin_appends_defaults_after_the_configs_profiles(cfg, with_defaults):
    user = {"kernel": "gemm_tuned", "backend": "quack", "allow": {"tile_m": [256]}}
    profiles = _autotune_profiles(DictDefault({**cfg, "autotune_profiles": [user]}))
    assert profiles[0] == user
    defaults = profiles[1:]
    assert bool(defaults) == with_defaults
    assert all(p["kernel"] != "gemm_tuned" for p in defaults)


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int
    tile_n: int
    cluster_m: int
    cluster_n: int
    is_dynamic_persistent: bool
    use_tma_gather: bool
    swap_ab: bool
    pingpong: bool = False
    device_capacity: int = 10


class AutotuneConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __hash__(self):
        return hash(tuple(self.kwargs.items()))

    def __eq__(self, other):
        return self.kwargs == other.kwargs


def _sm100_grid():
    """quack's SM100 candidates (``_get_sm100_configs``)."""
    tile_n = (64, 128, 160, 192, 224, 256)
    tiles = [(128, n, c) for c in ((1, 1), (1, 2), (2, 1), (2, 2)) for n in tile_n]
    tiles += [(256, n, c) for c in ((2, 1), (2, 2)) for n in tile_n]
    tiles += [(256, 512, (2, 1))]
    return [
        AutotuneConfig(config=GemmConfig(m, n, cm, cn, clc, tma, swap))
        for m, n, (cm, cn) in tiles
        for swap in (False, True)
        for clc in (True, False)
        for tma in (True, False)
    ]


def _prune(configs, named_args, **kwargs):
    """The SM100 parts of quack's ``prune_invalid_gemm_configs``."""
    args = named_args | kwargs
    gather = args.get("A_idx") is not None
    if gather or args.get("cu_seqlens_m") is not None:
        configs = [c for c in configs if not c.kwargs["config"].swap_ab]
    if gather:
        return [c for c in configs if c.kwargs["config"].cluster_n == 1]
    return [c for c in configs if not c.kwargs["config"].use_tma_gather]


def _sm90_grid():
    """quack's SM90 candidates as the hub sonic-moe kernel listed them on H100."""
    pingpong = [(128, n) for n in (128, 160, 192, 208)] + [(192, 128)]
    tiles = (
        pingpong + [(128, 224), (128, 256)] + [(256, n) for n in (128, 160, 192, 208)]
    )
    return [
        AutotuneConfig(config=GemmConfig(m, n, cm, cn, False, False, swap, (m, n) in pingpong, 9))
        for m, n in tiles
        for cm, cn in ((1, 2), (2, 1))
        for swap in (False, True)
    ]  # fmt: skip


def _sm90_prune(configs, named_args, **kwargs):
    """What quack's ``prune_invalid_gemm_configs`` left of ``_sm90_grid`` on H100."""
    args = named_args | kwargs
    if args.get("A_idx") is not None:
        return [
            c
            for c in configs
            if not c.kwargs["config"].swap_ab
            and c.kwargs["config"].cluster_n == 1
            and c.kwargs["config"].tile_n != 208
        ]
    if args.get("cu_seqlens_m") is not None:
        return [c for c in configs if not c.kwargs["config"].swap_ab]
    return configs


class Autotuner:
    __module__ = "fake_quack.autotuner"

    def __init__(self, name, sm90=False):
        self.fn = SimpleNamespace(__name__=name)
        self.arg_names = ["A", "B", "out", "C", "postact_out", "PreAct"]
        self.arg_names += ["cu_seqlens_m", "cu_seqlens_k", "A_idx"]
        self.keys = []
        self.configs = _sm90_grid() if sm90 else _sm100_grid()
        self.early_config_prune = _sm90_prune if sm90 else _prune
        self.nargs = None

    def prune_configs(self, kwargs):
        return self.early_config_prune(self.configs, self.nargs, **kwargs)

    def _bench(self, *args, config, **meta):
        return [1.0, 1.0, 1.0]


# F5 at EP=8: H=4096, I=2048, 32 local experts, 16384 tokens x top-8, LoRA rank 32
T, TK, H, TWO_I, I, R = 16384, 131072, 4096, 4096, 2048, 32
CU = (33,)
CASES = {
    "gate/up forward": ("gemm_gated_tuned", 0, dict(A=(T, H), B=(32, H, TWO_I), postact_out=(TK, I), cu_seqlens_m=CU, A_idx=(TK,))),
    "gate/up forward + LoRA addend": ("gemm_gated_tuned", 0, dict(A=(T, H), B=(32, H, TWO_I), C=(TK, TWO_I), postact_out=(TK, I), cu_seqlens_m=CU, A_idx=(TK,))),
    "down backward": ("gemm_dgated_tuned", 1, dict(A=(T, H), B=(32, H, I), PreAct=(TK, TWO_I), cu_seqlens_m=CU, A_idx=(TK,))),
    "dW1": ("gemm_tuned", 2, dict(A=(H, T), B=(TK, TWO_I), out=(32, H, TWO_I), cu_seqlens_k=CU, A_idx=(TK,))),
    "LoRA grad, r rows": ("gemm_tuned", 3, dict(A=(R, TK), B=(TK, TWO_I), out=(32, R, TWO_I), cu_seqlens_k=CU)),
    "LoRA grad, r columns": ("gemm_tuned", 4, dict(A=(H, T), B=(TK, R), out=(32, H, R), cu_seqlens_k=CU, A_idx=(TK,))),
    "x P, gathered": ("gemm_tuned", 5, dict(A=(T, H), B=(32, H, R), out=(TK, R), cu_seqlens_m=CU, A_idx=(TK,))),
    "a P": ("gemm_tuned", 5, dict(A=(TK, I), B=(32, I, R), out=(TK, R), cu_seqlens_m=CU)),
    "u Q": ("gemm_tuned", 6, dict(A=(TK, R), B=(32, R, TWO_I), out=(TK, TWO_I), cu_seqlens_m=CU)),
    "down forward + LoRA addend": ("gemm_tuned", 7, dict(A=(TK, I), B=(32, I, H), C=(TK, H), out=(TK, H), cu_seqlens_m=CU)),
    "dX": ("gemm_tuned", 7, dict(A=(TK, TWO_I), B=(32, TWO_I, H), out=(TK, H), cu_seqlens_m=CU)),
}  # fmt: skip


@pytest.mark.parametrize("call", sorted(CASES))
def test_each_sonicmoe_call_sweeps_its_profile(call):
    kernel, index, shapes = CASES[call]
    tuner = Autotuner(kernel)
    _install_quack(tuner, [p for p in B200_SONICMOE_PROFILES if p["kernel"] == kernel])
    tuner.nargs = {k: torch.empty(v, device="meta") for k, v in shapes.items()}
    picked = [c.kwargs["config"] for c in tuner.prune_configs({})]
    valid = _prune(
        [AutotuneConfig(config=GemmConfig(**c)) for c in B200_SONICMOE_PROFILES[index]["configs"]],
        tuner.nargs,
    )  # fmt: skip
    assert picked == [c.kwargs["config"] for c in valid] and picked
    assert tuner._axolotl_skip_bench == (len(picked) == 1)


@pytest.mark.parametrize("call", sorted(CASES))
def test_each_sonicmoe_call_sweeps_its_h100_profile(call):
    kernel, index, shapes = CASES[call]
    tuner = Autotuner(kernel, sm90=True)
    _install_quack(tuner, [p for p in H100_SONICMOE_PROFILES if p["kernel"] == kernel])
    tuner.nargs = {k: torch.empty(v, device="meta") for k, v in shapes.items()}
    picked = [c.kwargs["config"] for c in tuner.prune_configs({})]
    listed = [
        c
        for w in H100_SONICMOE_PROFILES[index]["configs"]
        for c in tuner.configs
        if all(getattr(c.kwargs["config"], f) == v for f, v in w.items())
    ]
    valid = [c.kwargs["config"] for c in _sm90_prune(listed, tuner.nargs)]
    assert picked == valid and picked
    assert tuner._axolotl_skip_bench == (len(picked) == 1)


@pytest.mark.parametrize("call", sorted(CASES))
def test_the_sm90_rule_picks_a_valid_candidate(call):
    kernel, _, shapes = CASES[call]
    args = {k: torch.empty(v, device="meta") for k, v in shapes.items()}
    wanted = heuristic_fields(kernel, args, capacity=9)
    valid = [c.kwargs["config"] for c in _sm90_prune(_sm90_grid(), args)]
    assert any(all(getattr(c, f) == v for f, v in wanted.items()) for c in valid)


def test_dense_calls_keep_the_full_search():
    tuner = Autotuner("gemm_tuned")
    _install_quack(
        tuner, [p for p in B200_SONICMOE_PROFILES if p["kernel"] == "gemm_tuned"]
    )
    shapes = dict(A=(T, H), B=(H, H), out=(T, H))
    tuner.nargs = {k: torch.empty(v, device="meta") for k, v in shapes.items()}
    assert len(tuner.prune_configs({})) == len(_prune(tuner.configs, tuner.nargs))


def test_post_model_load_applies_the_defaults(monkeypatch):
    import axolotl.integrations.kernels.autotune_profiles as profiles_mod
    import axolotl.integrations.kernels.libs.sonicmoe.epilogue as epilogue_mod
    import axolotl.integrations.kernels.plugin as plugin_mod
    import axolotl.integrations.kernels.quack_autotune_policy as policy_mod

    applied = []
    monkeypatch.setattr(plugin_mod, "get_active_adapters", lambda cfg: [])
    monkeypatch.setattr(plugin_mod, "_preload_sonicmoe_kernel", lambda: True)
    monkeypatch.setattr(profiles_mod, "apply_autotune_profiles", applied.append)
    monkeypatch.setattr(policy_mod, "install_quack_key_policy", lambda *a: 0)
    monkeypatch.setattr(epilogue_mod, "check_model_epilogues", lambda model: None)
    user = {
        "kernel": "gemm_dgated_tuned",
        "backend": "quack",
        "configs": [{"tile_m": 256}],
    }
    cfg = DictDefault({"use_sonicmoe": True, "autotune_profiles": [user]})

    plugin_mod.KernelsPlugin().post_model_load(cfg, object())

    (profiles,) = applied
    assert profiles[0] == user
    assert [p["kernel"] for p in profiles[1:]] == [
        p["kernel"]
        for p in SONICMOE_DEFAULT_PROFILES
        if p["kernel"] != "gemm_dgated_tuned"
    ]
    assert cfg.autotune_profiles == [user]
