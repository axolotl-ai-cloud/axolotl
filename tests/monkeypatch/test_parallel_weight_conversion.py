# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Parallel preallocated stack/cat must match torch.stack/torch.cat exactly."""

import torch

from axolotl.monkeypatch import parallel_weight_conversion as pwc


def _tensors(n, shape, dtype=torch.bfloat16):
    return [torch.randn(*shape).to(dtype) for _ in range(n)]


def test_parallel_stack_matches_torch(monkeypatch):
    monkeypatch.setattr(pwc, "MIN_PARALLEL_BYTES", 0)
    tensors = _tensors(6, (33, 17))
    for dim in (0, 1, 2):
        assert torch.equal(pwc.parallel_stack(tensors, dim), torch.stack(tensors, dim))


def test_parallel_cat_matches_torch(monkeypatch):
    monkeypatch.setattr(pwc, "MIN_PARALLEL_BYTES", 0)
    tensors = [torch.randn(4, 5), torch.randn(4, 7), torch.randn(4, 1)]
    assert torch.equal(pwc.parallel_cat(tensors, 1), torch.cat(tensors, 1))
    assert torch.equal(pwc.parallel_cat(tensors, -1), torch.cat(tensors, -1))
    rows = [torch.randn(2, 3), torch.randn(5, 3)]
    assert torch.equal(pwc.parallel_cat(rows, 0), torch.cat(rows, 0))


def test_small_inputs_take_the_serial_path(monkeypatch):
    called = []
    monkeypatch.setattr(pwc, "_pool", lambda: called.append(1))
    tensors = _tensors(3, (2, 2))
    assert torch.equal(pwc.parallel_stack(tensors, 0), torch.stack(tensors, 0))
    assert not called


def test_patch_routes_transformers_converters(monkeypatch):
    cml = __import__("transformers.core_model_loading", fromlist=["x"])
    monkeypatch.setattr(cml, "_axolotl_parallel_convert", False, raising=False)
    orig_merge, orig_concat = cml.MergeModulelist.convert, cml.Concatenate.convert
    try:
        assert pwc.patch_parallel_weight_conversion()
        monkeypatch.setattr(pwc, "MIN_PARALLEL_BYTES", 0)
        experts = _tensors(4, (3, 2))
        merged = cml.MergeModulelist(dim=0).convert(
            {"experts.*.w": experts}, ["experts.*.w"], ["experts.w"]
        )
        assert torch.equal(merged["experts.w"], torch.stack(experts, 0))
        gate, up = _tensors(1, (4, 3)), _tensors(1, (4, 3))
        fused = cml.Concatenate(dim=0).convert(
            {"gate": gate, "up": up}, ["gate", "up"], ["gate_up"]
        )
        assert torch.equal(fused["gate_up"], torch.cat(gate + up, 0))
    finally:
        cml.MergeModulelist.convert = orig_merge
        cml.Concatenate.convert = orig_concat
        cml._axolotl_parallel_convert = False
