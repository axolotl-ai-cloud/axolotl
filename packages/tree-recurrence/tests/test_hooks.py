# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Hooks on class methods come off without a trace."""

from tree_recurrence._hooks import install_hook, remove_hook


def passthrough(hook):
    def forward(self, *args, **kwargs):
        return hook.original(self, *args, **kwargs)

    return forward


def test_a_hook_on_an_inherited_method_leaves_the_subclass_inheriting():
    class Base:
        def forward(self):
            return "base"

    class Sub(Base):
        pass

    hook = install_hook(Sub, "forward", passthrough)
    assert "forward" in vars(Sub) and Sub().forward() == "base"
    remove_hook(hook)
    assert "forward" not in vars(Sub)
    Base.forward = lambda self: "patched"
    assert Sub().forward() == "patched"


def test_a_hook_on_an_own_method_restores_it():
    class Layer:
        def forward(self):
            return "own"

    original = vars(Layer)["forward"]
    remove_hook(install_hook(Layer, "forward", passthrough))
    assert vars(Layer)["forward"] is original
