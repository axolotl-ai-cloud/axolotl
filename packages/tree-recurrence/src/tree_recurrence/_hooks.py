# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Reversible replacement of module functions and class methods.

A hook calls the attribute it replaced through ``Hook.original`` at call time, so
whatever the original dispatches to when it runs (a hub kernel ``kernelize`` bound
later, say) is what the hook runs too, and the original can be swapped underneath it.
"""

from __future__ import annotations

import dis
import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

_HOOK_ATTR = "__tree_recurrence_hook__"


@dataclass(eq=False)
class Hook:
    owner: Any  # the module or class whose attribute is replaced
    name: str
    original: Any
    wrapper: Any = None
    enabled: bool = True
    # whether a class owner defined ``name`` itself rather than inheriting it
    owned: bool = True
    # (original, its instance forward, whether it runs a kernel), as last checked
    _kernel_check: tuple[Any, Any, bool] | None = None
    announced: bool = False

    def kernel_backed(self) -> bool:
        """Whether ``original`` runs a kernel rather than its torch reference (see
        ``dispatches_to_kernel``), checked again when it is rebound."""
        original = self.original
        bound = getattr(original, "__dict__", {}).get("forward")
        check = self._kernel_check
        if check is None or check[0] is not original or check[1] is not bound:
            check = (original, bound, dispatches_to_kernel(original))
            self._kernel_check = check
        return check[2]


_HOOKS: dict[tuple[Any, str], Hook] = {}


def hook_of(owner: Any, name: str) -> Hook | None:
    """Our live hook on ``owner.name``, when it is the attribute there now."""
    hook = _HOOKS.get((owner, name))
    if hook is not None and _current(owner, name) is hook.wrapper:
        return hook
    return None


def _current(owner: Any, name: str) -> Any:
    if isinstance(owner, type):
        return owner.__dict__.get(name)
    return getattr(owner, name, None)


def _ours(function: Any) -> Hook | None:
    hook = getattr(function, _HOOK_ATTR, None)
    return hook if hook is not None and hook.wrapper is function else None


def install_hook(owner: Any, name: str, build: Callable[[Hook], Callable]) -> Hook:
    """Replace ``owner.name`` with ``build(hook)``; installed once."""
    live = hook_of(owner, name)
    if live is not None:
        live.enabled = True
        return live
    stale = _HOOKS.pop((owner, name), None)
    if stale is not None:  # wrapped by someone else since: let it pass through
        stale.enabled = False
    original = getattr(owner, name)
    owned = not isinstance(owner, type) or name in owner.__dict__
    previous = _ours(original)
    if previous is not None:  # a hook of ours left on the attribute: replace it
        previous.enabled = False
        original = previous.original
        owned = previous.owned
    hook = Hook(owner, name, original, owned=owned)
    wrapper = build(hook)
    functools.update_wrapper(
        wrapper,
        original,
        assigned=("__module__", "__name__", "__qualname__", "__doc__"),
        updated=(),
    )
    setattr(wrapper, _HOOK_ATTR, hook)
    hook.wrapper = wrapper
    setattr(owner, name, wrapper)
    _HOOKS[(owner, name)] = hook
    return hook


def remove_hook(hook: Hook) -> None:
    """Restore the original, or, if something wrapped the hook since, make the hook
    pass everything through."""
    if _current(hook.owner, hook.name) is hook.wrapper:
        if hook.owned:
            setattr(hook.owner, hook.name, hook.original)
        else:
            delattr(hook.owner, hook.name)
    else:
        hook.enabled = False
    if _HOOKS.get((hook.owner, hook.name)) is hook:
        del _HOOKS[(hook.owner, hook.name)]


def hooks() -> list[Hook]:
    return list(_HOOKS.values())


def _nonlocals(fn: Any) -> dict[str, Any]:
    try:
        return dict(inspect.getclosurevars(fn).nonlocals)
    except (TypeError, ValueError):
        return {}


def dispatches_to_kernel(fn: Any) -> bool:
    """Whether transformers' kernel dispatcher ``fn`` runs a kernel.

    transformers decorates its kernel functions with
    ``use_kernel_func_from_hub_with_fallback``, which binds the package kernel (FLA,
    mamba-ssm, causal-conv1d) when it imports, or else the function's own PyTorch
    reference; with ``kernels`` installed the function becomes a module whose
    ``forward`` ``kernelize`` may later swap for a hub kernel. The references take
    no ``cu_seqlens``/``seq_idx``. Anything else counts as a kernel: if it then
    rejects the boundaries, it raises rather than mixing samples.
    """
    hook = _ours(fn)
    if hook is not None:
        return dispatches_to_kernel(hook.original)
    if isinstance(fn, torch.nn.Module):
        bound = vars(fn).get("forward")
        if (
            bound is not None
            and getattr(bound, "__func__", None) is not type(fn).forward
        ):
            return True
        inner = _nonlocals(type(fn).forward).get("func")
        return True if inner is None else dispatches_to_kernel(inner)
    resolved = _nonlocals(fn).get("is_new_implementation")
    return True if resolved is None else bool(resolved)


def loads_global(function: Any, name: str) -> bool:
    """Whether ``function``'s own code looks ``name`` up as a module global."""
    try:
        code = inspect.unwrap(function).__code__
    except AttributeError:
        return False
    return any(
        instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME")
        and instruction.argval == name
        for instruction in dis.get_instructions(code)
    )
