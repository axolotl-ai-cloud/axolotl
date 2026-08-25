"""Factory for the PoLoRA optimizer."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from axolotl.integrations.base import BaseOptimizerFactory
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch.optim import Optimizer

LOG = get_logger(__name__)

_INSTALL_HINT = (
    "optimizer: polora requires the `polora` package, which is not installed. "
    "Install via:\n"
    "    pip install git+https://github.com/nikhilgsh/polora.git"
)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


# optim_args may arrive as strings from the "key=value" form.
_POLORA_ARGS: dict[str, Any] = {
    "beta1": float,
    "epsilon": float,
    "delta": float,
    "curvature_beta": float,
    "ns_steps": int,
    "higham_iters": int,
    "compile": _as_bool,
}


class _PoloraCompat:
    """Adapts upstream polora to the Trainer loop.

    Upstream keeps momentum and preconditioners in a plain ``pair_state`` dict keyed
    by pair index rather than in ``Optimizer.state``, so the inherited ``state_dict()``
    saves nothing and a resumed run silently restarts from zero. It also raises a bare
    ``ValueError`` when a factor has no gradient, which the Trainer triggers whenever a
    LoRA layer sits out a step.

    Under FSDP2 the factors arrive as DTensors sharded on dim 0, which for ``A`` is the
    LoRA rank itself. The algorithm is built on ``r x r`` Gram matrices and global
    spectral norms, so ``_sharded_pairs`` holds the real params and the optimizer runs
    over gathered stand-ins instead; see :func:`_gathered_pairs`.
    """

    _sharded_pairs: list | None = None

    def step(self, closure=None):
        if self._sharded_pairs is not None:
            _gather_into(self._sharded_pairs, self.pairs)  # type: ignore[attr-defined]
        try:
            loss = super().step(closure)  # type: ignore[misc]
        except ValueError as exc:
            if "Gradients are required" not in str(exc):
                raise
            raise RuntimeError(
                "polora requires a gradient on every LoRA factor at every step, but "
                "some had none. A LoRA layer that does not run on every batch causes "
                "this: a vision tower under lora_target_linear on a multimodal model, "
                "or an unrouted MoE expert. Exclude those with lora_exclude_modules."
            ) from exc
        if self._sharded_pairs is not None:
            _scatter_from(self.pairs, self._sharded_pairs)  # type: ignore[attr-defined]
        return loss

    def state_dict(self) -> dict:
        state_dict = super().state_dict()  # type: ignore[misc]
        state_dict["pair_state"] = {
            idx: dict(state)
            for idx, state in self.pair_state.items()  # type: ignore[attr-defined]
        }
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        current = self.pair_state  # type: ignore[attr-defined]
        saved_pairs = state_dict.get("pair_state")
        # Checked before super(), which raises on a param-count change of its own.
        if not isinstance(saved_pairs, dict) or not _pair_state_matches(
            saved_pairs, current
        ):
            LOG.warning(
                "polora: checkpoint pair state does not match this model's LoRA layout; "
                "resuming with fresh momentum and preconditioners."
            )
            return

        super().load_state_dict(state_dict)  # type: ignore[misc]
        for idx, saved in saved_pairs.items():
            device = self.pairs[int(idx)][0].device  # type: ignore[attr-defined]
            current[int(idx)] = {key: val.to(device) for key, val in saved.items()}


def _pair_state_matches(saved_pairs: dict, current: dict) -> bool:
    """Whether a checkpoint's pair state lines up with the freshly built one.

    Pairs are matched by discovery order, so a changed LoRA config invalidates them.
    """
    if len(saved_pairs) != len(current):
        return False
    return all(
        int(idx) in current
        and isinstance(saved, dict)
        and all(
            key in saved and saved[key].shape == current[int(idx)][key].shape
            for key in ("M_A", "M_B")
        )
        for idx, saved in saved_pairs.items()
    )


def _is_sharded(tensor) -> bool:
    from torch.distributed.tensor import DTensor

    return isinstance(tensor, DTensor) and any(
        placement.is_shard() for placement in tensor.placements
    )


def _gathered_pairs(pairs: list) -> list:
    """Dense full-shape stand-ins for sharded factors, sharing storage when replicated."""
    import torch

    gathered = []
    for pair in pairs:
        stand_ins = []
        for param in pair:
            if _is_sharded(param):
                stand_ins.append(
                    torch.nn.Parameter(param.full_tensor().detach().clone())
                )
            else:
                stand_ins.append(param)
        gathered.append(tuple(stand_ins))
    return gathered


def _gather_into(real_pairs: list, stand_ins: list) -> None:
    for real_pair, stand_in_pair in zip(real_pairs, stand_ins, strict=True):
        for real, stand_in in zip(real_pair, stand_in_pair, strict=True):
            if real is stand_in:
                continue
            stand_in.data.copy_(real.full_tensor().detach())
            grad = real.grad
            stand_in.grad = grad.full_tensor().detach() if _is_sharded(grad) else grad


def _scatter_from(stand_ins: list, real_pairs: list) -> None:
    from torch.distributed.tensor import distribute_tensor

    for stand_in_pair, real_pair in zip(stand_ins, real_pairs, strict=True):
        for stand_in, real in zip(stand_in_pair, real_pair, strict=True):
            if real is stand_in:
                continue
            # Every rank ran the same update on the same gathered inputs, so each can
            # slice its own shard out without a further collective.
            sharded = distribute_tensor(
                stand_in.data, real.device_mesh, real.placements
            )
            real.data.to_local().copy_(sharded.to_local())


@lru_cache(maxsize=1)
def _polora_cls() -> type:
    from polora import Polora

    return type("Polora", (_PoloraCompat, Polora), {})


class PoloraOptimizerFactory(BaseOptimizerFactory):
    """Builds a :class:`polora.Polora` over the model's trainable LoRA ``(A, B)`` pairs."""

    def __call__(
        self, opt_model, training_args=None, **optimizer_kwargs
    ) -> "Optimizer":
        if importlib.util.find_spec("polora") is None:
            raise ImportError(_INSTALL_HINT)
        from polora.optim import collect_lora_pairs

        lr = optimizer_kwargs.pop("lr")
        optimizer_kwargs.pop("weight_decay", None)  # polora has no decay term

        pairs = collect_lora_pairs(opt_model)
        if not pairs:
            raise ValueError(
                "optimizer: polora found no trainable LoRA (A, B) pairs on the model. "
                "It only optimizes LoRA factors, so it requires adapter: lora or qlora."
            )

        names = {
            id(param): name
            for name, param in opt_model.named_parameters()
            if param.requires_grad
        }
        covered = {id(param) for pair in pairs for param in pair}
        unhandled = [name for pid, name in names.items() if pid not in covered]
        if unhandled:
            raise ValueError(
                "optimizer: polora only updates LoRA (A, B) factors, but the model has "
                f"{len(unhandled)} other trainable parameter(s) that would never be "
                f"trained, e.g. {unhandled[:5]}. Freeze them or pick a different optimizer."
            )

        # polora unpacks every factor as (r, d_in) / (d_out, r); conv LoRA weights are 4D.
        non_2d = [
            names.get(id(param), "?")
            for pair in pairs
            for param in pair
            if param.ndim != 2
        ]
        if non_2d:
            raise ValueError(
                "optimizer: polora only supports 2D LoRA factors, but "
                f"{non_2d[:5]} are not. Drop convolutional targets from "
                "lora_target_modules or pick a different optimizer."
            )

        kwargs = {}
        for key, cast in _POLORA_ARGS.items():
            if key in optimizer_kwargs:
                kwargs[key] = cast(optimizer_kwargs.pop(key))
        if optimizer_kwargs:
            raise ValueError(
                f"Unsupported optim_args for polora: {sorted(optimizer_kwargs)}. "
                f"Supported: {sorted(_POLORA_ARGS)}."
            )

        sharded = any(_is_sharded(param) for pair in pairs for param in pair)
        step_pairs = _gathered_pairs(pairs) if sharded else pairs
        LOG.info(
            f"polora: optimizing {len(pairs)} LoRA (A, B) pairs"
            f"{' (FSDP2-sharded, gathered per step)' if sharded else ''}."
        )

        optimizer = _polora_cls()(pairs=step_pairs, lr=lr, **kwargs)
        if sharded:
            optimizer._sharded_pairs = pairs  # pylint: disable=protected-access
        return optimizer
