"""Public user-facing wrappers for the ProTrain runtime (§1).

Three entry points compose the full M1-M4 pipeline:

* :func:`auto_wrap` — paper Figure 1 ergonomics. Constructs a
  :class:`HardwareProfile` from live ``torch.cuda`` queries and calls
  :func:`protrain_model_wrapper` with auto-mode enabled. Drop-in for
  direct API users who don't go through the Axolotl plugin path.
* :func:`protrain_model_wrapper` — lower-level wrapper called once
  after model construction. Runs profiler (cached), layout, searcher,
  and installs block hooks. Use directly when fine-grained control
  over the :class:`HardwareProfile` or override knobs is required.
* :func:`protrain_optimizer_wrapper` — replaces the user's
  ``torch.optim.AdamW`` with the GPU/CPU FusedAdam adapter pair that
  the scheduler drives under the hood.
"""

from __future__ import annotations

from axolotl.integrations.protrain.api.model_wrapper import (
    auto_wrap,
    protrain_model_wrapper,
)
from axolotl.integrations.protrain.api.optim_wrapper import protrain_optimizer_wrapper
from axolotl.integrations.protrain.types import WrappedModel

__all__ = [
    "WrappedModel",
    "auto_wrap",
    "protrain_model_wrapper",
    "protrain_optimizer_wrapper",
]
