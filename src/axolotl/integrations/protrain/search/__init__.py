"""ProTrain 5-knob searcher (M4).

Public surface:

- ``derive_bounds`` — upper bounds on the five tunable knobs
  (including ``n_offload`` — the OFFLOAD axis).
- ``search`` — exhaustive enumeration with OOM pruning over all five
  knobs (``n_persist``, ``n_swap``, ``n_ckpt``, ``n_offload``,
  ``micro_bs``); returns the minimum-runtime ``SearchResult`` that fits
  under the given GPU capacity.
"""

from __future__ import annotations

from axolotl.integrations.protrain.search.exhaustive import search
from axolotl.integrations.protrain.search.knobs import derive_bounds

__all__ = ["derive_bounds", "search"]
