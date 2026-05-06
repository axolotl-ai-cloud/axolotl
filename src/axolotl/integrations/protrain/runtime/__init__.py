"""ProTrain runtime subpackage.

Hosts the runtime-side machinery used by the dispatcher and scheduler:
``streams.py`` (single-default-stream allocator and manual dealloc sync),
``scheduler.py`` (param prefetch / grad reduce+offload / CPU optimizer step
/ activation swap orchestration), and ``hooks.py`` (forward/backward hook
contracts for the wrapped blocks). All three modules are present in the
runtime package.
"""

from __future__ import annotations

__all__: list[str] = []
