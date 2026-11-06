"""Skip conditions that query the CUDA device when a test is set up instead of at import.

Device queries such as ``torch.cuda.get_device_capability()`` initialize CUDA. In a module-level
``skipif`` they run during collection, so any session that merely collects the module starts with
CUDA initialized, which can break fork-based tests later in the run.
"""

from collections.abc import Callable

import torch


class AtSetup:
    """A ``skipif`` condition that pytest resolves, via ``bool()``, when a test is set up.

    The result is cached after the first evaluation. Use it only as the last operand of ``or`` /
    ``and`` (``not torch.cuda.is_available() or AtSetup(...)``); ``not`` evaluates it immediately.
    """

    def __init__(self, condition: Callable[[], bool]):
        self._condition = condition
        self._result: bool | None = None

    def __bool__(self) -> bool:
        if self._result is None:
            self._result = bool(self._condition())
        return self._result


def cuda_capability_below(major: int, minor: int = 0) -> AtSetup:
    """True without CUDA or on a device older than sm<major><minor>."""
    return AtSetup(
        lambda: (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability() < (major, minor)
        )
    )


def cuda_major_not_in(*majors: int) -> AtSetup:
    """True without CUDA or when the device's compute-capability major is not one of ``majors``."""
    return AtSetup(
        lambda: (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] not in majors
        )
    )
