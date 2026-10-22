"""Retry checkpoint writes that fail on a full disk, an exceeded quota, or an I/O error.

While a checkpoint is saved, the file writers used by transformers, PEFT, accelerate and
DeepSpeed are wrapped so that a write failing with a disk error waits and rewrites the
same file on the rank that failed. That rank never skips or reorders a collective, so
the other ranks only wait at their next collective (keep the wait below ``ddp_timeout``).
Distributed checkpoint (DCP) saves raise on every rank together, so every rank retries
those together.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import importlib
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_DISK_ERRNOS = {
    code
    for code in (
        getattr(errno, name, None)
        for name in ("ENOSPC", "EDQUOT", "EIO", "EFBIG", "ESTALE")
    )
    if code is not None
}
# Rust io::Error renders as "... (os error 122)" or "Os { code: 122, ... }"
_OS_ERROR_CODE = re.compile(r"os error (\d+)|\bcode: (\d+)|\[errno (\d+)\]")
_DISK_PHRASES = (
    "no space left on device",
    "not enough space on the disk",
    "quota exceeded",
    "quotaexceeded",
    "storagefull",
    "input/output error",
    "file too large",
    "stale file handle",
    # torch.save's C++ writer reports failed writes without an errno: a short write
    # ("unexpected pos 704 vs 598"), a failed stream ("basic_ios::clear: iostream
    # error"), or a failed fwrite
    "unexpected pos",
    "iostream error",
    "file write failed",
)


def _causes(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.extend((current.__cause__, current.__context__))
        stack.extend(getattr(current, "exceptions", ()) or ())
        # torch.distributed.checkpoint.CheckpointException: {rank: (exception, traceback)}
        for failure in (getattr(current, "failures", None) or {}).values():
            stack.append(failure[0] if isinstance(failure, tuple) else failure)


def is_disk_error(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything it wraps) is a full disk, exceeded quota or I/O error."""
    for current in _causes(exc):
        if isinstance(current, OSError) and current.errno in _DISK_ERRNOS:
            return True
        text = str(current).lower()
        if any(phrase in text for phrase in _DISK_PHRASES):
            return True
        for groups in _OS_ERROR_CODE.findall(text):
            if any(group and int(group) in _DISK_ERRNOS for group in groups):
                return True
    return False


@dataclass(frozen=True)
class RetryPolicy:
    wait_seconds: float
    retries: int


_local = threading.local()


def _active_policy() -> RetryPolicy | None:
    stack = getattr(_local, "policies", None)
    return stack[-1] if stack else None


def _free_space(target: Any) -> str:
    try:
        path = Path(target)
        while not path.exists() and path != path.parent:
            path = path.parent
        free = shutil.disk_usage(path).free
    except (OSError, TypeError, ValueError):
        return ""
    return f" ({free / 2**30:.1f} GiB free)"


def _retry(fn: Callable, target: Any, *, collective: bool = False):
    """Call ``fn`` and, under an active policy, retry it after disk errors.

    Only the outermost wrapped writer retries; nested ones (a processor saving its
    tokenizer) pass through, so waits do not multiply.
    """
    if getattr(_local, "retrying", 0):
        return fn()
    attempt = 0
    while True:
        _local.retrying = getattr(_local, "retrying", 0) + 1
        try:
            return fn()
        except BaseException as exc:
            policy = _active_policy()
            if (
                policy is None
                or attempt >= policy.retries
                or not is_disk_error(exc)
                or (collective and type(exc).__name__ != "CheckpointException")
            ):
                raise
            attempt += 1
            LOG.warning(
                f"Writing {target} failed with a disk error{_free_space(target)}: "
                f"{exc}. Retrying in {policy.wait_seconds:g}s "
                f"(retry {attempt} of {policy.retries})."
            )
            time.sleep(policy.wait_seconds)
        finally:
            _local.retrying -= 1


def retry_call(target: Any, fn: Callable, *args, **kwargs):
    """Run a write under the active retry policy, if any."""
    return _retry(functools.partial(fn, *args, **kwargs), target)


def _path_arg(index: int, name: str) -> Callable[[tuple, dict], Any]:
    def target(args: tuple, kwargs: dict) -> Any:
        value = kwargs.get(name, args[index] if len(args) > index else None)
        return value if isinstance(value, (str, os.PathLike)) else None

    return target


def _checkpoint_id(args: tuple, kwargs: dict) -> Any:
    writer = kwargs.get("storage_writer")
    return kwargs.get("checkpoint_id") or getattr(writer, "path", None) or "checkpoint"


def _wrap(fn: Callable, target_of: Callable, *, collective: bool) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        target = target_of(args, kwargs)
        if target is None:
            # file objects and streams cannot be rewritten from the start
            return fn(*args, **kwargs)
        return _retry(
            functools.partial(fn, *args, **kwargs), target, collective=collective
        )

    wrapper._axolotl_disk_retry_original = fn  # type: ignore[attr-defined]
    return wrapper


# (module, attribute or Class.method, target extractor, collective)
_WRITERS: tuple[tuple[str, str, Callable, bool], ...] = (
    ("torch", "save", _path_arg(1, "f"), False),
    ("safetensors.torch", "save_file", _path_arg(1, "filename"), False),
    ("transformers.modeling_utils", "safe_save_file", _path_arg(1, "filename"), False),
    ("peft.peft_model", "safe_save_file", _path_arg(1, "filename"), False),
    (
        "huggingface_hub.serialization._torch",
        "safe_save_file",
        _path_arg(1, "filename"),
        False,
    ),
    ("torch.distributed.checkpoint", "save", _checkpoint_id, True),
    ("torch.distributed.checkpoint", "save_state_dict", _checkpoint_id, True),
    # rank-local writers of the small files in a checkpoint (config, tokenizer, state)
    (
        "transformers.trainer_callback",
        "TrainerState.save_to_json",
        _path_arg(1, "json_path"),
        False,
    ),
    (
        "transformers.configuration_utils",
        "PretrainedConfig.save_pretrained",
        _path_arg(1, "save_directory"),
        False,
    ),
    (
        "transformers.generation.configuration_utils",
        "GenerationConfig.save_pretrained",
        _path_arg(1, "save_directory"),
        False,
    ),
    (
        "transformers.tokenization_utils_base",
        "PreTrainedTokenizerBase.save_pretrained",
        _path_arg(1, "save_directory"),
        False,
    ),
    (
        "transformers.processing_utils",
        "ProcessorMixin.save_pretrained",
        _path_arg(1, "save_directory"),
        False,
    ),
    (
        "peft.config",
        "PeftConfigMixin.save_pretrained",
        _path_arg(1, "save_directory"),
        False,
    ),
)

_patch_lock = threading.Lock()
_patch_depth = 0
_originals: list[tuple[Any, str, Any]] = []


def _install() -> None:
    for module_name, path, target_of, collective in _WRITERS:
        try:
            owner: Any = importlib.import_module(module_name)
        except ImportError:
            continue
        *parents, attr = path.split(".")
        for parent in parents:
            owner = getattr(owner, parent, None)
        original = owner.__dict__.get(attr) if isinstance(owner, type) else None
        if original is None:
            original = getattr(owner, attr, None)
        if original is None or hasattr(original, "_axolotl_disk_retry_original"):
            continue
        _originals.append((owner, attr, original))
        setattr(owner, attr, _wrap(original, target_of, collective=collective))


def _uninstall() -> None:
    while _originals:
        owner, attr, original = _originals.pop()
        setattr(owner, attr, original)


@contextlib.contextmanager
def retry_disk_writes(wait_seconds: float, retries: int = 1):
    """Retry checkpoint file writes that fail with disk errors while this context is open."""
    global _patch_depth
    policies = getattr(_local, "policies", None)
    if policies is None:
        policies = _local.policies = []
    policies.append(RetryPolicy(wait_seconds=wait_seconds, retries=retries))
    with _patch_lock:
        if _patch_depth == 0:
            _install()
        _patch_depth += 1
    try:
        yield
    finally:
        policies.pop()
        with _patch_lock:
            _patch_depth -= 1
            if _patch_depth == 0:
                _uninstall()
