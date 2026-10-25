"""Picklable reward functions for the async GRPO reward-pool tests.

A separate module with no heavy imports, so the spawned reward workers start fast.
"""

import os
import signal


def length_reward(completions, **kwargs):
    return [float(len(c[0]["content"])) for c in completions]


def pid_reward(completions, **kwargs):
    return [float(os.getpid())] * len(completions)


def alarm_reward(completions, **kwargs):
    # signal.alarm raises ValueError outside the main thread, which is where TRL's
    # thread-based scoring would run it.
    signal.alarm(0)
    return [1.0] * len(completions)


def failing_reward(completions, **kwargs):
    raise ZeroDivisionError("bad completion")


def overlap_reward(completions, **kwargs):
    """1.0 once two calls have run at the same time, 0.0 if none did within 10 s."""
    import time
    import uuid
    from pathlib import Path

    rendezvous = Path(os.environ["TRL_ASYNC_TEST_REWARD_RENDEZVOUS"])
    overlapped = rendezvous / "overlapped"
    running = rendezvous / f"running-{uuid.uuid4().hex}"
    running.touch()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if len(list(rendezvous.glob("running-*"))) > 1:
                overlapped.touch()
            if overlapped.exists():
                return [1.0] * len(completions)
            time.sleep(0.01)
        return [0.0] * len(completions)
    finally:
        running.unlink()


def lookup_tool(query: str) -> str:
    """Look up a query.

    Args:
        query: What to look up.
    """
    return query
