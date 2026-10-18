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


def start_time_reward(completions, **kwargs):
    import time

    start = time.time()
    time.sleep(0.3)
    return [start] * len(completions)
