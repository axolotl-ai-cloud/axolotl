"""Entry point of the reward processes async GRPO spawns.

Kept outside ``axolotl.core`` so a spawned worker imports only this module and the
reward functions it runs.
"""

import traceback


def reward_worker_main(conn, reward_funcs: list) -> None:
    """Run ``reward_funcs[i](**kwargs)`` for each ``(i, kwargs)`` received on ``conn``.

    Replies ``(True, rewards)`` or ``(False, traceback)``, and exits once the other end
    of the pipe closes.
    """
    while True:
        try:
            func_index, kwargs = conn.recv()
        except EOFError:
            return
        try:
            conn.send((True, list(reward_funcs[func_index](**kwargs))))
        except Exception:  # noqa: BLE001
            conn.send((False, traceback.format_exc()))
