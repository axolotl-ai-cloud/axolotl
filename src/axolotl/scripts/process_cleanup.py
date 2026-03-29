"""Reusable process lifecycle management for vLLM serve scripts.

Handles graceful shutdown, orphan cleanup, and health monitoring for
multiprocessing-based server architectures where a main process
dispatches work to worker subprocesses that spawn GPU-heavy children
(e.g., vLLM EngineCore).

Usage:

    from axolotl.scripts.process_cleanup import ProcessManager

    manager = ProcessManager(processes, connections)
    manager.register_signal_handlers()

    # In FastAPI lifespan:
    async with manager.lifespan_context():
        yield  # server runs here

    # In endpoints:
    manager.check_workers_alive()  # raises if dead

    # In worker command loop:
    if manager.is_fatal_error(exc):
        break  # exit worker
"""

import asyncio
import atexit
import logging
import os
from multiprocessing import Process
from multiprocessing.connection import Connection

logger = logging.getLogger(__name__)


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its descendants (depth-first)."""
    import subprocess  # nosec B404

    try:
        result = subprocess.run(  # nosec B603 B607
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for child_pid in result.stdout.strip().split("\n"):
                child_pid = child_pid.strip()
                if child_pid:
                    kill_process_tree(int(child_pid))
    except (FileNotFoundError, ValueError):
        pass

    try:
        os.kill(pid, 9)
    except (ProcessLookupError, PermissionError):
        pass


def cleanup_orphan_processes(*patterns: str) -> None:
    """Kill orphan processes matching any of the given patterns.

    Uses ``pgrep -f`` to find processes. Skips the current process.
    Intended for cleaning up GPU-holding subprocesses (EngineCore)
    that survive their parent's death.
    """
    import subprocess  # nosec B404

    my_pid = os.getpid()
    for pattern in patterns:
        try:
            result = subprocess.run(  # nosec B603 B607
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for pid in result.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid and int(pid) != my_pid:
                        try:
                            os.kill(int(pid), 9)
                            logger.info("Killed orphan process %s (%s)", pid, pattern)
                        except (ProcessLookupError, ValueError):
                            pass
        except FileNotFoundError:
            pass


def is_fatal_worker_error(exc: Exception) -> bool:
    """Check if an exception indicates the worker should exit.

    Returns True for errors from which the worker cannot recover,
    such as the vLLM EngineCore dying.
    """
    exc_str = str(exc)
    exc_type = type(exc).__name__
    return (
        "EngineCore" in exc_str
        or "EngineDeadError" in exc_type
        or "engine" in exc_str.lower()
        and "died" in exc_str.lower()
    )


def safe_recv(conn: Connection):
    """Receive from a pipe, returning an error dict if the pipe is broken."""
    try:
        return conn.recv()
    except EOFError:
        return {"error": "Worker process died (pipe closed)", "kind": "worker_dead"}


class ProcessManager:
    """Manages worker process lifecycle for a FastAPI-based serve script.

    Handles:
    - Signal-based shutdown (SIGTERM)
    - Background health monitoring (detects dead workers)
    - Process tree cleanup on exit
    - Orphan EngineCore cleanup

    Args:
        processes: List of worker Process objects.
        connections: List of parent-side Pipe connections to workers.
        orphan_patterns: Process name patterns to search for orphans on cleanup.
            Defaults to ``["VLLM::EngineCore"]``.
        monitor_interval: Seconds between worker health checks.
        shutdown_timeout: Seconds to wait for graceful worker exit before SIGTERM.
        kill_timeout: Seconds to wait after SIGTERM before SIGKILL.
    """

    def __init__(
        self,
        processes: list[Process],
        connections: list[Connection],
        orphan_patterns: list[str] | None = None,
        monitor_interval: float = 5.0,
        shutdown_timeout: float = 30.0,
        kill_timeout: float = 15.0,
    ):
        self.processes = processes
        self.connections = connections
        self.orphan_patterns = orphan_patterns or ["VLLM::EngineCore"]
        self.monitor_interval = monitor_interval
        self.shutdown_timeout = shutdown_timeout
        self.kill_timeout = kill_timeout

    def register_cleanup(self) -> None:
        """Register atexit cleanup for orphan processes.

        Does NOT override SIGTERM — let uvicorn handle it naturally,
        which triggers the lifespan shutdown where ``_shutdown_workers``
        runs. The atexit handler is a safety net for abnormal exits.
        """
        atexit.register(self._cleanup_orphans)

    def check_workers_alive(self) -> None:
        """Raise RuntimeError if any worker process has died.

        Call this at the start of request handlers to fail fast
        instead of hanging on a broken pipe.
        """
        dead = [i for i, p in enumerate(self.processes) if not p.is_alive()]
        if dead:
            raise RuntimeError(
                f"vLLM worker(s) {dead} died. Restart the server to recover."
            )

    def get_health_status(self) -> dict:
        """Return health status dict. Use as the /health endpoint response."""
        dead = [i for i, p in enumerate(self.processes) if not p.is_alive()]
        if dead:
            return {
                "status": "unhealthy",
                "dead_workers": dead,
                "message": "Worker(s) died. Restart the server.",
            }
        return {"status": "ok"}

    async def monitor_workers(self) -> None:
        """Background coroutine that detects dead workers and exits.

        When all workers are dead, cleans up their process trees and
        orphan subprocesses, then force-exits the server.
        """
        while True:
            await asyncio.sleep(self.monitor_interval)
            alive = [p.is_alive() for p in self.processes]
            if not any(alive):
                logger.error(
                    "All vLLM workers died. Shutting down server. "
                    "Check logs for EngineCore errors and restart."
                )
                # Kill process trees for any workers that left orphans
                for p in self.processes:
                    if p.pid is not None:
                        kill_process_tree(p.pid)
                self._cleanup_orphans()
                os._exit(1)

    def _shutdown_workers(self) -> None:
        """Send shutdown commands and escalate to kill if needed."""
        for conn in self.connections:
            try:
                conn.send({"type": "shutdown"})
            except Exception:
                pass
        for i, p in enumerate(self.processes):
            if not p.is_alive():
                continue
            p.join(timeout=self.shutdown_timeout)
            if p.is_alive():
                logger.warning(
                    "Worker %d didn't exit in %.0fs, sending SIGTERM",
                    i,
                    self.shutdown_timeout,
                )
                p.terminate()
                p.join(timeout=self.kill_timeout)
            if p.is_alive():
                logger.warning("Worker %d didn't respond to SIGTERM, force killing", i)
                p.kill()
                p.join(timeout=5)
        self._cleanup_orphans()
        logger.info("Worker shutdown complete")

    def _cleanup_orphans(self) -> None:
        cleanup_orphan_processes(*self.orphan_patterns)
