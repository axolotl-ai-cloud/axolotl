"""Ray cluster bring-up/teardown helpers backing `axolotl ray up|down|status`.

The model is deliberately simple: `axolotl ray up` runs on the head node, starts
a local Ray head, and fans `ray start --address=<head>` out to the hostfile's
worker entries over plain ssh. Cluster metadata is recorded so `down`/`status`
need no arguments. KubeRay/autoscaler setups should be managed with their own
tooling; `--launcher ray` attaches to any running cluster either way.
"""

import json
import shutil
import socket
import subprocess  # nosec
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import click

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

STATE_FILE = Path.home() / ".cache" / "axolotl" / "ray-cluster.json"
TEMP_DIR_ROOT = Path.home() / ".cache" / "axolotl" / "ray"
_SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]


@dataclass
class ClusterState:
    """Recorded metadata for a cluster started by `axolotl ray up`."""

    head_ip: str
    port: int
    dashboard_port: int
    temp_dir: str
    workers: list[str] = field(default_factory=list)
    ssh_user: str | None = None
    ssh_key: str | None = None

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "ClusterState | None":
        if not STATE_FILE.exists():
            return None
        return cls(**json.loads(STATE_FILE.read_text(encoding="utf-8")))

    @property
    def address(self) -> str:
        return f"{self.head_ip}:{self.port}"


def parse_hostfile(path: str) -> list[str]:
    """Parse an MPI/pdsh-style hostfile (`hostname [slots=N]`, `#` comments)."""
    hosts: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        hosts.append(line.split()[0])
    if not hosts:
        raise click.UsageError(f"hostfile {path} contains no hosts")
    return hosts


def cluster_up(
    hostfile: str | None = None,
    port: int | None = None,
    dashboard_port: int | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    runtime=None,
) -> None:
    """Start a Ray head on this node and join hostfile workers over ssh."""
    _require_ray_binary()

    cluster_cfg = runtime.ray.cluster if runtime and runtime.ray else None
    hostfile = hostfile or (cluster_cfg.hostfile if cluster_cfg else None)
    port = port or (cluster_cfg.head_port if cluster_cfg else 6379)
    dashboard_port = dashboard_port or (
        cluster_cfg.dashboard_port if cluster_cfg else 8265
    )
    ssh_user = ssh_user or (cluster_cfg.ssh_user if cluster_cfg else None)
    ssh_key = ssh_key or (cluster_cfg.ssh_key if cluster_cfg else None)

    head_ip = _primary_ip()
    workers = (
        [h for h in parse_hostfile(hostfile) if not _is_local_host(h)]
        if hostfile
        else []
    )

    if _probe_cluster(f"{head_ip}:{port}"):
        LOG.info("Ray head already running at %s:%s", head_ip, port)
        state = ClusterState.load() or ClusterState(
            head_ip=head_ip,
            port=port,
            dashboard_port=dashboard_port,
            temp_dir=str(TEMP_DIR_ROOT / "existing"),
        )
    else:
        temp_dir = TEMP_DIR_ROOT / f"cluster-{str(uuid.uuid4())[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        LOG.info(
            "starting Ray head on %s:%s (dashboard :%s)", head_ip, port, dashboard_port
        )
        subprocess.run(  # nosec B603 B607
            [
                "ray",
                "start",
                "--head",
                f"--port={port}",
                f"--dashboard-port={dashboard_port}",
                "--dashboard-host=0.0.0.0",
                f"--temp-dir={temp_dir}",
                "--disable-usage-stats",
            ],
            check=True,
        )
        state = ClusterState(
            head_ip=head_ip,
            port=port,
            dashboard_port=dashboard_port,
            temp_dir=str(temp_dir),
        )

    state.ssh_user = ssh_user
    state.ssh_key = ssh_key
    failures: list[tuple[str, str]] = []
    joined: list[str] = []
    for host in workers:
        LOG.info("joining worker %s", host)
        result = subprocess.run(  # nosec B603
            _ssh_cmd(host, ssh_user, ssh_key)
            + [
                "ray",
                "start",
                f"--address={state.address}",
                f"--temp-dir={state.temp_dir}",
                "--disable-usage-stats",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            joined.append(host)
        else:
            stderr_head = "\n".join(result.stderr.strip().splitlines()[:5])
            failures.append((host, stderr_head))
            LOG.error(
                "failed to join %s (exit %d):\n%s", host, result.returncode, stderr_head
            )

    state.workers = joined
    state.save()

    expected_nodes = 1 + len(joined)
    _wait_for_nodes(state.address, expected_nodes)

    if failures:
        for host, _ in failures:
            LOG.error(
                "worker %s did not join; passwordless ssh from the head is required"
                " (test with `ssh %s true`) and `ray` must be on its PATH",
                host,
                host,
            )
        raise SystemExit(1)
    LOG.info(
        "Ray cluster up: %d node(s); dashboard http://%s:%s — train with"
        " `--launcher ray` or submit jobs to http://%s:%s",
        expected_nodes,
        state.head_ip,
        state.dashboard_port,
        state.head_ip,
        state.dashboard_port,
    )


def cluster_down(force: bool = False) -> None:
    """Stop the recorded cluster (`ray stop` per node; --force pkills by temp-dir)."""
    _require_ray_binary()
    state = ClusterState.load()
    if state is None:
        LOG.warning(
            "no recorded cluster (%s); running local `ray stop` only", STATE_FILE
        )
        subprocess.run(["ray", "stop"], check=False)  # nosec B603 B607
        return

    for host in state.workers:
        LOG.info("stopping worker %s", host)
        cmd = _ssh_cmd(host, state.ssh_user, state.ssh_key)
        if force:
            # targeted: only daemons started from our unique temp dir
            cmd += ["pkill", "-9", "-f", Path(state.temp_dir).name]
        else:
            cmd += ["ray", "stop"]
        subprocess.run(cmd, check=False)  # nosec B603

    if force:
        subprocess.run(  # nosec B603 B607
            ["pkill", "-9", "-f", Path(state.temp_dir).name], check=False
        )
    else:
        subprocess.run(["ray", "stop"], check=False)  # nosec B603 B607

    STATE_FILE.unlink(missing_ok=True)
    shutil.rmtree(state.temp_dir, ignore_errors=True)
    LOG.info("Ray cluster stopped")


def cluster_status(address: str | None = None) -> None:
    """Show `ray status` for the recorded (or given) cluster."""
    _require_ray_binary()
    state = ClusterState.load()
    address = address or (state.address if state else None)
    cmd = ["ray", "status"]
    if address:
        cmd.append(f"--address={address}")
    result = subprocess.run(cmd, check=False)  # nosec B603 B607
    if result.returncode != 0 and address:
        raise click.ClickException(
            f"no Ray cluster reachable at {address}; start one with `axolotl ray up`"
        )
    if result.returncode != 0:
        raise click.ClickException(
            "no running Ray cluster found; start one with `axolotl ray up`"
        )


def _wait_for_nodes(address: str, expected: int, timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(  # nosec B603 B607
            ["ray", "status", f"--address={address}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.count("node_") >= expected:
            return
        time.sleep(3)
    LOG.warning(
        "cluster did not report %d node(s) within %ds; check `axolotl ray status`",
        expected,
        timeout,
    )


def _probe_cluster(address: str) -> bool:
    result = subprocess.run(  # nosec B603 B607
        ["ray", "status", f"--address={address}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _ssh_cmd(host: str, ssh_user: str | None, ssh_key: str | None) -> list[str]:
    cmd = list(_SSH_BASE)
    if ssh_key:
        cmd += ["-i", ssh_key]
    target = f"{ssh_user}@{host}" if ssh_user else host
    return cmd + [target]


def _is_local_host(host: str) -> bool:
    if host in ("localhost", "127.0.0.1", socket.gethostname(), _primary_ip()):
        return True
    try:
        return socket.gethostbyname(host) == _primary_ip()
    except OSError:
        return False


def _primary_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _require_ray_binary() -> None:
    if shutil.which("ray") is None:
        raise click.UsageError(
            "the `ray` CLI is not on PATH; install with `pip install axolotl[ray]`"
        )
