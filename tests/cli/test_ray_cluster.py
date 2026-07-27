"""Unit tests for the axolotl ray cluster helpers."""

import click
import pytest

from axolotl.cli.launchers import ray_cluster


class TestParseHostfile:
    """Hostfile parsing."""

    def test_basic_hosts(self, tmp_path):
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("node1\nnode2\n")
        assert ray_cluster.parse_hostfile(str(hostfile)) == ["node1", "node2"]

    def test_slots_and_comments_ignored(self, tmp_path):
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("# cluster hosts\nnode1 slots=8\n\nnode2 slots=8  # gpu box\n")
        assert ray_cluster.parse_hostfile(str(hostfile)) == ["node1", "node2"]

    def test_empty_hostfile_raises(self, tmp_path):
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("# nothing here\n")
        with pytest.raises(click.UsageError, match="no hosts"):
            ray_cluster.parse_hostfile(str(hostfile))


class TestClusterState:
    """State file round-trip."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ray_cluster, "STATE_FILE", tmp_path / "state.json")
        state = ray_cluster.ClusterState(
            head_ip="10.0.0.4",
            port=6379,
            dashboard_port=8265,
            temp_dir="/tmp/cluster-abc",
            workers=["node2", "node3"],
            ssh_user="ubuntu",
        )
        state.save()
        loaded = ray_cluster.ClusterState.load()
        assert loaded == state
        assert loaded.address == "10.0.0.4:6379"

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ray_cluster, "STATE_FILE", tmp_path / "missing.json")
        assert ray_cluster.ClusterState.load() is None


class TestSshCmd:
    """ssh command construction."""

    def test_plain_host(self):
        cmd = ray_cluster._ssh_cmd("node1", None, None)
        assert cmd[0] == "ssh"
        assert cmd[-1] == "node1"
        assert "-i" not in cmd

    def test_user_and_key(self):
        cmd = ray_cluster._ssh_cmd("node1", "ubuntu", "/keys/id_ed25519")
        assert cmd[-1] == "ubuntu@node1"
        assert cmd[cmd.index("-i") + 1] == "/keys/id_ed25519"


class TestMissingRayBinary:
    """Friendly error when the ray CLI is absent."""

    def test_up_without_ray_binary(self, monkeypatch):
        monkeypatch.setattr(ray_cluster.shutil, "which", lambda _: None)
        with pytest.raises(click.UsageError, match="axolotl\\[ray\\]"):
            ray_cluster.cluster_up()

    def test_status_without_ray_binary(self, monkeypatch):
        monkeypatch.setattr(ray_cluster.shutil, "which", lambda _: None)
        with pytest.raises(click.UsageError, match="axolotl\\[ray\\]"):
            ray_cluster.cluster_status()
