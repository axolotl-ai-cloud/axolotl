"""Unit tests for the ray launcher backend (fake ray module, no ray install needed)."""

import sys
import types
from types import SimpleNamespace

import click
import pytest
import yaml

from axolotl.cli.launchers import ray_ as ray_launcher, ray_cluster
from axolotl.utils.schemas.runtime import RuntimeConfig


@pytest.fixture
def fake_ray(monkeypatch, tmp_path):
    """Install a minimal fake `ray` module and record init calls."""
    # keep tests independent of any cluster recorded by `axolotl ray up`
    monkeypatch.setattr(ray_cluster, "STATE_FILE", tmp_path / "no-cluster.json")
    state = SimpleNamespace(
        init_calls=[],
        shutdown_called=False,
        initialized=False,
        cluster_resources={"GPU": 8.0, "CPU": 64.0},
        auto_raises=False,
    )

    fake = types.ModuleType("ray")

    def init(*_args, **kwargs):
        if state.auto_raises and kwargs.get("address") == "auto":
            raise ConnectionError("no cluster")
        state.init_calls.append(kwargs)
        state.initialized = True

    fake.init = init
    fake.is_initialized = lambda: state.initialized
    fake.shutdown = lambda: setattr(state, "shutdown_called", True)
    fake.cluster_resources = lambda: dict(state.cluster_resources)

    monkeypatch.setitem(sys.modules, "ray", fake)
    return state


class TestEnsureRayInitialized:
    """ensure_ray_initialized behavior."""

    def test_short_circuits_when_initialized(self, fake_ray):
        fake_ray.initialized = True
        ray_launcher.ensure_ray_initialized(None)
        assert fake_ray.init_calls == []

    def test_explicit_address(self, fake_ray):
        ray_rt = RuntimeConfig.model_validate(
            {"ray": {"address": "ray://head:10001"}}
        ).ray
        ray_launcher.ensure_ray_initialized(ray_rt)
        assert fake_ray.init_calls[0]["address"] == "ray://head:10001"

    def test_runtime_env_lands_on_init(self, fake_ray):
        ray_rt = RuntimeConfig.model_validate(
            {
                "ray": {
                    "runtime_env": {
                        "env_vars": {"HF_HOME": "/mnt/hf"},
                        "pip": ["peft"],
                    }
                }
            }
        ).ray
        ray_launcher.ensure_ray_initialized(ray_rt, extra_env={"NCCL_DEBUG": "WARN"})
        runtime_env = fake_ray.init_calls[0]["runtime_env"]
        assert runtime_env["env_vars"] == {"NCCL_DEBUG": "WARN", "HF_HOME": "/mnt/hf"}
        assert runtime_env["pip"] == ["peft"]

    def test_no_address_uses_plain_init(self, fake_ray):
        # no address must never probe address="auto": auto attaches to other
        # drivers' embedded instances, coupling concurrent runs (CI regression)
        ray_launcher.ensure_ray_initialized(None)
        assert len(fake_ray.init_calls) == 1
        assert "address" not in fake_ray.init_calls[0]

    def test_explicit_auto_attaches(self, fake_ray):
        ray_rt = RuntimeConfig.model_validate({"ray": {"address": "auto"}}).ray
        ray_launcher.ensure_ray_initialized(ray_rt)
        assert fake_ray.init_calls[0]["address"] == "auto"

    def test_auto_attach_falls_back_to_local(self, fake_ray):
        fake_ray.auto_raises = True
        ray_rt = RuntimeConfig.model_validate({"ray": {"address": "auto"}}).ray
        ray_launcher.ensure_ray_initialized(ray_rt)
        assert len(fake_ray.init_calls) == 1
        assert "address" not in fake_ray.init_calls[0]

    def test_gpuless_cluster_restarts_local(self, fake_ray, monkeypatch):
        fake_ray.cluster_resources = {"CPU": 8.0}
        monkeypatch.setattr(ray_launcher, "_local_gpu_count", lambda: 4)
        ray_rt = RuntimeConfig.model_validate({"ray": {"address": "auto"}}).ray
        ray_launcher.ensure_ray_initialized(ray_rt)
        assert fake_ray.shutdown_called
        # first init attached with address=auto, second init started local
        assert fake_ray.init_calls[0]["address"] == "auto"
        assert "address" not in fake_ray.init_calls[1]

    def test_recorded_cluster_preferred(self, fake_ray):
        import socket

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            ray_cluster.ClusterState(
                head_ip="127.0.0.1", port=port, dashboard_port=8265, temp_dir="/tmp/x"
            ).save()
            ray_launcher.ensure_ray_initialized(None)
            assert fake_ray.init_calls[0]["address"] == f"127.0.0.1:{port}"
        finally:
            listener.close()

    def test_stale_recorded_cluster_ignored(self, fake_ray):
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
        sock.close()
        ray_cluster.ClusterState(
            head_ip="127.0.0.1", port=dead_port, dashboard_port=8265, temp_dir="/tmp/x"
        ).save()
        ray_launcher.ensure_ray_initialized(None)
        assert "address" not in fake_ray.init_calls[0]


class TestResolveNumWorkers:
    """resolve_num_workers derivation."""

    def test_explicit_int_passthrough(self, fake_ray):
        assert ray_launcher.resolve_num_workers(4, {"GPU": 1}) == 4

    def test_auto_from_gpus(self, fake_ray):
        assert ray_launcher.resolve_num_workers("auto", {"GPU": 1}) == 8

    def test_auto_fractional_binding_resource(self, fake_ray):
        fake_ray.cluster_resources = {
            "GPU": 8.0,
            "CPU": 64.0,
            "accelerator_type:H100": 0.008,
        }
        workers = ray_launcher.resolve_num_workers(
            "auto", {"GPU": 1, "accelerator_type:H100": 0.001}
        )
        assert workers == 8

    def test_auto_unsatisfiable_raises(self, fake_ray):
        fake_ray.cluster_resources = {"CPU": 8.0}
        with pytest.raises(click.UsageError, match="cannot satisfy"):
            ray_launcher.resolve_num_workers("auto", {"GPU": 1})


class TestRuntimeRewrite:
    """_rewrite_runtime_for_cluster + _build_entrypoint."""

    def test_rewrite_strips_submission_fields(self):
        runtime = RuntimeConfig.model_validate(
            {
                "launcher": "ray",
                "env": {"WANDB_PROJECT": "proj"},
                "ray": {
                    "address": "http://head:8265",
                    "detach": True,
                    "num_workers": "auto",
                    "runtime_env": {"working_dir": ".", "pip": ["peft"]},
                },
            }
        )
        data = ray_launcher._rewrite_runtime_for_cluster(runtime, {})
        assert data["launcher"] == "ray"
        assert "address" not in data["ray"]
        assert "detach" not in data["ray"]
        assert "runtime_env" not in data["ray"]
        assert "env" not in data
        assert data["ray"]["num_workers"] == "auto"

    def test_rewrite_folds_cli_overrides(self):
        runtime = RuntimeConfig.model_validate(
            {"launcher": "ray", "ray": {"num_workers": "auto"}}
        )
        kwargs = {"ray_num_workers": 4, "resources_per_worker": {"GPU": 1}}
        data = ray_launcher._rewrite_runtime_for_cluster(runtime, kwargs)
        assert data["ray"]["num_workers"] == 4
        assert data["ray"]["resources_per_worker"] == {"GPU": 1}

    def test_rewrite_drops_foreign_blocks(self):
        runtime = RuntimeConfig.model_validate(
            {"launcher": "ray", "torchrun": {"nnodes": 2}}
        )
        data = ray_launcher._rewrite_runtime_for_cluster(runtime, {})
        assert "torchrun" not in data

    def test_entrypoint_forwards_non_ray_kwargs(self):
        entrypoint = ray_launcher._build_entrypoint(
            {"use_ray": True, "ray_num_workers": 2, "learning_rate": "1e-4"}
        )
        assert entrypoint.startswith("axolotl train axolotl_job_config.yaml")
        assert "--runtime axolotl_job_runtime.yaml" in entrypoint
        assert "--learning-rate=1e-4" in entrypoint
        assert "use-ray" not in entrypoint
        assert "ray-num-workers" not in entrypoint


class TestPrepareJob:
    """_prepare_job staging."""

    def test_staging_without_working_dir(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("base_model: test\n")
        runtime = RuntimeConfig.model_validate(
            {"launcher": "ray", "ray": {"address": "http://head:8265"}}
        )
        staging, entrypoint, runtime_env = ray_launcher._prepare_job(
            str(cfg_file), {}, runtime, env={"WANDB_PROJECT": "proj"}
        )
        try:
            assert (staging / "axolotl_job_config.yaml").exists()
            staged_runtime = yaml.safe_load(
                (staging / "axolotl_job_runtime.yaml").read_text()
            )
            assert "address" not in staged_runtime["ray"]
            assert runtime_env["working_dir"] == str(staging)
            assert runtime_env["env_vars"]["AXOLOTL_RAY_SUBMITTED"] == "1"
            assert runtime_env["env_vars"]["WANDB_PROJECT"] == "proj"
            assert "axolotl_job_config.yaml" in entrypoint
        finally:
            import shutil

            shutil.rmtree(staging, ignore_errors=True)

    def test_staging_with_working_dir(self, tmp_path):
        workdir = tmp_path / "project"
        workdir.mkdir()
        (workdir / "data.jsonl").write_text("{}\n")
        (workdir / "secret.bin").write_text("x")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("base_model: test\n")
        runtime = RuntimeConfig.model_validate(
            {
                "launcher": "ray",
                "ray": {
                    "address": "http://head:8265",
                    "runtime_env": {
                        "working_dir": str(workdir),
                        "excludes": ["*.bin"],
                    },
                },
            }
        )
        staging, _, runtime_env = ray_launcher._prepare_job(
            str(cfg_file), {}, runtime, env=None
        )
        try:
            assert (staging / "data.jsonl").exists()
            assert not (staging / "secret.bin").exists()
            assert "working_dir" not in runtime_env or runtime_env[
                "working_dir"
            ] == str(staging)
        finally:
            import shutil

            shutil.rmtree(staging, ignore_errors=True)


class TestSubmitRayJob:
    """submit_ray_job flow with a fake JobSubmissionClient."""

    @pytest.fixture
    def fake_job_submission(self, fake_ray, monkeypatch):
        state = SimpleNamespace(
            submitted=None,
            stopped=None,
            status="SUCCEEDED",
            logs=["line1\n", "line2\n"],
        )
        job_status = SimpleNamespace(SUCCEEDED="SUCCEEDED", FAILED="FAILED")

        class FakeClient:
            def __init__(self, address):
                state.address = address

            def submit_job(self, entrypoint, runtime_env=None, metadata=None):
                state.submitted = SimpleNamespace(
                    entrypoint=entrypoint, runtime_env=runtime_env, metadata=metadata
                )
                return "raysubmit_123"

            def tail_job_logs(self, submission_id):
                async def agen():
                    for line in state.logs:
                        yield line

                return agen()

            def get_job_status(self, submission_id):
                return state.status

            def stop_job(self, submission_id):
                state.stopped = submission_id

        module = types.ModuleType("ray.job_submission")
        module.JobSubmissionClient = FakeClient
        module.JobStatus = job_status
        monkeypatch.setitem(sys.modules, "ray.job_submission", module)
        return state

    @pytest.fixture
    def submit_setup(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("base_model: test\n")
        runtime = RuntimeConfig.model_validate(
            {"launcher": "ray", "ray": {"address": "http://head:8265"}}
        )
        return str(cfg_file), runtime

    def test_successful_submission(self, fake_job_submission, submit_setup, capsys):
        cfg_file, runtime = submit_setup
        submission_id = ray_launcher.submit_ray_job(cfg_file, {}, runtime)
        assert submission_id == "raysubmit_123"
        assert fake_job_submission.address == "http://head:8265"
        submitted = fake_job_submission.submitted
        assert "axolotl train axolotl_job_config.yaml" in submitted.entrypoint
        assert submitted.runtime_env["env_vars"]["AXOLOTL_RAY_SUBMITTED"] == "1"
        assert "line1" in capsys.readouterr().out

    def test_failed_job_exits_nonzero(self, fake_job_submission, submit_setup):
        cfg_file, runtime = submit_setup
        fake_job_submission.status = "FAILED"
        with pytest.raises(SystemExit) as exc_info:
            ray_launcher.submit_ray_job(cfg_file, {}, runtime)
        assert exc_info.value.code == 1

    def test_detach_skips_log_tail(self, fake_job_submission, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("base_model: test\n")
        runtime = RuntimeConfig.model_validate(
            {"launcher": "ray", "ray": {"address": "http://head:8265", "detach": True}}
        )
        submission_id = ray_launcher.submit_ray_job(str(cfg_file), {}, runtime)
        assert submission_id == "raysubmit_123"
        assert fake_job_submission.stopped is None

    def test_refuses_resubmission_inside_job(self, submit_setup, monkeypatch):
        cfg_file, runtime = submit_setup
        monkeypatch.setenv("AXOLOTL_RAY_SUBMITTED", "1")
        with pytest.raises(RuntimeError, match="refusing to submit"):
            ray_launcher.launch_ray_training(cfg_file, {}, runtime)


class TestRequireRay:
    """Missing-ray UX (real environment has no ray installed)."""

    def test_missing_ray_raises_usage_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ray", None)  # forces ImportError
        with pytest.raises(click.UsageError, match="axolotl\\[ray\\]"):
            ray_launcher._require_ray()
