# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CKS backend: manifests, output placement, and submission."""

import subprocess
from unittest.mock import MagicMock

import pytest
import yaml

from axolotl.integrations.coreweave.cloud import CoreWeaveCloud
from axolotl.integrations.coreweave.cloud_runner import launch_command


def provider(**kwargs):
    return CoreWeaveCloud(
        {
            "provider": "coreweave",
            "backend": "cks",
            "image": "axolotlai/axolotl-cloud:main-latest",
            "namespace": "tenant-a",
            "pvc": "shared",
            "nodes": 2,
            "job_name": "cks-test",
            "secret_env": {"HF_TOKEN": "hf-token/token"},
            "object_storage": {"org_id": "org1", "region": "US-EAST-15A"},
            **kwargs,
        }
    )


def test_dry_run_renders_service_configmap_and_indexed_job(capsys):
    result = provider(dry_run=True).train(
        "base_model: test\noutput_dir: outputs",
        launcher="torchrun",
        learning_rate=0.001,
    )
    service, configmap, job = result["manifests"]
    assert (service["kind"], configmap["kind"], job["kind"]) == (
        "Service",
        "ConfigMap",
        "Job",
    )
    assert service["spec"]["clusterIP"] == "None"
    spec = job["spec"]
    assert spec["completionMode"] == "Indexed" and spec["completions"] == 2
    pod = spec["template"]["spec"]
    container = pod["containers"][0]
    assert container["resources"]["requests"]["rdma/ib"] == "1"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert pod["subdomain"] == "cks-test"
    assert "podAntiAffinity" in pod["affinity"]
    env = {e["name"]: e for e in container["env"]}
    assert env["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "hf-token",
        "key": "token",
    }
    names = {v["name"] for v in pod["volumes"]}
    assert {"data", "job", "shm", "scratch", "caios-token"} <= names
    training = yaml.safe_load(configmap["data"]["train.yaml"])
    assert training["learning_rate"] == 0.001
    assert training["output_dir"] == "/data/axolotl-jobs/cks-test/outputs"
    entry = configmap["data"]["entrypoint.sh"]
    assert "MASTER_ADDR=cks-test-0.cks-test.tenant-a.svc.cluster.local" in entry
    assert "NCCL_IB_HCA=ibp" in entry
    assert "temporary-credentials/oidc/org1" in entry
    assert "---" in capsys.readouterr().out


def test_runner_derives_topology_from_pod_env():
    launch = {"launcher": "torchrun", "launcher_args": [], "gpus_per_node": 8}
    env = {
        "AXOLOTL_NNODES": "2",
        "AXOLOTL_NODE_RANK": "1",
        "MASTER_ADDR": "cks-test-0.cks-test.tenant-a.svc.cluster.local",
        "MASTER_PORT": "29400",
    }
    command = launch_command(launch, env)
    assert command[:5] == ["axolotl", "train", "train.yaml", "--launcher", "torchrun"]
    assert "--node_rank" in command and command[command.index("--node_rank") + 1] == "1"
    assert command[command.index("--nnodes") + 1] == "2"


def test_submit_applies_through_kubectl(monkeypatch):
    run = MagicMock(
        return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")
    )
    monkeypatch.setattr(subprocess, "run", run)
    result = provider(kubeconfig="/tmp/kc", context="use15a").train("base_model: test")
    lookup = run.call_args_list[0].args[0]
    assert lookup[7:] == ["get", "job", "cks-test", "--ignore-not-found", "-o", "name"]
    command = run.call_args.args[0]
    assert command[:7] == [
        "kubectl",
        "--kubeconfig",
        "/tmp/kc",
        "--context",
        "use15a",
        "-n",
        "tenant-a",
    ]
    assert command[7:] == ["apply", "-f", "-"]
    assert "kind: Job" in run.call_args.kwargs["input"]
    assert result["logs_command"].endswith("logs -f job/cks-test")


def test_existing_job_name_is_rejected_before_apply(monkeypatch):
    run = MagicMock(
        return_value=subprocess.CompletedProcess(
            [], 0, stdout="job.batch/cks-test\n", stderr=""
        )
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="already exists") as error:
        provider().train("base_model: test")
    assert "delete job,service,configmap cks-test" in str(error.value)
    assert run.call_count == 1


def test_generated_job_name_skips_existence_check(monkeypatch):
    run = MagicMock(
        return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")
    )
    monkeypatch.setattr(subprocess, "run", run)
    result = provider(job_name=None).train("base_model: test")
    assert run.call_count == 1
    assert run.call_args.args[0][-3:] == ["apply", "-f", "-"]
    assert result["job_name"].startswith("axolotl-")


def test_job_name_length_is_capped():
    with pytest.raises(ValueError, match="52"):
        provider(job_name="a" * 53, dry_run=True)


def test_cks_rejects_outputs_off_the_volume_and_topology_args():
    with pytest.raises(ValueError, match="pvc_mount"):
        provider(dry_run=True).train("base_model: test\noutput_dir: /tmp/out")
    with pytest.raises(ValueError, match="topology"):
        provider(dry_run=True).train(
            "base_model: test", launcher_args=["--nnodes", "4"]
        )
    with pytest.raises(ValueError, match="requires pvc"):
        CoreWeaveCloud(
            {"provider": "coreweave", "backend": "cks", "image": "x", "namespace": "n"}
        )
