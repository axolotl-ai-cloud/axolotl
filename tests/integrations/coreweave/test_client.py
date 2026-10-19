"""Mock the documented management and rollout API contracts."""

from unittest.mock import MagicMock

import pytest

from axolotl.integrations.coreweave.client import CoreWeaveClient

from .test_checkpoints import config


def make_client(monkeypatch, responses):
    monkeypatch.setenv("CW_API_TOKEN", "test-token")
    session = MagicMock()
    session.request.side_effect = [MagicMock(json=lambda r=r: r) for r in responses]
    return CoreWeaveClient(config(poll_interval=0.001, timeout=0.05), session)


def operation(state="HOT_LOAD_STATE_COMPLETED"):
    return {
        "hotLoad": {
            "spec": {"id": "op", "identity": "next", "deploymentId": "deployment"},
            "status": {"state": state},
        }
    }


def ready():
    return {"state": "READY", "currentSnapshotIdentity": "next"}


def test_delta_request_waits_for_all_replica_pages(monkeypatch):
    client = make_client(
        monkeypatch,
        [
            operation(),
            operation(),
            {"replicas": [ready()], "nextPageToken": "page2"},
            {"replicas": [ready()]},
        ],
    )
    assert client.hot_load("next", base_identity="base") == "op"
    assert client.ready_identity == "next"
    calls = client.session.request.call_args_list
    body = calls[0].kwargs["json"]
    assert body["promptCachePolicy"] == "PROMPT_CACHE_POLICY_RESET_ALL"
    assert body["incrementalSnapshotMetadata"]["previousSnapshotIdentity"] == "base"
    assert calls[-1].kwargs["params"]["pageToken"] == "page2"


@pytest.mark.parametrize("state", ["HOT_LOAD_STATE_FAILED", "HOT_LOAD_STATE_CANCELED"])
def test_failed_hot_load_blocks_sampling(monkeypatch, state):
    client = make_client(monkeypatch, [operation(), operation(state)])
    client.ready_identity = "old"
    with pytest.raises(RuntimeError):
        client.hot_load("next")
    with pytest.raises(RuntimeError, match="convergence"):
        client.sample([], n=1, max_tokens=10, temperature=1)


def test_empty_or_mixed_replicas_are_not_ready(monkeypatch):
    client = make_client(
        monkeypatch,
        [
            {"replicas": []},
            {
                "replicas": [
                    ready(),
                    {"state": "READY", "currentSnapshotIdentity": "old"},
                ]
            },
        ],
    )
    assert not client._replicas_ready("/op", "next", float("inf"))
    assert not client._replicas_ready("/op", "next", float("inf"))


def test_sample_uses_exact_token_ids(monkeypatch):
    client = make_client(
        monkeypatch,
        [
            {
                "choices": [
                    {
                        "token_ids": [42, 7],
                        "logprobs": {"token_logprobs": [-0.5, -1.0]},
                    }
                ]
            }
        ],
    )
    client.ready_identity = "next"
    assert client.sample([11, 12], n=1, max_tokens=10, temperature=1) == [
        {"tokens": [42, 7], "logprobs": [-0.5, -1.0]}
    ]

    call = client.session.request.call_args
    assert call.args[1].endswith("/v1/completions")
    assert call.kwargs["json"]["prompt"] == [11, 12]
    assert call.kwargs["json"]["return_tokens_as_token_ids"] is True
    assert "messages" not in call.kwargs["json"]


@pytest.mark.parametrize("encoded", [["token_id:42", "token_id:7"], ["hello", "world"]])
def test_dynamo_token_id_logprobs(monkeypatch, encoded):
    client = make_client(
        monkeypatch,
        [
            {
                "choices": [
                    {
                        "logprobs": {
                            "tokens": encoded,
                            "token_logprobs": [-0.5, -1.0],
                        }
                    }
                ]
            }
        ],
    )
    client.ready_identity = "next"
    if encoded[0].startswith("token_id:"):
        assert client.sample([11, 12], n=1, max_tokens=10, temperature=1)[0][
            "tokens"
        ] == [42, 7]
    else:
        with pytest.raises(ValueError, match="exact completion token IDs"):
            client.sample([11, 12], n=1, max_tokens=10, temperature=1)


@pytest.mark.parametrize("tokens", [["1"], [True], [-1], []])
def test_invalid_prompt_tokens_are_rejected(monkeypatch, tokens):
    client = make_client(monkeypatch, [])
    client.ready_identity = "next"
    with pytest.raises(ValueError, match="token IDs"):
        client.sample(tokens, n=1, max_tokens=10, temperature=1)
    client.session.request.assert_not_called()


def test_unsupported_token_endpoint_does_not_fall_back_to_chat(monkeypatch):
    import requests

    client = make_client(monkeypatch, [])
    client.ready_identity = "next"
    response = requests.Response()
    response.status_code = 404
    client.session.request.side_effect = requests.HTTPError(response=response)
    with pytest.raises(RuntimeError, match="No chat fallback"):
        client.sample([11, 12], n=1, max_tokens=10, temperature=1)
    client.session.request.assert_called_once()
