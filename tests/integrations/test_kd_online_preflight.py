"""
Tests for the online-teacher startup preflight and for failing fast on a teacher that
rejects the request, including from inside a dataloader worker.
"""

import gc
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import orjson
import pytest
import requests
from torch.utils.data import DataLoader, Dataset

from axolotl.integrations.kd.collator_online_teacher import (
    OnlineTeacherCollator,
    TeacherRequestError,
)

TOP_K = 8
MAX_LOGPROBS_ERROR = (
    "Requested prompt logprobs of 8, which is greater than max allowed: 5"
)


def fake_tokenizer():
    tokenizer = MagicMock()
    tokenizer.deprecation_warnings = {}
    tokenizer.padding_side = "right"
    return tokenizer


def fake_response(payload, status_code=200):
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.content = orjson.dumps(payload)
    response.text = json.dumps(payload)
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def probe_rows(width=TOP_K, prompt_len=3, empty=False):
    rows = [None]
    for _ in range(prompt_len - 1):
        rows.append(
            {}
            if empty
            else {str(1000 + i): {"logprob": -float(i)} for i in range(width)}
        )
    return rows


def vllm_probe_response(**kwargs):
    return {"choices": [{"prompt_logprobs": probe_rows(**kwargs)}]}


def sglang_probe_response(width=TOP_K, prompt_len=3):
    rows = [None]
    for _ in range(prompt_len - 1):
        rows.append([[-float(i), 1000 + i, f"<{i}>"] for i in range(width)])
    return [{"meta_info": {"input_top_logprobs": rows}}]


@pytest.fixture(name="stub_session")
def stub_session_fixture(monkeypatch):
    """Pin a stub HTTP session onto the collator class, so even the preflight uses it."""
    session = MagicMock(spec=requests.Session)
    monkeypatch.setattr(
        OnlineTeacherCollator, "http_session", property(lambda self: session)
    )
    return session


def build_collator(session, response=None, exc=None, server="vllm", preflight=True):
    if exc is not None:
        session.post.side_effect = exc
    else:
        session.post.return_value = response

    return OnlineTeacherCollator(
        fake_tokenizer(),
        kd_online_server_base_url="http://teacher:8000",
        kd_online_topk=TOP_K,
        kd_online_server=server,
        kd_online_preflight=preflight,
    )


def test_preflight_surfaces_the_servers_error_and_the_remedy(stub_session):
    body = {"object": "error", "message": MAX_LOGPROBS_ERROR, "code": 400}

    with pytest.raises(TeacherRequestError) as excinfo:
        build_collator(stub_session, response=fake_response(body, status_code=400))

    message = str(excinfo.value)
    assert MAX_LOGPROBS_ERROR in message
    assert "--max-logprobs >= kd_online_topk (8)" in message
    assert "--no-enable-prefix-caching" in message


def test_preflight_success_sends_one_probe_and_returns(stub_session):
    collator = build_collator(
        stub_session, response=fake_response(vllm_probe_response())
    )

    assert stub_session.post.call_count == 1
    _, kwargs = stub_session.post.call_args
    assert kwargs["json"] == {
        "prompt": [[1, 2, 3]],
        "max_tokens": 0,
        "echo": True,
        "prompt_logprobs": TOP_K,
    }
    assert collator.kd_online_topk == TOP_K


def test_preflight_rejects_a_server_that_caps_logprobs_silently(stub_session):
    with pytest.raises(TeacherRequestError, match="only 5 logprobs per position"):
        build_collator(
            stub_session, response=fake_response(vllm_probe_response(width=5))
        )


def test_preflight_rejects_empty_prompt_logprobs(stub_session):
    with pytest.raises(TeacherRequestError, match="prefix-cached"):
        build_collator(
            stub_session, response=fake_response(vllm_probe_response(empty=True))
        )


def test_preflight_rejects_a_short_logprob_list(stub_session):
    with pytest.raises(TeacherRequestError, match="did not return per-token logprobs"):
        build_collator(
            stub_session, response=fake_response(vllm_probe_response(prompt_len=2))
        )


def test_preflight_rejects_a_response_without_prompt_logprobs(stub_session):
    with pytest.raises(TeacherRequestError, match="did not return per-token logprobs"):
        build_collator(
            stub_session, response=fake_response({"choices": [{"text": ""}]})
        )


def test_preflight_reports_an_unreachable_teacher(stub_session):
    with pytest.raises(TeacherRequestError, match="could not reach the online teacher"):
        build_collator(stub_session, exc=requests.exceptions.ConnectionError("refused"))


def test_preflight_sglang_success_and_failure(stub_session):
    build_collator(
        stub_session, response=fake_response(sglang_probe_response()), server="sglang"
    )
    _, kwargs = stub_session.post.call_args
    assert kwargs["json"]["top_logprobs_num"] == TOP_K
    assert kwargs["json"]["sampling_params"] == {"max_new_tokens": 0}

    with pytest.raises(TeacherRequestError, match="only 2 logprobs per position"):
        build_collator(
            stub_session,
            response=fake_response(sglang_probe_response(width=2)),
            server="sglang",
        )


def test_preflight_can_be_disabled(stub_session):
    collator = build_collator(stub_session, response=None, preflight=False)

    assert stub_session.post.call_count == 0
    assert collator.kd_online_topk == TOP_K


def test_client_errors_are_not_retried_during_collation(stub_session):
    body = {"object": "error", "message": MAX_LOGPROBS_ERROR, "code": 400}
    collator = build_collator(
        stub_session, response=fake_response(vllm_probe_response())
    )
    stub_session.post.reset_mock()
    stub_session.post.return_value = fake_response(body, status_code=400)

    with pytest.raises(TeacherRequestError) as excinfo:
        collator._attach_teacher_logprobs(
            [{"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, 4]}]
        )

    assert stub_session.post.call_count == 1, "a rejected request must not be retried"
    assert MAX_LOGPROBS_ERROR in str(excinfo.value)


def test_http_session_is_not_shared_across_processes(monkeypatch):
    collator = OnlineTeacherCollator(
        fake_tokenizer(),
        kd_online_server_base_url="http://teacher:8000",
        kd_online_topk=TOP_K,
        kd_online_preflight=False,
    )

    monkeypatch.setattr("os.getpid", lambda: 111)
    parent_session = collator.http_session
    assert collator.http_session is parent_session

    monkeypatch.setattr("os.getpid", lambda: 222)
    assert collator.http_session is not parent_session


class _RejectingTeacher(BaseHTTPRequestHandler):
    """Stands in for a vllm server whose --max-logprobs is below kd_online_topk."""

    protocol_version = "HTTP/1.1"
    body = json.dumps(
        {"object": "error", "message": MAX_LOGPROBS_ERROR, "code": 400}
    ).encode()

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


class _TinyDataset(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, index):
        return {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, 4]}


@pytest.fixture(name="rejecting_teacher", scope="module")
def rejecting_teacher_fixture():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RejectingTeacher)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


# one worker is enough to cross the process boundary; each extra worker adds ~5s of
# dataloader shutdown to this failure path
@pytest.mark.parametrize("num_workers", [0, 1])
def test_rejected_request_fails_the_run_from_a_dataloader_worker(
    rejecting_teacher, num_workers
):
    """
    Regression for the observed hang: a persistently rejecting teacher must terminate
    iteration promptly, with the server's message, whether collation happens in the main
    process or in a dataloader worker.
    """
    collator = OnlineTeacherCollator(
        fake_tokenizer(),
        kd_online_server_base_url=rejecting_teacher,
        kd_online_topk=TOP_K,
        kd_online_timeout=2,
        kd_online_preflight=False,
    )
    loader = DataLoader(
        _TinyDataset(),
        batch_size=2,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers else None,
    )

    started = time.perf_counter()
    error = None
    elapsed = None
    iterator = iter(loader)
    try:
        for _ in iterator:
            pass
    except TeacherRequestError as exc:
        error = str(exc)
        elapsed = time.perf_counter() - started
    finally:
        # drop the iterator here so its worker processes (which inherited this test's
        # listening socket) are torn down before the next test runs
        del iterator
        gc.collect()

    assert error is not None, "iteration completed without surfacing the teacher error"
    assert MAX_LOGPROBS_ERROR in error
    assert elapsed < 30, f"failure took {elapsed:.1f}s, the run should die promptly"


def test_preflight_fails_the_run_before_any_dataloader_exists(rejecting_teacher):
    with pytest.raises(TeacherRequestError) as excinfo:
        OnlineTeacherCollator(
            fake_tokenizer(),
            kd_online_server_base_url=rejecting_teacher,
            kd_online_topk=TOP_K,
            kd_online_timeout=10,
        )

    assert MAX_LOGPROBS_ERROR in str(excinfo.value)
    assert "--max-logprobs" in str(excinfo.value)
