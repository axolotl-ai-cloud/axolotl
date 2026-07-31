"""The remote teacher: server round-trip, client ordering, head partial-load."""

import threading

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary.distill import (
    _forward_inputs,
    _last_hidden,
    _load_lm_head_weight,
    _RemoteTeacherClient,
)
from axolotl.integrations.ternary.teacher_server import AUTHKEY


def _tiny_teacher() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=48,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    ).eval()


def _serve_in_thread(teacher, port: int) -> threading.Thread:
    from multiprocessing.connection import Listener

    def _loop():
        with Listener(("127.0.0.1", port), authkey=AUTHKEY) as listener:
            with listener.accept() as conn:
                try:
                    while True:
                        request = conn.recv()
                        if request is None:
                            return
                        with torch.no_grad():
                            hidden = _last_hidden(
                                teacher(**_forward_inputs(request["inputs"], teacher)),
                                teacher,
                            )
                        conn.send(hidden)
                except EOFError:
                    return

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _packed_batch(width: int, seed: int) -> dict:
    torch.manual_seed(seed)
    return {
        "input_ids": torch.randint(0, 48, (1, width)),
        "attention_mask": torch.ones(1, width, dtype=torch.long),
        "position_ids": torch.arange(width).reshape(1, width),
    }


def test_client_round_trips_fused_windows():
    teacher = _tiny_teacher()
    port = 49131
    _serve_in_thread(teacher, port)

    client = _RemoteTeacherClient(f"127.0.0.1:{port}", device="cpu")
    batches = [_packed_batch(8, 1), _packed_batch(8, 2), _packed_batch(12, 3)]
    client.submit_many(batches)

    for batch in batches:
        hidden = client.take(batch, timeout=30.0)
        assert hidden.shape[:2] == (1, batch["input_ids"].shape[1])
    client.close()


def test_out_of_order_take_raises():
    teacher = _tiny_teacher()
    port = 49132
    _serve_in_thread(teacher, port)

    client = _RemoteTeacherClient(f"127.0.0.1:{port}", device="cpu")
    first, second = _packed_batch(8, 4), _packed_batch(8, 5)
    client.submit_many([first, second])

    with pytest.raises(RuntimeError, match="order broke"):
        client.take(second, timeout=30.0)
    client.close()


def test_unreachable_endpoint_message(monkeypatch):
    monkeypatch.setattr(
        "axolotl.integrations.ternary.distill.time.sleep", lambda _s: None
    )
    with pytest.raises(ConnectionError, match="not reachable"):
        _RemoteTeacherClient("127.0.0.1:49999", device="cpu")


def test_head_partial_load(tmp_path):
    from safetensors.torch import save_file

    teacher = _tiny_teacher()
    save_file(
        {"lm_head.weight": teacher.lm_head.weight.detach().contiguous()},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text("{}")

    weight = _load_lm_head_weight(str(tmp_path), device="cpu")
    assert torch.equal(weight, teacher.lm_head.weight.detach())
