"""A frozen-teacher forward server: the teacher in its own process.

Runs the FP teacher on its own device in its own process so its Python/kernel
enqueue never shares a thread (or a CUDA context) with the trainer. The trainer's
client sends `(inputs, spans)` windows — already sequence-fused — and receives the
final hidden states back, CPU-to-CPU over a `multiprocessing.connection` socket.

Run:  python -m axolotl.integrations.ternary.teacher_server \
          --model <name-or-path> --port 47831
"""

from __future__ import annotations

import argparse
from multiprocessing.connection import Listener

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

AUTHKEY: bytes = b"ternary-teacher"


def serve(model_name: str, port: int, dtype: torch.dtype = torch.bfloat16) -> None:
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = (
        AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    )
    LOG.info(f"ternary: teacher server ready on port {port} ({model_name}, {device})")

    from .distill import _forward_inputs, _last_hidden

    with Listener(("127.0.0.1", port), authkey=AUTHKEY) as listener:
        while True:
            with listener.accept() as conn:
                LOG.info("ternary: teacher server client connected")
                try:
                    while True:
                        request = conn.recv()
                        if request is None:
                            return
                        inputs = {
                            k: v.to(device, non_blocking=True)
                            for k, v in request["inputs"].items()
                        }
                        with torch.no_grad():
                            hidden = _last_hidden(
                                teacher(**_forward_inputs(inputs, teacher)), teacher
                            )
                        conn.send(hidden.to("cpu", dtype))
                except EOFError:
                    LOG.info("ternary: teacher server client disconnected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=47831)
    args = parser.parse_args()
    serve(args.model, args.port)


if __name__ == "__main__":
    main()
