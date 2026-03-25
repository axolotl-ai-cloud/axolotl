"""vLLM serve script with native LoRA adapter support.

Extends TRL's vllm_serve to enable direct LoRA adapter loading in vLLM,
instead of merging adapter weights into the base model before syncing.

Usage:
    Set ``vllm.serve_module: axolotl.scripts.vllm_serve_lora`` in your config,
    or ``trl.vllm_lora_sync: true`` to auto-select.

Benefits over merge-sync:
    - Syncs only LoRA adapter weights via filesystem instead of full merged model via NCCL
    - vLLM handles LoRA application natively (Punica kernels)
    - No NCCL communicator needed for weight sync
"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any

from trl.scripts.vllm_serve import (
    ScriptArguments,
    chunk_list,
    extract_logprobs,
    get_open_port,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


@dataclass
class LoRAScriptArguments(ScriptArguments):
    """Extended script arguments with LoRA support."""

    enable_lora: bool = field(
        default=True,
        metadata={"help": "Enable LoRA adapter support in vLLM."},
    )
    max_lora_rank: int = field(
        default=64,
        metadata={"help": "Maximum LoRA rank supported."},
    )
    max_loras: int = field(
        default=2,
        metadata={"help": "Maximum number of LoRA adapters loaded simultaneously."},
    )
    lora_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Data type for LoRA weights."},
    )


def llm_worker(
    script_args: LoRAScriptArguments,
    data_parallel_rank: int,
    master_port: int,
    connection: Connection,
) -> None:
    """Worker process that creates a vLLM LLM with LoRA enabled."""
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        # Use batch-capable worker extension (adds batch_update_named_params + auto-close)
        worker_extension_cls="axolotl.scripts.vllm_worker_ext.BatchWeightSyncWorkerExtension",
        trust_remote_code=script_args.trust_remote_code,
        model_impl=script_args.vllm_model_impl,
        logprobs_mode="processed_logprobs",
        # LoRA
        enable_lora=script_args.enable_lora,
        max_lora_rank=script_args.max_lora_rank,
        max_loras=script_args.max_loras,
        lora_dtype=script_args.lora_dtype,
    )

    connection.send({"status": "ready"})

    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            llm.collective_rpc(method="close_communicator")
            break

        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args = command.get("args", ())
            kwargs = command.get("kwargs", {})

            # Reconstruct LoRARequest from serialized dict (can't pickle across pipe)
            if "lora_request" in kwargs and kwargs["lora_request"] is not None:
                lr = kwargs["lora_request"]
                kwargs["lora_request"] = LoRARequest(
                    lora_name=lr["lora_name"],
                    lora_int_id=lr["lora_int_id"],
                    lora_path=lr["lora_path"],
                    load_inplace=lr.get("load_inplace", False),
                )

            try:
                method = getattr(llm, method_name)
                result = method(*args, **kwargs)
            except Exception as exc:
                logger.warning("Worker method %s failed: %s", method_name, exc)
                if command["type"] == "call":
                    connection.send({"error": str(exc), "kind": "worker_error"})
                continue
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def main(script_args: ScriptArguments):
    """Start vLLM workers with LoRA support and the HTTP server."""
    import asyncio

    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel, Field as PydanticField

    # Request/Response models (defined locally like TRL's vllm_serve.main)
    class GenerateRequest(BaseModel):
        prompts: list[str]
        images: list[str] | None = None
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        logprobs: int | None = 0
        truncate_prompt_tokens: int | None = None
        structured_outputs_regex: str | None = None
        generation_kwargs: dict = PydanticField(default_factory=dict)

    class GenerateResponse(BaseModel):
        prompt_ids: list[list[int]]
        completion_ids: list[list[int]]
        logprobs: list[list[list[float]]]
        logprob_token_ids: list[list[list[int]]]

    class ChatRequest(BaseModel):
        messages: list[list[dict]]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        logprobs: int | None = 0
        truncate_prompt_tokens: int | None = None
        structured_outputs_regex: str | None = None
        generation_kwargs: dict = PydanticField(default_factory=dict)
        chat_template_kwargs: dict = PydanticField(default_factory=dict)

    class ChatResponse(BaseModel):
        prompt_ids: list[list[int]]
        completion_ids: list[list[int]]
        logprobs: list[list[list[float]]]
        logprob_token_ids: list[list[list[int]]]

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int
        client_device_uuid: str

    # Wrap plain ScriptArguments with LoRA defaults
    if not isinstance(script_args, LoRAScriptArguments):
        lora_args = LoRAScriptArguments.__new__(LoRAScriptArguments)
        for f in ScriptArguments.__dataclass_fields__:
            setattr(lora_args, f, getattr(script_args, f))
        # Apply LoRA defaults
        for f in LoRAScriptArguments.__dataclass_fields__:
            if f not in ScriptArguments.__dataclass_fields__:
                setattr(
                    lora_args, f, LoRAScriptArguments.__dataclass_fields__[f].default
                )
        script_args = lora_args

    # Spawn workers
    master_port = get_open_port()
    connections: list[Connection] = []
    processes: list[Process] = []
    for dp_rank in range(script_args.data_parallel_size):
        parent_conn, child_conn = Pipe()
        process = Process(
            target=llm_worker,
            args=(script_args, dp_rank, master_port, child_conn),
        )
        process.start()
        connections.append(parent_conn)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import time

        startup_timeout = 300  # 5 minutes
        start_time = time.monotonic()
        ready: set[int] = set()
        while len(ready) < script_args.data_parallel_size:
            elapsed = time.monotonic() - start_time
            if elapsed > startup_timeout:
                raise RuntimeError(
                    f"vLLM workers failed to start within {startup_timeout}s "
                    f"({len(ready)}/{script_args.data_parallel_size} ready)"
                )
            for i, (conn, proc) in enumerate(zip(connections, processes, strict=True)):
                if id(conn) in ready:
                    continue
                if not proc.is_alive():
                    raise RuntimeError(
                        f"vLLM worker {i} exited unexpectedly during startup"
                    )
                if conn.poll():
                    msg = conn.recv()
                    if isinstance(msg, dict) and msg.get("status") == "ready":
                        ready.add(id(conn))
            await asyncio.sleep(0.1)
        yield
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()

    app = FastAPI(lifespan=lifespan)

    # --- Access logging middleware ---
    import time as _time

    @app.middleware("http")
    async def access_log_middleware(request, call_next):
        t0 = _time.monotonic()
        response = await call_next(request)
        elapsed = _time.monotonic() - t0
        logger.info(
            "%s %s %d %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    # --- Active LoRA state (shared across endpoints via closure) ---
    active_lora: dict = {"request": None}

    # ------------------------------------------------------------------
    # LoRA-specific endpoints
    # ------------------------------------------------------------------

    class SetLoRARequest(BaseModel):
        lora_name: str
        lora_int_id: int
        lora_path: str
        load_inplace: bool = False

    @app.post("/set_lora_adapter/")
    async def set_lora_adapter(request: SetLoRARequest):
        """Register a LoRA adapter for all subsequent generate/chat calls."""
        active_lora["request"] = {
            "lora_name": request.lora_name,
            "lora_int_id": request.lora_int_id,
            "lora_path": request.lora_path,
            "load_inplace": request.load_inplace,
        }
        logger.info(
            "Set active LoRA: %s (id=%d, path=%s)",
            request.lora_name,
            request.lora_int_id,
            request.lora_path,
        )
        return {"status": "ok"}

    @app.post("/clear_lora_adapter/")
    async def clear_lora_adapter():
        """Clear active LoRA adapter (revert to base model)."""
        active_lora["request"] = None
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Standard endpoints (mirrors TRL's vllm_serve)
    # ------------------------------------------------------------------

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        return {
            "world_size": script_args.tensor_parallel_size
            * script_args.data_parallel_size
        }

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate completions with optional LoRA adapter."""
        import base64
        from io import BytesIO

        import vllm
        from packaging.version import Version

        try:
            from vllm.sampling_params import GuidedDecodingParams
        except ImportError:
            GuidedDecodingParams = None  # not available in vLLM 0.17+

        images: list[str | None] = request.images or [None] * len(request.prompts)  # type: ignore[assignment,list-item]
        prompts: list[dict[str, Any]] = []
        for prompt, image in zip(request.prompts, images, strict=True):
            row: dict[str, Any] = {"prompt": prompt}
            if image is not None:
                from PIL import Image

                row["multi_modal_data"] = {
                    "image": Image.open(BytesIO(base64.b64decode(image)))
                }
            prompts.append(row)

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "logprobs": request.logprobs,
        }
        generation_kwargs.update(request.generation_kwargs)

        if Version(vllm.__version__) <= Version("0.10.2"):
            key = "guided_decoding"
            if request.structured_outputs_regex is not None:
                generation_kwargs[key] = GuidedDecodingParams(
                    regex=request.structured_outputs_regex
                )
            else:
                generation_kwargs.setdefault(key, None)
        else:
            from vllm.sampling_params import StructuredOutputsParams

            key = "structured_outputs"
            if request.structured_outputs_regex is not None:
                generation_kwargs[key] = StructuredOutputsParams(
                    regex=request.structured_outputs_regex
                )
            elif isinstance(generation_kwargs.get(key), dict):
                generation_kwargs[key] = StructuredOutputsParams(
                    **generation_kwargs[key]
                )
            else:
                generation_kwargs.setdefault(key, None)

        sampling_params = SamplingParams(**generation_kwargs)
        chunked_prompts = chunk_list(prompts, script_args.data_parallel_size)

        for conn, chunk in zip(connections, chunked_prompts, strict=True):
            if not chunk:
                chunk = [{"prompt": "<placeholder>"}]
            kwargs = {
                "prompts": chunk,
                "sampling_params": sampling_params,
                "lora_request": active_lora["request"],
            }
            conn.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Use run_in_executor so blocking recv() doesn't freeze the event loop
        # (allows /set_lora_adapter/ and other endpoints to be served concurrently)
        loop = asyncio.get_running_loop()
        all_outputs = await asyncio.gather(
            *(loop.run_in_executor(None, conn.recv) for conn in connections)
        )
        all_outputs = [
            o for o, c in zip(all_outputs, chunked_prompts, strict=True) if c
        ]
        all_outputs = list(chain.from_iterable(all_outputs))

        return {
            "prompt_ids": [o.prompt_token_ids for o in all_outputs],
            "completion_ids": [
                list(out.token_ids) for o in all_outputs for out in o.outputs
            ],
            "logprobs": extract_logprobs(all_outputs)[0],
            "logprob_token_ids": extract_logprobs(all_outputs)[1],
        }

    @app.post("/chat/", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Chat endpoint with optional LoRA adapter."""
        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "logprobs": request.logprobs,
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)
        chunked = chunk_list(request.messages, script_args.data_parallel_size)
        for conn, chunk in zip(connections, chunked, strict=True):
            if not chunk:
                chunk = [[{"role": "user", "content": "<placeholder>"}]]
            kwargs = {
                "messages": chunk,
                "sampling_params": sampling_params,
                "use_tqdm": False,
                "lora_request": active_lora["request"],
            }
            conn.send({"type": "call", "method": "chat", "kwargs": kwargs})

        loop = asyncio.get_running_loop()
        all_outputs = await asyncio.gather(
            *(loop.run_in_executor(None, conn.recv) for conn in connections)
        )
        all_outputs = [o for o, c in zip(all_outputs, chunked, strict=True) if c]
        all_outputs = list(chain.from_iterable(all_outputs))

        return {
            "prompt_ids": [o.prompt_token_ids for o in all_outputs],
            "completion_ids": [
                list(out.token_ids) for o in all_outputs for out in o.outputs
            ],
            "logprobs": extract_logprobs(all_outputs)[0],
            "logprob_token_ids": extract_logprobs(all_outputs)[1],
        }

    # --- OpenAI-compatible endpoints (for NeMo Gym agent integration) ---

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible models endpoint."""
        return {
            "object": "list",
            "data": [
                {"id": script_args.model, "object": "model", "owned_by": "axolotl"}
            ],
        }

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request_body: dict):
        """OpenAI-compatible chat completions endpoint.

        Translates OpenAI format to our internal /chat/ format so NeMo Gym's
        model server proxy can call us directly.
        """
        messages_list = request_body.get("messages", [])
        temperature = request_body.get("temperature", 1.0)
        max_tokens = request_body.get("max_tokens", 512)
        top_p = request_body.get("top_p", 1.0)
        n = request_body.get("n", 1)

        generation_kwargs = {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "logprobs": 0,  # Always return logprobs (NeMo Gym needs them)
        }
        sampling_params = SamplingParams(
            **{k: v for k, v in generation_kwargs.items() if v is not None}
        )

        # Send to vLLM worker
        chunked = chunk_list([messages_list], script_args.data_parallel_size)
        for conn, chunk in zip(connections, chunked, strict=True):
            if not chunk:
                chunk = [[{"role": "user", "content": "<placeholder>"}]]
            kwargs = {
                "messages": chunk,
                "sampling_params": sampling_params,
                "use_tqdm": False,
                "lora_request": active_lora["request"],
            }
            conn.send({"type": "call", "method": "chat", "kwargs": kwargs})

        all_outputs = [conn.recv() for conn in connections]
        all_outputs = [o for o, c in zip(all_outputs, chunked, strict=True) if c]
        all_outputs = list(chain.from_iterable(all_outputs))

        if not all_outputs:
            return {"choices": [], "model": script_args.model}

        # Format as OpenAI response
        import uuid

        choices = []
        for i, output in enumerate(all_outputs):
            for j, out in enumerate(output.outputs):
                text = out.text
                # Extract token IDs if requested
                # Build logprobs in OpenAI format
                lp_list = None
                if out.logprobs:
                    lp_list = {
                        "content": [
                            {"token": "", "logprob": next(iter(lp.values())).logprob}  # nosec B105
                            for lp in out.logprobs
                        ]
                    }

                choice = {
                    "index": i * n + j,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                    if out.finish_reason == "stop"
                    else "length",
                    "logprobs": lp_list,
                }
                # Include token ID information for NeMo Gym
                choice["prompt_token_ids"] = output.prompt_token_ids
                choice["generation_token_ids"] = list(out.token_ids)
                if out.logprobs:
                    choice["generation_log_probs"] = [
                        next(iter(lp.values())).logprob for lp in out.logprobs
                    ]
                choices.append(choice)

        prompt_tokens = len(all_outputs[0].prompt_token_ids) if all_outputs else 0
        completion_tokens = sum(
            len(out.token_ids) for o in all_outputs for out in o.outputs
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": script_args.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # --- Weight sync endpoints (legacy fallback, same as TRL) ---

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        world_size = (
            script_args.tensor_parallel_size * script_args.data_parallel_size + 1
        )
        kwargs = {
            "method": "init_communicator",
            "args": (
                request.host,
                request.port,
                world_size,
                request.client_device_uuid,
            ),
        }
        msg = {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(loop.run_in_executor(None, c.send, msg) for c in connections)
        )
        return {"message": "Initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        kwargs = {
            "method": "update_named_param",
            "args": (request.name, request.dtype, tuple(request.shape)),
        }
        msg = {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(loop.run_in_executor(None, c.send, msg) for c in connections)
        )
        return {"message": "Updating parameter"}

    class BatchUpdateWeightsRequest(BaseModel):
        params: list[dict]

    @app.post("/batch_update_named_params/")
    async def batch_update_named_params(request: BatchUpdateWeightsRequest):
        params_list = [
            (p["name"], p["dtype"], tuple(p["shape"])) for p in request.params
        ]
        kwargs = {"method": "batch_update_named_params", "args": (params_list,)}
        msg = {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(loop.run_in_executor(None, c.send, msg) for c in connections)
        )
        return {"message": f"Batch update for {len(params_list)} params"}

    class HTTPWeightUpdateRequest(BaseModel):
        """Weight update via HTTP (no NCCL needed)."""

        params: list[
            dict
        ]  # [{"name": str, "dtype": str, "shape": list, "data": str (base64)}]

    @app.post("/http_update_weights/")
    async def http_update_weights(request: HTTPWeightUpdateRequest):
        """Update model weights via HTTP — no NCCL communicator required.

        Tensor data is sent as base64-encoded raw bytes in the request body.
        Slower than NCCL for large models but works without cross-process setup.
        """
        from axolotl.utils.weight_serde import (
            decode_from_http,
            encode_for_ipc,
        )

        weights_to_load = [decode_from_http(p) for p in request.params]

        # Send all weights in a single IPC call.  Tensors don't survive
        # vLLM's multiproc IPC, so serialize as raw bytes + metadata.
        param_entries = [
            encode_for_ipc(name, weight) for name, weight in weights_to_load
        ]
        kwargs = {
            "method": "http_load_weights_batch",
            "kwargs": {"params": param_entries},
        }
        msg = {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(loop.run_in_executor(None, c.send, msg) for c in connections)
        )
        return {"message": f"HTTP weight update for {len(weights_to_load)} params"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        # Fire-and-forget: send reset without expecting a reply.
        # Using "fire_and_forget" type so workers don't send back a response
        # that would sit in the pipe and corrupt the next recv() for
        # generate/chat calls.
        for conn in connections:
            conn.send({"type": "fire_and_forget", "method": "reset_prefix_cache"})
        return {"message": "Reset prefix cache received"}

    @app.post("/close_communicator/")
    async def close_communicator():
        kwargs = {"method": "close_communicator"}
        for conn in connections:
            conn.send(
                {
                    "type": "fire_and_forget",
                    "method": "collective_rpc",
                    "kwargs": kwargs,
                }
            )
        return {"message": "Closing communicator"}

    uvicorn.run(
        app,
        host=script_args.host,
        port=script_args.port,
        log_level=script_args.log_level,
        access_log=True,
    )
