"""Dataset-loading plugin for offline projection sampling and ordinary SFT."""

import gc
import hashlib
import importlib
import json
import os
import tempfile
from glob import glob
from pathlib import Path

import torch
from filelock import FileLock

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .args import ProjectionSamplingConfig
from .backend import TransformersBackend
from .sampler import ProjectionSampler

LOG = get_logger(__name__)
CACHE_VERSION = 1


def cache_path(cfg, config: ProjectionSamplingConfig) -> Path:
    """Fingerprint settings, tokenization, source configuration, and local data."""
    settings = config.model_dump(exclude={"cache_dir", "device"})
    payload = {
        "version": CACHE_VERSION,
        "sampling": settings,
        "datasets": cfg.datasets,
        "model": {
            key: cfg.get(key)
            for key in (
                "base_model",
                "revision_of_model",
                "tokenizer_config",
                "tokenizer_type",
                "tokenizer_use_fast",
                "tokenizer_legacy",
                "tokenizer_use_mistral_common",
                "trust_remote_code",
                "chat_template",
                "chat_template_jinja",
                "special_tokens",
                "tokens",
                "added_tokens_overrides",
                "default_system_message",
            )
        },
    }
    local_hashes = {}
    for dataset in cfg.datasets:
        files = dataset.get("data_files") or []
        if isinstance(files, str):
            files = [files]
        for pattern in [dataset["path"], *files]:
            for filename in glob(pattern, recursive=True):
                path = Path(filename)
                paths = sorted(path.rglob("*")) if path.is_dir() else [path]
                for source in paths:
                    if source.is_file():
                        with source.open("rb") as stream:
                            local_hashes[str(source.resolve())] = hashlib.file_digest(
                                stream, "sha256"
                            ).hexdigest()
    filename = cfg.get("chat_template_jinja")
    try:
        template_path = (
            Path(filename) if filename and Path(filename).is_file() else None
        )
    except OSError:
        template_path = None
    if template_path is not None:
        local_hashes[str(template_path.resolve())] = hashlib.sha256(
            template_path.read_bytes()
        ).hexdigest()
    payload["local_files"] = local_hashes
    payload["local_models"] = {
        key: [
            (str(file.relative_to(root)), file.stat().st_size, file.stat().st_mtime_ns)
            for file in sorted(root.rglob("*"))
            if file.is_file()
        ]
        for key in ("base_model", "tokenizer_config")
        if cfg.get(key) and (root := Path(cfg[key])).is_dir()
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return Path(config.cache_dir).resolve() / f"{digest}.jsonl"


class ProjectionSamplingPlugin(BasePlugin):
    """Sample once during preprocessing and reuse immutable traces during training."""

    def get_input_args(self):
        return "axolotl.integrations.projection_sampling.args.ProjectionSamplingArgs"

    def load_datasets(self, cfg, preprocess=False):
        if cfg.get("projection_sampling") is None:
            return None
        from axolotl.common.datasets import load_datasets

        config = ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
        for key in (
            "rl",
            "streaming",
            "pretraining_dataset",
            "skip_prepare_dataset",
            "processor_type",
        ):
            if cfg.get(key):
                raise ValueError(f"projection_sampling does not support {key}")
        if not cfg.datasets:
            raise ValueError(
                "projection_sampling requires datasets of expert question/response pairs"
            )
        for source in cfg.datasets:
            if not source.get("path"):
                raise ValueError("projection_sampling datasets require a source path")
            if source.get("input_transform") or source.get("preprocess_shards"):
                raise ValueError(
                    "projection_sampling does not support input_transform or preprocess_shards"
                )
        path = cache_path(cfg, config)
        if not path.exists():
            if not preprocess:
                raise FileNotFoundError(
                    "Projection sampling cache is missing. Run `axolotl preprocess "
                    "config.yaml` in a single process before training. "
                    f"Expected cache: {path}"
                )
            if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise ValueError(
                    "Run projection sampling preprocessing in a single process"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock"):
                if not path.exists():
                    self._generate_cache(cfg, config, path)
        LOG.info("Using projection sampling dataset: %s", path)
        prepared = DictDefault(cfg.to_dict())
        prepared.datasets = [
            DictDefault(
                {
                    "path": str(path),
                    "ds_type": "json",
                    "split": "train",
                    "type": "axolotl.integrations.projection_sampling.tokenization",
                }
            )
        ]
        return load_datasets(cfg=prepared)

    def _generate_cache(self, cfg, config, path):
        from datasets import DatasetDict

        from axolotl.utils.data.shared import load_dataset_with_config

        verifier = None
        if config.verifier:
            module, name = config.verifier.rsplit(".", 1)
            verifier = getattr(importlib.import_module(module), name)
            if not callable(verifier):
                raise ValueError(
                    "projection_sampling.verifier must resolve to a callable"
                )
        backend = None
        temporary = None
        count = 0
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=path.parent,
                suffix=".jsonl",
                delete=False,
                encoding="utf-8",
            ) as output:
                temporary = Path(output.name)
                device = torch.device(config.device)
                devices = []
                if device.type == "cuda":
                    devices = [
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    ]
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(config.seed)
                    if devices:
                        with torch.cuda.device(devices[0]):
                            torch.cuda.manual_seed(config.seed)
                    backend = TransformersBackend.from_config(cfg, config)
                    sampler = ProjectionSampler(backend, config)
                    for source in cfg.datasets:
                        source = DictDefault(source)
                        dataset = load_dataset_with_config(
                            source, cfg.hf_use_auth_token, streaming=False
                        )
                        if isinstance(dataset, DatasetDict):
                            dataset = dataset[source.split or "train"]
                        if source.shards:
                            dataset = dataset.shuffle(seed=config.seed).shard(
                                source.shards, source.shards_idx or 0
                            )
                        if source.weight is not None and source.weight < 1:
                            dataset = dataset.shuffle(seed=config.seed).select(
                                range(int(len(dataset) * source.weight))
                            )
                        for row in dataset:
                            question, expert = (
                                row[config.question_field],
                                row[config.response_field],
                            )
                            if (
                                not isinstance(question, str)
                                or not isinstance(expert, str)
                                or not question.strip()
                                or not expert.strip()
                            ):
                                raise ValueError(
                                    "Projection sampling requires nonempty string questions and expert responses"
                                )
                            result = sampler.sample(question, expert)
                            tokens = result.token_ids
                            response = backend.tokenizer.decode(
                                tokens, skip_special_tokens=True
                            )
                            verified = (
                                None
                                if verifier is None
                                else bool(
                                    verifier(
                                        question=question,
                                        expert_response=expert,
                                        response=response,
                                    )
                                )
                            )
                            fallback = (
                                not result.finished
                                or not response.strip()
                                or verified is False
                            )
                            if fallback:
                                tokens = backend.tokenizer.encode(
                                    expert, add_special_tokens=False
                                )
                                eos = backend.tokenizer.eos_token_id
                                if eos is not None and (
                                    not tokens or tokens[-1] != eos
                                ):
                                    tokens.append(eos)
                                response = expert
                            record = {
                                "prompt": question,
                                "response": response,
                                "expert_response": expert,
                                "prompt_token_ids": sampler.prompt_ids(question),
                                "response_token_ids": tokens,
                                "sampling": {
                                    "attempts": result.attempts,
                                    "accepted": result.accepted,
                                    "target_logprob": result.target_logprob,
                                    "finished": result.finished,
                                    "verified": verified,
                                    "fallback_to_expert": fallback,
                                },
                            }
                            output.write(
                                json.dumps(record, ensure_ascii=False, allow_nan=False)
                                + "\n"
                            )
                            count += 1
                            LOG.info(
                                "Projection sampled row %s: accepted %s/%s, fallback=%s",
                                count,
                                result.accepted,
                                result.attempts,
                                fallback,
                            )
            if not count:
                raise ValueError("Projection sampling source dataset is empty")
            temporary.replace(path)
            LOG.info("Cached %s projection sampling traces at %s", count, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if backend is not None:
                backend.close()
                gc.collect()
