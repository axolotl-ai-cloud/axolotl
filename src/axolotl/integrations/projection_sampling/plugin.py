"""Dataset-loading plugin for offline projection sampling and ordinary SFT."""

import hashlib
import importlib
import json
import os
import tempfile
from glob import glob
from pathlib import Path

from filelock import FileLock

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .args import ProjectionSamplingConfig, get_seed
from .backend import load_backend
from .inspection import export_dataset
from .sampler import ProjectionSampler

LOG = get_logger(__name__)
CACHE_VERSION = 1


def cache_path(cfg, config: ProjectionSamplingConfig) -> Path:
    """Fingerprint settings, tokenization, source configuration, and local data."""
    settings = config.model_dump(exclude={"cache_dir", "device"})
    settings["seed"] = get_seed(cfg)
    if config.proposal_batch_size == 1:
        settings.pop("proposal_batch_size")
    if config.backend == "transformers" and not config.backend_kwargs:
        settings.pop("backend")
        settings.pop("backend_kwargs")
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
    if cfg.get("eot_tokens"):
        payload["sampling_eot_tokens"] = cfg.eot_tokens
    if any(source.get("type") == "chat_template" for source in cfg.datasets):
        payload["chat_tokenization"] = {
            key: cfg.get(key)
            for key in (
                "train_on_inputs",
                "sequence_len",
                "eot_tokens",
                "chat_template_kwargs",
            )
        }
        payload["chat_tokenization"]["proposal_context_format"] = "messages_json"
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
                "projection_sampling requires expert chat messages or question/response pairs"
            )
        for source in cfg.datasets:
            if not source.get("path"):
                raise ValueError("projection_sampling datasets require a source path")
            if source.get("type") not in (None, "chat_template"):
                raise ValueError(
                    "projection_sampling supports type: chat_template or flat pairs without a dataset type"
                )
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
        if preprocess or int(os.environ.get("RANK", "0")) == 0:
            exported = export_dataset(path, cfg.output_dir, seed=get_seed(cfg))
            LOG.info("Exported rewritten dataset for inspection: %s", exported)
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
                with load_backend(cfg, config) as backend:
                    sampler = ProjectionSampler(backend, config, seed=get_seed(cfg))
                    for source in cfg.datasets:
                        source = DictDefault(source)
                        dataset = load_dataset_with_config(
                            source, cfg.hf_use_auth_token, streaming=False
                        )
                        if isinstance(dataset, DatasetDict):
                            dataset = dataset[source.split or "train"]
                        if source.shards:
                            dataset = dataset.shuffle(seed=get_seed(cfg)).shard(
                                source.shards, source.shards_idx or 0
                            )
                        if source.weight is not None and source.weight < 1:
                            dataset = dataset.shuffle(seed=get_seed(cfg)).select(
                                range(int(len(dataset) * source.weight))
                            )
                        chat_strategy = None
                        if source.type == "chat_template":
                            from axolotl.prompt_strategies.chat_template import load

                            chat_strategy = load(backend.tokenizer, cfg, source)
                        for row in dataset:
                            if chat_strategy is not None:
                                from .chat import sample_chat

                                record = sample_chat(
                                    row, chat_strategy, sampler, verifier
                                )
                                output.write(
                                    json.dumps(
                                        record, ensure_ascii=False, allow_nan=False
                                    )
                                    + "\n"
                                )
                                count += 1
                                LOG.info("Projection sampled chat row %s", count)
                                continue
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
                                    **sampler.proposal_statistics(result),
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
