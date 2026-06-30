"""Pydantic models for PEFT-related configuration"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from axolotl.utils.schemas.model import LEGACY_MQC_STRING_TO_BACKEND


class LoftQConfig(BaseModel):
    """LoftQ configuration subset"""

    loftq_bits: int = Field(
        default=4, json_schema_extra={"description": "typically 4 bits"}
    )


class PeftConfig(BaseModel):
    """PEFT configuration subset"""

    loftq_config: LoftQConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Configuration options for loftq initialization for LoRA"
        },
    )


class LoraConfig(BaseModel):
    """Peft / LoRA configuration subset"""

    load_in_8bit: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "DEPRECATED: prefer `model_quantization_config: {bnb: "
                "{weight_dtype: int8}}`. Legacy bnb 8-bit shorthand; kept "
                "for backward compatibility, translated to the structured "
                "form at validation time with a deprecation warning."
            )
        },
    )
    load_in_4bit: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": (
                "DEPRECATED: prefer `model_quantization_config: {bnb: "
                "{weight_dtype: nf4}}`. Legacy bnb 4-bit shorthand; kept "
                "for backward compatibility, translated to the structured "
                "form at validation time with a deprecation warning."
            )
        },
    )

    adapter: Literal["lora", "qlora", "llama-adapter"] | str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Built-in adapters: `lora`, `llama-adapter`, or `qlora` "
                "(deprecated alias for lora + a bnb nf4 base quant). Plugins "
                "may register additional adapter names. Leave blank to train "
                "all parameters of the original model."
            )
        },
    )
    lora_model_dir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "If you already have a lora model trained that you want to load, put that here. This means after training, if you want to test the model, you should set this to the value of `output_dir`. Note that if you merge an adapter to the base model, a new subdirectory `merged` will be created under the `output_dir`."
        },
    )
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_fan_in_fan_out: bool | None = None
    lora_target_modules: str | list[str] | None = None
    lora_target_parameters: str | list[str] | None = None
    lora_target_linear: bool | None = Field(
        default=None,
        json_schema_extra={"description": "If true, will target all linear modules"},
    )
    lora_modules_to_save: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "If you added new tokens to the tokenizer, you may need to save some LoRA modules because they need to know the new tokens. For LLaMA and Mistral, you need to save `embed_tokens` and `lm_head`. It may vary for other models. `embed_tokens` converts tokens to embeddings, and `lm_head` converts embeddings to token probabilities."
        },
    )
    lora_dropout: float | None = 0.0
    peft_layers_to_transform: list[int] | None = Field(
        default=None,
        json_schema_extra={
            "description": "The layer indices to transform, otherwise, apply to all layers"
        },
    )
    peft_layers_pattern: list[str] | None = None
    peft: PeftConfig | None = None
    peft_use_dora: bool | None = Field(
        default=None, json_schema_extra={"description": "Whether to use DoRA."}
    )
    peft_use_rslora: bool | None = Field(
        default=None, json_schema_extra={"description": "Whether to use RSLoRA."}
    )
    peft_layer_replication: list[tuple[int, int]] | None = Field(
        default=None,
        json_schema_extra={"description": "List of layer indices to replicate."},
    )
    peft_init_lora_weights: bool | str | None = Field(
        default=None,
        json_schema_extra={
            "description": "How to initialize LoRA weights. Default to True which is MS original implementation."
        },
    )
    peft_trainable_token_indices: list[int] | dict[str, list[int]] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "A list of token indices to fine-tune on the `embed_tokens` layer.\n"
                "Otherwise, a dict mapping an embedding layer name to its trainable token indices.\n"
                "See https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/lora#efficiently-train-tokens-alongside-lora"
            )
        },
    )
    peft_ensure_weight_tying: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Whether to tie adapter weights for tied model weights. "
                "See https://github.com/huggingface/peft/issues/2864"
            )
        },
    )
    peft_autocast_adapter_dtype: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to upcast the LoRA adapter to fp32. This is enabled by default in PEFT."
        },
    )

    qlora_sharded_model_loading: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "load qlora model in sharded format for FSDP using answer.ai technique."
        },
    )
    lora_on_cpu: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Do the LoRA/PEFT loading on CPU -- this is required if the base model is so large it takes up most or all of the available GPU VRAM, e.g. during a model and LoRA merge"
        },
    )
    gptq: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether you are training a 4-bit GPTQ quantized model"
        },
    )
    bnb_config_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "optional overrides to the bnb 4bit quantization configuration"
        },
    )

    loraplus_lr_ratio: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "loraplus learning rate ratio lr_B / lr_A. Recommended value is 2^4."
        },
    )
    loraplus_lr_embedding: float | None = Field(
        default=1e-6,
        json_schema_extra={
            "description": "loraplus learning rate for lora embedding layers. Default value is 1e-6."
        },
    )

    merge_lora: bool | None = None
    merge_method: Literal["legacy", "memory_efficient"] | None = Field(
        default="memory_efficient",
        json_schema_extra={
            "description": "Method to use for LoRA merging. 'memory_efficient' (default) processes shards individually to reduce memory usage, 'legacy' loads the full model into memory."
        },
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_base_quant_inputs(cls, data):
        """Funnel every legacy base-quant spelling into the canonical
        ``model_quantization_config`` structured form, emit deprecation
        warnings, and then mirror back into ``load_in_4bit`` /
        ``load_in_8bit`` for the downstream loader.

        ``model_quantization_config`` is the source of truth; the legacy
        flags are kept in lockstep for loader code that still keys off them.
        An explicit legacy flag that contradicts the structured form is an
        error, never a silent winner.

        Translations (legacy → canonical, all with deprecation warnings):
        - ``adapter: qlora``               → ``adapter: lora`` + bnb nf4
        - ``load_in_4bit: true`` (alone)   → bnb nf4
        - ``load_in_8bit: true`` (alone)   → bnb int8
        - ``Mxfp4Config`` / ``FineGrainedFP8Config`` strings → structured form
        """
        from axolotl.utils.logging import get_logger

        log = get_logger(__name__)

        mqc = data.get("model_quantization_config")
        if isinstance(mqc, BaseModel):
            mqc = mqc.model_dump(exclude_none=True)
            data["model_quantization_config"] = mqc
        elif isinstance(mqc, str) and mqc in LEGACY_MQC_STRING_TO_BACKEND:
            converted: dict[str, Any] = {"backend": LEGACY_MQC_STRING_TO_BACKEND[mqc]}
            if data.get("model_quantization_config_kwargs"):
                converted["config_kwargs"] = data["model_quantization_config_kwargs"]
                data["model_quantization_config_kwargs"] = None
            log.warning(
                "DEPRECATED: `model_quantization_config: %s` (string form) is "
                "being translated to the structured form `{backend: %s}`. The "
                "string form will be removed in a future release.",
                mqc,
                converted["backend"],
            )
            mqc = converted
            data["model_quantization_config"] = mqc

        backend = mqc.get("backend") if isinstance(mqc, dict) else None
        weight_dtype = mqc.get("weight_dtype") if isinstance(mqc, dict) else None
        had_qlora_adapter = data.get("adapter") == "qlora"
        had_load_in_4bit = bool(data.get("load_in_4bit"))
        had_load_in_8bit = bool(data.get("load_in_8bit"))
        explicit_load_in_4bit_false = (
            "load_in_4bit" in data and data.get("load_in_4bit") is False
        )
        explicit_load_in_8bit_false = (
            "load_in_8bit" in data and data.get("load_in_8bit") is False
        )
        merge_mode = bool(data.get("merge_lora"))

        if had_qlora_adapter:
            if data.get("gptq"):
                raise ValueError(
                    "adapter: qlora with gptq is not supported. QLoRA is a "
                    "bitsandbytes 4-bit base quant; gptq is a separate "
                    "quantization backend. Pick one."
                )
            if backend and not (backend == "bnb" and weight_dtype == "nf4"):
                raise ValueError(
                    "adapter: qlora implies a bnb nf4 base quant, which "
                    "contradicts model_quantization_config backend "
                    f"'{backend}' (weight_dtype: {weight_dtype}). Use "
                    "adapter: lora with the structured form."
                )
            if had_load_in_8bit and not had_load_in_4bit:
                raise ValueError(
                    "adapter: qlora with load_in_8bit is ambiguous (QLoRA is "
                    "a 4-bit base quant). Use adapter: lora with "
                    "model_quantization_config: {backend: bnb, weight_dtype: "
                    "int8} for 8-bit LoRA."
                )
            if explicit_load_in_4bit_false and not merge_mode:
                raise ValueError(
                    "adapter: qlora with load_in_4bit: false is "
                    "contradictory. QLoRA requires a 4-bit base; either drop "
                    "load_in_4bit or switch to adapter: lora."
                )
            data["adapter"] = "lora"
            log.warning(
                "DEPRECATED: `adapter: qlora` is being normalized to "
                "`adapter: lora`. QLoRA is just LoRA with a 4-bit base "
                "quant — express it via `model_quantization_config: "
                "{backend: bnb, weight_dtype: nf4}`. The qlora alias will "
                "be removed in a future release."
            )

        if merge_mode:
            # Only the merge CLI's explicit load_in_*bit: false makes skipping
            # the synthesis/mirror safe; a leftover merge_lora: true must
            # hard-error, never silently drop the quant.
            wants_bnb_quant = (
                had_qlora_adapter
                or had_load_in_4bit
                or had_load_in_8bit
                or backend == "bnb"
            )
            if wants_bnb_quant and not (
                explicit_load_in_4bit_false and explicit_load_in_8bit_false
            ):
                raise ValueError(
                    "Can't merge a LoRA adapter on top of a quantized base. "
                    "`axolotl merge-lora` disables quantization "
                    "automatically; remove `merge_lora: true` from the "
                    "config file."
                )
            return data

        # Step 1: if user only spelled out a legacy flag (no structured
        # form), synthesise the structured form so it becomes the single
        # source of truth.
        if mqc is None:
            if had_load_in_4bit or had_qlora_adapter:
                data["model_quantization_config"] = {
                    "backend": "bnb",
                    "weight_dtype": "nf4",
                }
                if had_load_in_4bit:
                    log.warning(
                        "DEPRECATED: `load_in_4bit: true` is being translated "
                        "to `model_quantization_config: "
                        "{backend: bnb, weight_dtype: nf4}`. Drop "
                        "`load_in_4bit` from your config; it will be removed "
                        "as a user-facing knob in a future release."
                    )
            elif had_load_in_8bit:
                data["model_quantization_config"] = {
                    "backend": "bnb",
                    "weight_dtype": "int8",
                }
                log.warning(
                    "DEPRECATED: `load_in_8bit: true` is being translated "
                    "to `model_quantization_config: "
                    "{backend: bnb, weight_dtype: int8}`. Drop `load_in_8bit` "
                    "from your config; it will be removed as a user-facing "
                    "knob in a future release."
                )
        # Step 2: mirror the structured bnb form back into load_in_4bit /
        # load_in_8bit for downstream loader compat; reject explicit
        # conflicting legacy flags instead of letting them win.
        mqc = data.get("model_quantization_config")
        if isinstance(mqc, dict):
            backend = mqc.get("backend")
            weight_dtype = mqc.get("weight_dtype")
            if backend == "bnb":
                if weight_dtype == "nf4":
                    if explicit_load_in_4bit_false or had_load_in_8bit:
                        raise ValueError(
                            "model_quantization_config {backend: bnb, "
                            "weight_dtype: nf4} conflicts with the legacy "
                            "load_in_4bit: false / load_in_8bit: true flags. "
                            "Drop the deprecated load_in_* flags."
                        )
                    data["load_in_4bit"] = True
                    data["load_in_8bit"] = False
                elif weight_dtype == "int8":
                    if explicit_load_in_8bit_false or had_load_in_4bit:
                        raise ValueError(
                            "model_quantization_config {backend: bnb, "
                            "weight_dtype: int8} conflicts with the legacy "
                            "load_in_8bit: false / load_in_4bit: true flags. "
                            "Drop the deprecated load_in_* flags."
                        )
                    data["load_in_8bit"] = True
                    data["load_in_4bit"] = False
            elif backend and (had_load_in_4bit or had_load_in_8bit):
                raise ValueError(
                    "load_in_4bit/load_in_8bit are bitsandbytes flags and "
                    "cannot be combined with model_quantization_config "
                    f"backend '{backend}'."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_adapter(cls, data):
        if (
            not data.get("adapter")
            and not data.get("inference")
            and (data.get("load_in_8bit") or data.get("load_in_4bit"))
        ):
            raise ValueError(
                "load_in_8bit and load_in_4bit are not supported without setting an adapter for training."
                "If you want to full finetune, please turn off load_in_8bit and load_in_4bit."
            )
        adapter = data.get("adapter")
        # qlora stays in the allowed set as a deprecated alias
        # (normalize_base_quant_inputs demotes it to lora).
        if adapter and adapter not in ("lora", "qlora", "llama-adapter"):
            from axolotl.integrations.base import PluginManager

            plugin_manager = PluginManager.get_instance()
            if not plugin_manager.supports_adapter(adapter):
                raise ValueError(
                    f"Adapter '{adapter}' is not built in and was not registered by "
                    "a plugin. Add the plugin that provides this adapter to `plugins:`."
                )
        return data

    @model_validator(mode="after")
    def validate_qlora(self):
        mqc = getattr(self, "model_quantization_config", None)
        backend = getattr(mqc, "backend", None)
        is_torchao = backend == "torchao"

        # torchao int4 tensors implement linear as a bare quantized matmul
        # with no autograd support (unlike nf4/nvfp4/fp8, which dequantize or
        # ship a backward), so input gradients can only flow through axolotl's
        # LoRA kernels, which dequantize explicitly in their own fwd/bwd.
        if (
            is_torchao
            and mqc.weight_dtype == "int4"
            and not getattr(self, "inference", None)
            and not self.merge_lora
            and not (
                getattr(self, "lora_mlp_kernel", None)
                and getattr(self, "lora_qkv_kernel", None)
                and getattr(self, "lora_o_kernel", None)
            )
        ):
            raise ValueError(
                "model_quantization_config {backend: torchao, weight_dtype: "
                "int4} requires the LoRA kernels for training (torchao int4 "
                "has no autograd support; backward would fail). Set "
                "lora_mlp_kernel: true, lora_qkv_kernel: true, and "
                "lora_o_kernel: true — or use weight_dtype: nf4."
            )

        # bnb/torchao base quants exist to serve a LoRA adapter; without one
        # they would silently train full-precision (the loader gates on
        # adapter), so reject instead.
        if (
            backend in ("bnb", "torchao")
            and not self.adapter
            and not getattr(self, "inference", None)
            and not self.merge_lora
        ):
            raise ValueError(
                f"model_quantization_config backend '{backend}' is a "
                "LoRA base-weight quant and requires `adapter: lora`. "
                "For full fine-tuning, remove model_quantization_config "
                "(or use the qat/ptq config blocks)."
            )

        # PEFT's dora_init dequantizes via module.weight.dequantize(), but
        # torchao's NF4Tensor.dequantize is a 2-arg staticmethod — it raises
        # TypeError at adapter creation. The other torchao dtypes work.
        if is_torchao and mqc.weight_dtype == "nf4" and self.peft_use_dora:
            raise ValueError(
                "peft_use_dora is not supported with model_quantization_config "
                "{backend: torchao, weight_dtype: nf4}. Use a different "
                "torchao weight_dtype or the bnb backend."
            )

        # torchao + merge: the memory-efficient merger simulates bnb NF4
        # quantization. Force the legacy path until the efficient one learns
        # torchao tensor subclasses.
        if is_torchao and self.merge_lora and self.merge_method != "legacy":
            raise ValueError(
                "Merging a torchao-quantized LoRA adapter requires "
                "merge_method: legacy. The memory-efficient merger only "
                "supports bnb NF4 quantization today."
            )

        if self.merge_lora and self.adapter == "lora":
            # PEFT's merge_and_unload can't merge into a quantized base
            # (bnb Params4bit / Linear8bitLt / GPTQ tensors don't accept the
            # in-place A@B add). Reject the combo regardless of how the
            # quant was spelled.
            if self.load_in_8bit:
                raise ValueError("Can't merge a LoRA adapter on top of an 8bit base.")
            if self.load_in_4bit:
                raise ValueError("Can't merge a LoRA adapter on top of a 4bit base.")
            if self.gptq:
                raise ValueError("Can't merge a LoRA adapter on top of a gptq base.")
        return self

    @field_validator("loraplus_lr_embedding")
    @classmethod
    def convert_loraplus_lr_embedding(cls, loraplus_lr_embedding):
        if loraplus_lr_embedding and isinstance(loraplus_lr_embedding, str):
            loraplus_lr_embedding = float(loraplus_lr_embedding)
        return loraplus_lr_embedding

    @model_validator(mode="before")
    @classmethod
    def validate_lora_dropout(cls, data):
        if data.get("adapter") is not None and data.get("lora_dropout") is None:
            data["lora_dropout"] = 0.0
        return data

    @model_validator(mode="after")
    def validate_lora_target_parameters_dropout(self):
        if (
            self.lora_target_parameters
            and self.lora_dropout
            and self.lora_dropout != 0.0
        ):
            raise ValueError(
                "lora_dropout must be 0 when lora_target_parameters is set. "
                "PEFT's ParamWrapper does not support lora_dropout != 0."
            )
        return self


class ReLoRAConfig(BaseModel):
    """ReLoRA configuration subset"""

    relora: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use ReLoRA. Use with jagged_restart_*steps options."
        },
    )
    relora_prune_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "description": (
                "Fraction of optimizer state values to zero on each ReLoRA restart. "
                "When relora_prune_method='reset' and this is omitted, defaults to "
                "0.999 (paper-style near-full reset). For other methods, defaults to 0.9."
            )
        },
    )
    relora_prune_method: Literal["magnitude", "random", "reset"] | None = Field(
        default="magnitude",
        json_schema_extra={
            "description": (
                "Optimizer state pruning method on each ReLoRA restart. "
                "'magnitude' (default) keeps top-k by absolute value; "
                "'random' keeps a random subset at relora_prune_ratio; "
                "'reset' uses near-full random pruning (default ratio 0.999, "
                "honoring relora_prune_ratio when explicitly set). "
                "Paper-style recipe: relora_prune_method='reset' with no "
                "relora_prune_ratio, equivalent to 'random' with ratio=0.999."
            )
        },
    )
    relora_cpu_offload: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "True to perform lora weight merges on cpu during restarts, for modest gpu memory savings"
        },
    )
