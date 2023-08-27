"""Module for models and model loading"""


import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple  # noqa: F401

import bitsandbytes as bnb
import torch
import transformers
from optimum.bettertransformer import BetterTransformer
from transformers import (  # noqa: F401
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from axolotl.prompt_tokenizers import LLAMA_DEFAULT_EOS_TOKEN
from axolotl.utils.bench import log_gpu_memory_usage

LOG = logging.getLogger("axolotl")

if TYPE_CHECKING:
    from peft import PeftConfig  # noqa: F401

    from axolotl.utils.dict import DictDefault  # noqa: F401


def load_tokenizer(cfg):
    tokenizer_kwargs = {}
    use_fast = True  # this is the default

    if cfg.tokenizer_use_fast is not None:
        use_fast = cfg.tokenizer_use_fast
    if cfg.tokenizer_legacy is not None:
        # True is the default w/ https://github.com/huggingface/transformers/pull/25224
        tokenizer_kwargs["legacy"] = cfg.tokenizer_legacy

    tokenizer_cls = AutoTokenizer
    if cfg.tokenizer_type:
        tokenizer_cls = getattr(transformers, cfg.tokenizer_type)

    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    tokenizer = tokenizer_cls.from_pretrained(
        tokenizer_config,
        trust_remote_code=cfg.trust_remote_code or False,
        use_fast=use_fast,
        **tokenizer_kwargs,
    )

    if (
        tokenizer.__class__.__name__
        in [
            "LlamaTokenizer",
            "LlamaTokenizerFast",
        ]
        and hasattr(tokenizer, "pad_token")
        and not tokenizer.pad_token
    ):
        # set a pad_token, but use eos_token so we don't add a new token
        tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

    LOG.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
    LOG.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
    LOG.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
    LOG.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg.special_tokens:
        for k, val in cfg.special_tokens.items():
            tokenizer.add_special_tokens({k: val})
    if cfg.tokens:
        tokenizer.add_tokens(list(cfg.tokens))

    return tokenizer


def load_model(
    cfg, tokenizer
):  # type: (DictDefault, PreTrainedTokenizerBase) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
    """
    Load a model for a given configuration and tokenizer.
    """
    base_model = cfg.base_model
    base_model_config = cfg.base_model_config
    model_type = cfg.model_type

    # TODO refactor as a kwarg
    load_in_8bit = cfg.load_in_8bit
    cfg.is_llama_derived_model = (
        "llama" in base_model
        or (cfg.model_type and "llama" in cfg.model_type.lower())
        or cfg.is_llama_derived_model
    )

    if cfg.is_llama_derived_model and cfg.flash_attention:
        if cfg.device not in ["mps", "cpu"] and not cfg.inference:
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                replace_llama_attn_with_flash_attn,
            )

            LOG.info("patching with flash attention")
            replace_llama_attn_with_flash_attn(packed=cfg.sample_packing)
    elif cfg.is_llama_derived_model and cfg.xformers_attention:
        from axolotl.monkeypatch.llama_attn_hijack_xformers import (
            hijack_llama_attention,
        )

        LOG.info("patching with xformers attention")
        hijack_llama_attention()
    elif cfg.is_llama_derived_model and cfg.sdp_attention:
        from axolotl.monkeypatch.llama_attn_hijack_sdp import hijack_llama_sdp_attention

        LOG.info("patching with sdp attention")
        hijack_llama_sdp_attention()
    elif cfg.is_llama_derived_model and cfg.landmark_attention:
        from axolotl.monkeypatch.llama_landmark_attn import (
            MEM_TOKEN,
            patch_llama_with_landmark_attn,
        )

        LOG.info("patching with landmark attention")
        patch_llama_with_landmark_attn()

        # Note: This might overwrite previous additional_special_tokens
        tokenizer.add_special_tokens({"additional_special_tokens": [MEM_TOKEN]})

    if cfg.is_llama_derived_model and cfg.xpos_rope:
        from axolotl.monkeypatch.xpos_rope_llama_monkey_patch import (
            replace_llama_rope_with_xpos_rope,
        )

        LOG.info("patching with xpos rope")
        replace_llama_rope_with_xpos_rope()

    if (
        cfg.is_llama_derived_model
        and (cfg.max_packed_sequence_len or cfg.sample_packing)
        and not cfg.inference
    ):
        from axolotl.monkeypatch.llama_expand_mask import hijack_expand_mask

        LOG.info("patching _expand_mask")
        hijack_expand_mask()

    try:
        if cfg.gptq:
            from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import (
                replace_peft_model_with_int4_lora_model,
            )

            replace_peft_model_with_int4_lora_model()
    except Exception as err:
        LOG.exception(err)
        raise err

    if not cfg.gptq and (
        (cfg.adapter == "lora" and load_in_8bit)
        or (cfg.adapter == "qlora" and cfg.load_in_4bit)
    ):
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError:
            # For backward compatibility
            from peft import (
                prepare_model_for_int8_training as prepare_model_for_kbit_training,
            )

    model_kwargs = {}
    if cfg.model_revision:
        model_kwargs["revision"] = cfg.model_revision
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=cfg.torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    try:
        if cfg.gptq and cfg.is_llama_derived_model:
            from alpaca_lora_4bit.autograd_4bit import load_llama_model_4bit_low_ram
            from huggingface_hub import snapshot_download

            try:
                snapshot_download_kwargs = {}
                if cfg.base_model_ignore_patterns:
                    snapshot_download_kwargs[
                        "ignore_patterns"
                    ] = cfg.base_model_ignore_patterns
                cache_model_path = Path(
                    snapshot_download(base_model, **snapshot_download_kwargs)
                )
                files = (
                    list(cache_model_path.glob("*.pt"))
                    + list(cache_model_path.glob("*.safetensors"))
                    + list(cache_model_path.glob("*.bin"))
                )
                if len(files) > 0:
                    model_path = str(files[0])
                else:
                    LOG.warning(
                        "unable to find a cached model file, this will likely fail..."
                    )
                    model_path = str(cache_model_path)
            except Exception:  # pylint: disable=broad-exception-caught
                model_path = cfg.base_model
            model, _ = load_llama_model_4bit_low_ram(
                base_model_config if base_model_config else base_model,
                model_path,
                device_map=cfg.device_map,
                half=cfg.fp16,
                groupsize=cfg.gptq_groupsize if cfg.gptq_groupsize else -1,
                is_v1_model=cfg.gptq_model_v1
                if cfg.gptq_model_v1 is not None
                else True,
            )
            load_in_8bit = False
        elif cfg.is_llama_derived_model and not cfg.trust_remote_code:
            from transformers import LlamaForCausalLM

            config_kwargs = {}
            if cfg.rope_scaling:
                config_kwargs["rope_scaling"] = cfg.rope_scaling
            config = LlamaConfig.from_pretrained(
                base_model_config,
                **config_kwargs,
            )
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                config=config,
                device_map=cfg.device_map,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=cfg.torch_dtype,
                **model_kwargs,
            )
        # elif model_type == "GPTNeoXForCausalLM" and cfg.flash_attention:
        #     This is a WIP, still an issue with the backward pass
        #     RuntimeError: grad can be implicitly created only for scalar outputs
        #     TODO: try config.sequence_parallel = False
        #     # https://github.com/HazyResearch/flash-attention/blob/40a25c8ee7465cf547b929cfa2937034e37bfce9/tests/models/test_gpt_neox.py#L12
        #     # https://github.com/HazyResearch/flash-attention/tree/main/training#model-components
        #     # add `**kwargs` to https://github.com/HazyResearch/flash-attention/blob/40a25c8ee7465cf547b929cfa2937034e37bfce9/flash_attn/models/gpt.py#L442
        #     from flash_attn.utils.pretrained import state_dict_from_pretrained
        #     from flash_attn.models.gpt import GPTLMHeadModel
        #     from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox, gpt_neox_config_to_gpt2_config
        #     from transformers import GPTNeoXConfig
        #     config = gpt_neox_config_to_gpt2_config(GPTNeoXConfig.from_pretrained(base_model))
        #     config.use_flash_attn = True
        #     config.fused_bias_fc = True
        #     config.fused_mlp = True  # GPT-NeoX-20B uses "gelu_fast"
        #     config.activation_function = "gelu_fast"
        #     config.fused_dropout_add_ln = True
        #     # config.residual_in_fp32 = True
        #
        #     model: GPTLMHeadModel = GPTLMHeadModel.from_pretrained(
        #         base_model,
        #         config,
        #         dtype=torch_dtype,
        #         device=cfg.device,
        #     )
        #     model.train() # sets to train instead of eval mode
        elif model_type and not cfg.trust_remote_code:
            model = getattr(transformers, model_type).from_pretrained(
                base_model,
                device_map=cfg.device_map,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=cfg.torch_dtype,
                trust_remote_code=cfg.trust_remote_code or False,
                **model_kwargs,
            )
        else:
            config = AutoConfig.from_pretrained(
                base_model,
                trust_remote_code=cfg.trust_remote_code or False,
            )
            # Shouldn't be a problem most of the time. will obviously error if the model doesn't support this
            # when training starts
            if (
                hasattr(config, "max_seq_len")
                and config.max_seq_len
                and cfg.sequence_len > config.max_seq_len
            ):
                config.max_seq_len = cfg.sequence_len
                LOG.warning(f"increasing context length to {cfg.sequence_len}")
            elif (
                hasattr(config, "max_sequence_length")
                and config.max_sequence_length
                and cfg.sequence_len > config.max_sequence_length
            ):
                config.max_sequence_length = cfg.sequence_len
                LOG.warning(f"increasing context length to {cfg.sequence_len}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                config=config,
                device_map=cfg.device_map,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=cfg.torch_dtype,
                trust_remote_code=cfg.trust_remote_code or False,
                **model_kwargs,
            )
    except Exception as err:  # pylint: disable=broad-exception-caught
        LOG.error(
            "Exception raised attempting to load model, retrying with AutoModelForCausalLM"
        )
        LOG.exception(err)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=cfg.device_map,
            load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
            load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
            torch_dtype=cfg.torch_dtype,
            trust_remote_code=cfg.trust_remote_code or False,
            **model_kwargs,
        )

    embeddings_len = (
        math.ceil(len(tokenizer) / 32) * 32
        if cfg.resize_token_embeddings_to_32x
        else len(tokenizer)
    )
    model.resize_token_embeddings(embeddings_len)

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings
        and cfg.sequence_len >= model.config.max_position_embeddings
    ):
        LOG.warning(
            f"increasing model.config.max_position_embeddings to {cfg.sequence_len}"
        )
        model.config.max_position_embeddings = cfg.sequence_len

    if model.device.type == "cuda":
        log_gpu_memory_usage(LOG, "after model load", model.device)

    # make sure these are fp32 per Ramesh et al. (2021)
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch.float32)

    needs_fa2_dtype = not cfg.adapter
    if not cfg.gptq and (
        (cfg.adapter == "lora" and load_in_8bit)
        or (cfg.adapter == "qlora" and cfg.load_in_4bit)
    ):
        LOG.info("converting PEFT model w/ prepare_model_for_kbit_training")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
        needs_fa2_dtype = True

    # LlamaRMSNorm layers are in fp32 after kbit_training or full finetune, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    if needs_fa2_dtype and (cfg.flash_attention and cfg.is_llama_derived_model):
        for name, module in model.named_modules():
            if "norm" in name:
                module.to(cfg.torch_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module.to(cfg.torch_dtype)

    model, lora_config = load_adapter(model, cfg, cfg.adapter)

    if cfg.ddp and not load_in_8bit:
        model.to(f"cuda:{cfg.local_rank}")

    if cfg.gptq:
        # Scales to half
        LOG.info("Fitting 4bit scales and zeros to half")
        for _, module in model.named_modules():
            if "Autograd4bitQuantLinear" in str(type(module)) or "Linear4bitLt" in str(
                type(module)
            ):
                if hasattr(module, "is_v1_model") and module.is_v1_model:
                    module.zeros = module.zeros.half()
                module.scales = module.scales.half()
                module.bias = module.bias.half()

    if (
        torch.cuda.device_count() > 1
        and int(os.getenv("WORLD_SIZE", "1")) > 1
        and (cfg.gptq or cfg.load_in_4bit)
    ):
        # llama is PROBABLY model parallelizable, but the default isn't that it is
        # so let's only set it for the 4bit, see
        # https://github.com/johnsmith0031/alpaca_lora_4bit/blob/08b3fca4a4a9e0d3945be1bab4529f100a428636/finetune.py#L130-L133
        setattr(model, "is_parallelizable", True)
        setattr(model, "model_parallel", True)

    requires_grad = []
    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            requires_grad.append(f"{name}: {param.requires_grad}")
    if len(requires_grad) == 0:
        LOG.warning("there are no parameters that require gradient updates")
    model.config.use_cache = False

    if cfg.flash_optimum:
        model = BetterTransformer.transform(model)

    if cfg.adapter is not None:
        log_gpu_memory_usage(LOG, "after adapters", model.device)

    # TODO resume_from_checkpoint handling
    return model, lora_config


def load_adapter(model, cfg, adapter):
    # type: (PreTrainedModel, DictDefault, Optional[str]) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    if adapter is None:
        return model, None
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if adapter in ["lora", "qlora"]:
        return load_lora(model, cfg)
    if adapter == "llama-adapter":
        return load_llama_adapter(model, cfg)

    raise NotImplementedError(f"{adapter} peft adapter not available")


def load_llama_adapter(model, cfg):
    # type: (PreTrainedModel, DictDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
    from peft import AdaptionPromptConfig, PeftModel, get_peft_model

    peft_config = AdaptionPromptConfig(
        adapter_layers=cfg.peft_adapter.layers,  # layers (L)
        adapter_len=cfg.peft_adapter.len,  # prompt length (K)
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretained PEFT - llama_adapter")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            torch_dtype=torch.float16,
        )
    else:
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config


def find_all_linear_names(bits, model):
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def load_lora(model, cfg):
    # type: (PreTrainedModel, DictDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    from peft import LoraConfig, PeftModel, get_peft_model

    lora_target_modules = list(cfg.lora_target_modules or [])

    if cfg.lora_target_linear:
        bits = None
        if cfg.load_in_4bit:
            bits = 4
        elif cfg.load_in_8bit:
            bits = 8

        linear_names = find_all_linear_names(bits, model)
        LOG.info(f"found linear modules: {repr(linear_names)}")
        lora_target_modules = list(set(lora_target_modules + linear_names))

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretained PEFT - LoRA")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            is_trainable=not cfg.inference,
        )
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, lora_config
