import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import bitsandbytes as bnb
import torch
import transformers
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    AutoConfig,
    BitsAndBytesConfig,
)

try:
    from transformers import (
        LlamaForCausalLM,
        LlamaTokenizer,
    )
except:
    logging.warning(
        "This version of transformers does not support Llama. Consider upgrading."
    )

from axolotl.prompt_tokenizers import LLAMA_DEFAULT_PAD_TOKEN

if TYPE_CHECKING:
    from peft import PeftModel, PeftConfig
    from attrdict import AttrDefault
    from transformers import PreTrainedTokenizer


def load_model(
    base_model,
    base_model_config,
    model_type,
    tokenizer_type,
    cfg,
    adapter="lora",
    inference=False,
):
    # type: (str, str, str, str, AttrDefault, Optional[str], bool) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Optional[PeftConfig]]

    # TODO refactor as a kwarg
    load_in_8bit = cfg.load_in_8bit
    tokenizer = None
    is_llama_derived_model = "llama" in base_model or (
        cfg.model_type and "llama" in cfg.model_type.lower()
    )

    if is_llama_derived_model and cfg.flash_attention:
        if cfg.device not in ["mps", "cpu"] and inference is False:
            from axolotl.flash_attn import replace_llama_attn_with_flash_attn

            logging.info("patching with flash attention")
            replace_llama_attn_with_flash_attn()
    elif is_llama_derived_model and cfg.xformers_attention:
        from alpaca_lora_4bit.monkeypatch.llama_attn_hijack_xformers import (
            hijack_llama_attention,
        )

        logging.info("patching with xformers attention")
        hijack_llama_attention()

    if cfg.bf16:
        torch_dtype = torch.bfloat16
    elif cfg.load_in_8bit or cfg.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    try:
        if cfg.load_4bit:
            from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import (
                replace_peft_model_with_int4_lora_model,
            )

            replace_peft_model_with_int4_lora_model()
        from peft import prepare_model_for_int8_training
    except Exception as e:
        logging.exception(e)
        raise e

    model_kwargs = {}
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    try:
        if cfg.load_4bit and is_llama_derived_model:
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
                    logging.warning(
                        "unable to find a cached model file, this will likely fail..."
                    )
                    model_path = str(cache_model_path)
            except:
                model_path = cfg.base_model
            model, tokenizer = load_llama_model_4bit_low_ram(
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
        elif is_llama_derived_model and "LlamaForCausalLM" in globals():
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=torch_dtype,
                device_map=cfg.device_map,
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
        elif model_type:
            model = getattr(transformers, model_type).from_pretrained(
                base_model,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=torch_dtype,
                device_map=cfg.device_map,
                trust_remote_code=True if cfg.trust_remote_code is True else False,
                **model_kwargs,
            )
        else:
            config = AutoConfig.from_pretrained(
                base_model,
                trust_remote_code=True if cfg.trust_remote_code is True else False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                config=config,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=torch_dtype,
                device_map=cfg.device_map,
                trust_remote_code=True if cfg.trust_remote_code is True else False,
                **model_kwargs,
            )
    except Exception as e:
        logging.error(
            "Exception raised attempting to load model, retrying with AutoModelForCausalLM"
        )
        logging.exception(e)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
            trust_remote_code=True if cfg.trust_remote_code is True else False,
            **model_kwargs,
        )

    if not tokenizer:
        try:
            if is_llama_derived_model and "LlamaTokenizer" in globals():
                tokenizer = LlamaTokenizer.from_pretrained(
                    base_model_config,
                    trust_remote_code=True if cfg.trust_remote_code is True else False,
                )
            else:
                tokenizer = getattr(transformers, tokenizer_type).from_pretrained(
                    base_model_config,
                    trust_remote_code=True if cfg.trust_remote_code is True else False,
                )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_config,
                trust_remote_code=True if cfg.trust_remote_code is True else False,
            )

    logging.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
    logging.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
    logging.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
    logging.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    if tokenizer.__class__.__name__ in ["LlamaTokenizer", "LlamaTokenizerFast"]:
        tokenizer.pad_token = LLAMA_DEFAULT_PAD_TOKEN

    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg.special_tokens:
        for k, v in cfg.special_tokens.items():
            tokenizer.add_special_tokens({k: v})
    if cfg.tokens:
        tokenizer.add_tokens(list(cfg.tokens))

    embeddings_len = math.ceil(len(tokenizer) / 32) * 32
    model.resize_token_embeddings(embeddings_len)

    if (
        ((cfg.adapter == "lora" and load_in_8bit) or cfg.adapter == "qlora")
        and not cfg.load_4bit
        and (load_in_8bit or cfg.load_in_4bit)
    ):
        logging.info("converting PEFT model w/ prepare_model_for_int8_training")
        model = prepare_model_for_int8_training(model)

    model, lora_config = load_adapter(model, cfg, adapter)

    if cfg.ddp and not load_in_8bit:
        model.to(f"cuda:{cfg.local_rank}")

    if cfg.load_4bit:
        # Scales to half
        logging.info("Fitting 4bit scales and zeros to half")
        for n, m in model.named_modules():
            if "Autograd4bitQuantLinear" in str(type(m)) or "Linear4bitLt" in str(
                type(m)
            ):
                if hasattr(m, "is_v1_model") and m.is_v1_model:
                    m.zeros = m.zeros.half()
                m.scales = m.scales.half()
                m.bias = m.bias.half()

    if (
        torch.cuda.device_count() > 1
        and int(os.getenv("WORLD_SIZE", "1")) > 1
        and cfg.load_4bit
    ):
        # llama is PROBABLY model parallelizable, but the default isn't that it is
        # so let's only set it for the 4bit, see
        # https://github.com/johnsmith0031/alpaca_lora_4bit/blob/08b3fca4a4a9e0d3945be1bab4529f100a428636/finetune.py#L130-L133
        model.is_parallelizable = True
        model.model_parallel = True

    requires_grad = []
    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            requires_grad.append(f"{name}: {param.requires_grad}")
    if len(requires_grad) == 0:
        logging.warning("there are no parameters that require gradient updates")
    model.config.use_cache = False

    # TODO resume_from_checkpoint handling
    return model, tokenizer, lora_config


def load_adapter(model, cfg, adapter):
    # type: (PreTrainedModel, AttrDefault, Optional[str]) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    if adapter is None:
        return model, None
    if adapter in ["lora", "qlora"]:
        return load_lora(model, cfg)
    if adapter == "llama-adapter":
        return load_llama_adapter(model, cfg)

    raise NotImplementedError(f"{adapter} peft adapter not available")


def load_llama_adapter(model, cfg):
    # type: (PreTrainedModel, AttrDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
    from peft import (
        AdaptionPromptConfig,
        get_peft_model,
        PeftModel,
    )

    peft_config = AdaptionPromptConfig(
        adapter_layers=cfg.peft_adapter.layers,  # layers (L)
        adapter_len=cfg.peft_adapter.len,  # prompt length (K)
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        logging.info("Loading pretained LORA")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            device_map=cfg.device_map,
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
    # type: (PreTrainedModel, AttrDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
    )

    lora_target_modules = list(cfg.lora_target_modules)

    if cfg.lora_target_linear:
        bits = None
        if cfg.load_in_4bit:
            bits = 4
        elif cfg.load_in_8bit:
            bits = 8

        linear_names = find_all_linear_names(bits, model)
        logging.info(f"found linear modules: {repr(linear_names)}")
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
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            device_map=cfg.device_map,
            # torch_dtype=torch.float16,
        )
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, lora_config
