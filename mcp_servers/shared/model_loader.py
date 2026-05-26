"""Shared model loading helpers for MCP servers."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_adapter_folder(adapter_dir: str) -> str:
    """Prepare adapter artifacts in a temporary folder for reliable loading."""
    tmpdir = tempfile.mkdtemp(prefix="fixed_lora_")
    os.makedirs(tmpdir, exist_ok=True)

    cfg_src = os.path.join(adapter_dir, "adapter_config.json")
    cfg_dst = os.path.join(tmpdir, "adapter_config.json")
    shutil.copy(cfg_src, cfg_dst)
    peft_cfg = PeftConfig.from_pretrained(tmpdir)
    peft_cfg.inference_mode = True
    peft_cfg.save_pretrained(tmpdir)

    st_src = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = load_file(st_src, device="cpu")
    prefix = "base_model.model."
    needs_prefix = not next(iter(sd)).startswith(prefix)
    fixed = {}
    if needs_prefix:
        for key, value in sd.items():
            fixed[prefix + key] = value
    else:
        fixed = sd
    save_file(fixed, os.path.join(tmpdir, "adapter_model.safetensors"))
    return tmpdir


class ModelLoader:
    """Cache base model + adapter for low-latency MCP tool calls."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = torch.device("cpu")
        self._model_name = None
        self._adapter_dir = None
        self._prepared_adapter_dir = None

    def load(self, model_name: str, adapter_dir: str | None) -> dict[str, str | bool]:
        """Load and cache tokenizer/model. No-op when already loaded for same config."""
        if self._model is not None and self._model_name == model_name and self._adapter_dir == adapter_dir:
            return {
                "loaded": True,
                "cached": True,
                "model_name": model_name,
                "adapter_dir": adapter_dir or "",
            }

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(model_name)
        model = base

        if adapter_dir:
            if not Path(adapter_dir).exists():
                raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
            self._prepared_adapter_dir = prepare_adapter_folder(adapter_dir)
            model = PeftModel.from_pretrained(base, self._prepared_adapter_dir, local_files_only=True)

        self._model = model.to(self._device).eval()
        self._model_name = model_name
        self._adapter_dir = adapter_dir

        return {
            "loaded": True,
            "cached": False,
            "model_name": model_name,
            "adapter_dir": adapter_dir or "",
        }

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Generate a completion for a prompt using loaded model artifacts."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=False,
        )
        gen_ids = outputs[0, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    @property
    def status(self) -> dict[str, str | bool]:
        return {
            "loaded": self._model is not None,
            "model_name": self._model_name or "",
            "adapter_dir": self._adapter_dir or "",
        }


_MODEL_LOADER: ModelLoader | None = None


def get_model_loader() -> ModelLoader:
    """Return process-wide singleton loader."""
    global _MODEL_LOADER
    if _MODEL_LOADER is None:
        _MODEL_LOADER = ModelLoader()
    return _MODEL_LOADER
