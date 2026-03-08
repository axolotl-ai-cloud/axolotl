"""Merge trained text-only Gemma3 weights back into a full multimodal checkpoint.

After training with the Gemma3TextFromMultimodalPlugin, the saved checkpoint
contains only the language model weights (with ``model.language_model.*``
prefix, reversed by transformers v5's key_mapping on save).

This script reconstructs a full ``Gemma3ForConditionalGeneration`` checkpoint by
combining the trained language model weights with the original vision tower and
projector weights from the base multimodal model.

Usage::

    python scripts/merge_gemma3_multimodal_weights.py \\
        --original-model google/gemma-3-4b-it \\
        --trained-model /path/to/trained/output \\
        --output-dir /path/to/merged
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

LOG = logging.getLogger(__name__)


def collect_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load and merge all safetensors shard files in a directory."""
    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        LOG.info("Loading %s", shard.name)
        state_dict.update(load_file(str(shard)))
    return state_dict


def merge(
    original_model: str,
    trained_model: str,
    output_dir: str,
    *,
    trust_remote_code: bool = False,
) -> None:
    original_path = Path(original_model)
    trained_path = Path(trained_model)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the original multimodal checkpoint
    LOG.info("Loading original multimodal weights from %s", original_model)
    if original_path.is_dir():
        original_sd = collect_safetensors(original_path)
    else:
        from huggingface_hub import snapshot_download

        cached = Path(
            snapshot_download(original_model, allow_patterns=["*.safetensors"])
        )
        original_sd = collect_safetensors(cached)

    # 2. Load trained text-only weights (already reversed to model.language_model.* by
    #    transformers v5 key_mapping on save)
    LOG.info("Loading trained text-only weights from %s", trained_model)
    trained_sd = collect_safetensors(trained_path)

    # 3. Classify original keys
    lang_keys = {k for k in original_sd if k.startswith("model.language_model.")}
    vision_keys = {k for k in original_sd if k.startswith("model.vision_tower.")}
    projector_keys = {
        k for k in original_sd if k.startswith("model.multi_modal_projector.")
    }
    other_keys = set(original_sd.keys()) - lang_keys - vision_keys - projector_keys

    LOG.info(
        "Original checkpoint: %d language, %d vision, %d projector, %d other keys",
        len(lang_keys),
        len(vision_keys),
        len(projector_keys),
        len(other_keys),
    )

    # 4. Classify trained keys (reverse mapping on save gives model.language_model.* prefix)
    trained_lang_keys = {k for k in trained_sd if k.startswith("model.language_model.")}
    trained_other = set(trained_sd.keys()) - trained_lang_keys

    LOG.info(
        "Trained checkpoint: %d language keys, %d other keys",
        len(trained_lang_keys),
        len(trained_other),
    )

    # 5. Build merged state dict
    merged: dict[str, torch.Tensor] = {}

    # Keep vision tower and projector from original
    for key in vision_keys | projector_keys:
        merged[key] = original_sd[key]

    # Use trained language model weights (overwrite original)
    for key in trained_lang_keys:
        merged[key] = trained_sd[key]

    # For other trained keys (like lm_head.weight), use trained version
    for key in trained_other:
        merged[key] = trained_sd[key]

    # For any original other keys not covered by trained (shouldn't usually happen),
    # keep original
    for key in other_keys:
        if key not in merged:
            merged[key] = original_sd[key]

    # Check for missing language keys that were in original but not in trained
    missing_lang = lang_keys - trained_lang_keys
    if missing_lang:
        LOG.warning(
            "%d language keys in original but not in trained; keeping original: %s",
            len(missing_lang),
            list(missing_lang)[:5],
        )
        for key in missing_lang:
            merged[key] = original_sd[key]

    LOG.info("Merged checkpoint: %d total keys", len(merged))

    # 6. Save merged weights (sharded at 50GB, matching transformers default)
    LOG.info("Saving merged weights to %s", out_path)
    state_dict_split = split_torch_state_dict_into_shards(merged, max_shard_size="50GB")

    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {name: merged[name] for name in tensors}
        save_file(shard, str(out_path / filename))

    if state_dict_split.is_sharded:
        index = {
            "metadata": {
                "total_size": sum(t.numel() * t.element_size() for t in merged.values())
            },
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(out_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        LOG.info("Saved %d shards", len(state_dict_split.filename_to_tensors))

    # 7. Copy/update config
    LOG.info("Writing config.json")
    original_config = AutoConfig.from_pretrained(
        original_model, trust_remote_code=trust_remote_code
    )

    # Update text_config fields from trained model's config if available
    trained_config_path = trained_path / "config.json"
    if trained_config_path.exists():
        with open(trained_config_path) as f:
            trained_config_dict = json.load(f)

        # The trained config is the text sub-config; merge its fields into
        # the original composite config's text_config
        if hasattr(original_config, "text_config"):
            for key, val in trained_config_dict.items():
                if key not in ("model_type", "_name_or_path", "architectures"):
                    if hasattr(original_config.text_config, key):
                        setattr(original_config.text_config, key, val)

    original_config.save_pretrained(out_path)

    # 8. Copy tokenizer files from trained model if present
    tokenizer_files = list(trained_path.glob("tokenizer*")) + list(
        trained_path.glob("special_tokens_map*")
    )
    if tokenizer_files:
        import shutil

        for tok_file in tokenizer_files:
            shutil.copy2(tok_file, out_path / tok_file.name)
        LOG.info("Copied %d tokenizer files", len(tokenizer_files))

    LOG.info("Merge complete. Output saved to %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge trained text-only Gemma3 weights back into a multimodal checkpoint."
    )
    parser.add_argument(
        "--original-model",
        required=True,
        help="HuggingFace model ID or local path to the original multimodal model",
    )
    parser.add_argument(
        "--trained-model",
        required=True,
        help="Local path to the trained text-only model output directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the merged multimodal checkpoint",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code when loading model config",
    )
    args = parser.parse_args()

    merge(
        original_model=args.original_model,
        trained_model=args.trained_model,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
