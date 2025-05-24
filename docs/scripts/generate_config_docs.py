#!/usr/bin/env python3
"""
Enhanced Quarto documentation generator from Pydantic models.
Includes YAML example generation and better formatting.
"""

from typing import Any

import yaml
from pydantic import BaseModel

from axolotl.utils.schemas.config import AxolotlInputConfig


class QuartoGenerator:
    """Generate Quarto documentation from Pydantic models."""

    def __init__(self):
        self.field_categories = {
            "Model Configuration": {
                "description": "Basic model setup and tokenizer configuration",
                "fields": [
                    "base_model",
                    "base_model_config",
                    "base_model_ignore_patterns",
                    "model_type",
                    "tokenizer_type",
                    "tokenizer_config",
                    "trust_remote_code",
                    "tokenizer_use_fast",
                    "tokenizer_legacy",
                ],
            },
            "Training Hyperparameters": {
                "description": "Core training settings and optimization parameters",
                "fields": [
                    "num_epochs",
                    "micro_batch_size",
                    "gradient_accumulation_steps",
                    "learning_rate",
                    "warmup_steps",
                    "warmup_ratio",
                    "max_steps",
                    "optimizer",
                    "lr_scheduler",
                    "weight_decay",
                ],
            },
            "Dataset Configuration": {
                "description": "Dataset loading and preprocessing options",
                "fields": [
                    "datasets",
                    "test_datasets",
                    "val_set_size",
                    "sequence_len",
                    "sample_packing",
                    "pad_to_sequence_len",
                    "train_on_inputs",
                ],
            },
            "LoRA/PEFT Settings": {
                "description": "Low-rank adaptation and parameter-efficient fine-tuning",
                "fields": [
                    "adapter",
                    "lora_r",
                    "lora_alpha",
                    "lora_dropout",
                    "lora_target_modules",
                    "lora_model_dir",
                    "lora_target_linear",
                ],
            },
            "Memory & Performance": {
                "description": "Memory optimization and performance settings",
                "fields": [
                    "load_in_8bit",
                    "load_in_4bit",
                    "bf16",
                    "fp16",
                    "tf32",
                    "flash_attention",
                    "gradient_checkpointing",
                    "low_cpu_mem_usage",
                ],
            },
            "Distributed Training": {
                "description": "Multi-GPU and distributed training configuration",
                "fields": [
                    "deepspeed",
                    "fsdp",
                    "fsdp_config",
                    "ddp_timeout",
                    "ddp_bucket_cap_mb",
                    "world_size",
                    "local_rank",
                ],
            },
            "Logging & Monitoring": {
                "description": "Experiment tracking and monitoring options",
                "fields": [
                    "wandb_project",
                    "wandb_entity",
                    "wandb_watch",
                    "wandb_name",
                    "logging_steps",
                    "eval_steps",
                    "save_steps",
                    "output_dir",
                ],
            },
        }

    def _format_type(self, field_info: dict[str, Any]) -> str:
        """Format field type information in a readable way."""
        if "anyOf" in field_info:
            types = []
            is_optional = False

            for option in field_info["anyOf"]:
                if option.get("type") == "null":
                    is_optional = True
                elif option.get("type"):
                    types.append(option["type"])
                elif "$ref" in option:
                    ref_name = option["$ref"].split("/")[-1]
                    types.append(ref_name)

            type_str = " | ".join(types) if types else "unknown"
            return f"{type_str} | None" if is_optional else type_str

        field_type = field_info.get("type", "unknown")

        if field_type == "array":
            items = field_info.get("items", {})
            if items.get("type"):
                item_type = items["type"]
            elif "$ref" in items:
                item_type = items["$ref"].split("/")[-1]
            else:
                item_type = "unknown"
            return f"list[{item_type}]"

        if field_type == "object":
            return "dict"

        return field_type

    def _get_yaml_example(
        self, field_name: str, field_info: dict[str, Any]
    ) -> str | None:
        """Generate a YAML example for the field."""
        default = field_info.get("default")

        # Common examples based on field names
        examples = {
            "base_model": "./llama-7b-hf",
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "sequence_len": 2048,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_epochs": 4,
            "learning_rate": 0.00003,
            "warmup_steps": 100,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "v_proj"],
            "load_in_8bit": True,
            "bf16": True,
            "flash_attention": True,
            "gradient_checkpointing": True,
            "output_dir": "./completed-model",
            "wandb_project": "my-finetuning-project",
        }

        if field_name in examples:
            return yaml.dump(
                {field_name: examples[field_name]}, default_style=None
            ).strip()

        if default is not None:
            return yaml.dump({field_name: default}, default_style=None).strip()

        return None

    def _categorize_fields(self, properties: dict[str, Any]) -> dict[str, list[str]]:
        """Categorize fields into logical groups."""
        categorized: dict[str, list[str]] = {
            category: [] for category in self.field_categories
        }
        uncategorized = []

        for field_name in properties:
            if field_name.startswith("_"):
                continue

            found_category = False
            for category, config in self.field_categories.items():
                if field_name in config["fields"]:
                    categorized[category].append(field_name)
                    found_category = True
                    break

            if not found_category:
                uncategorized.append(field_name)

        if uncategorized:
            categorized["Other Options"] = uncategorized

        return categorized

    def generate_qmd(
        self, model_class: type[BaseModel], title: str | None = None
    ) -> str:
        """Auto-generate config reference documentation."""

        if title is None:
            title = f"{model_class.__name__} Reference"

        schema = model_class.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Start building QMD content
        qmd_lines = [
            "---",
            f"title: {title}",
            "description: A complete list of all configuration options.",
            "---",
            "",
        ]

        # Generate table of contents
        categorized = self._categorize_fields(properties)
        qmd_lines.extend(
            [
                "## Table of Contents",
                "",
            ]
        )

        for category, fields in categorized.items():
            if fields:
                qmd_lines.append(
                    f"- [{category}](#{category.lower().replace(' ', '-')})"
                )

        qmd_lines.append("")

        # Generate sections
        for category, fields in categorized.items():
            if not fields:
                continue

            category_config = self.field_categories.get(category, {})
            category_desc = category_config.get("description", "")

            qmd_lines.extend([f"## {category}", "", category_desc, ""])

            for field_name in sorted(fields):
                if field_name not in properties:
                    continue

                field_info = properties[field_name]
                field_type = self._format_type(field_info)
                is_required = field_name in required

                description = field_info.get("description", "")
                default = field_info.get("default")

                # Field header
                qmd_lines.append(f"### `{field_name}`")
                qmd_lines.append("")

                # Type and requirement info
                type_info = f"**Type:** `{field_type}`"
                if is_required:
                    type_info += " *(required)*"
                qmd_lines.append(type_info)
                qmd_lines.append("")

                # Description
                if description:
                    qmd_lines.append(description)
                    qmd_lines.append("")

                # Default value
                if default is not None:
                    qmd_lines.append(f"**Default:** `{default}`")
                    qmd_lines.append("")

                # YAML example
                yaml_example = self._get_yaml_example(field_name, field_info)
                if yaml_example:
                    qmd_lines.extend(
                        ["**Example:**", "```yaml", yaml_example, "```", ""]
                    )

                qmd_lines.append("---")
                qmd_lines.append("")

        return "\n".join(qmd_lines)


# Usage example
def main():
    """Example usage of the enhanced generator."""
    generator = QuartoGenerator()

    print("Generating config reference content...")
    qmd_content = generator.generate_qmd(AxolotlInputConfig, "Config Reference")

    print("Writing to file...")
    with open("docs/config-reference.qmd", "w", encoding="utf-8") as f:
        f.write(qmd_content)
    print("Done!")


if __name__ == "__main__":
    main()
