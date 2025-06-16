#!/usr/bin/env python3
"""
Enhanced Quarto documentation generator from Pydantic models.
Uses source code structure to automatically group fields.
"""

import ast
import inspect
import textwrap
from typing import Any

import yaml
from pydantic import BaseModel

from axolotl.utils.schemas.config import AxolotlInputConfig


class QuartoGenerator:
    """Generate Quarto documentation from Pydantic models."""

    def _wrap_comment(self, text: str, width: int = 88) -> list[str]:
        """Wrap a comment to specified width, accounting for '# ' prefix."""
        if not text.strip():
            return ["#"]
        
        # Account for "# " prefix (2 characters)
        content_width = width - 2
        wrapped_lines = textwrap.wrap(text, width=content_width)
        return [f"# {line}" for line in wrapped_lines]

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

    def _extract_field_groups_from_source(self, model_class: type[BaseModel]) -> list[dict]:
        """Extract field groups from source code based on blank lines and comments."""
        try:
            source = inspect.getsource(model_class)
            tree = ast.parse(source)
        except (OSError, TypeError):
            # Fallback if we can't get source code
            return [{"title": "Configuration Options", "fields": list(model_class.model_fields.keys())}]

        groups = []
        current_group_fields = []
        current_group_title = None
        current_group_comment = None
        
        # Find the class definition
        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == model_class.__name__:
                class_node = node
                break
        
        if not class_node:
            return [{"title": "Configuration Options", "fields": list(model_class.model_fields.keys())}]

        # Parse the source lines to detect groupings
        source_lines = source.split('\n')
        
        # Find assignments that correspond to model fields
        field_assignments = []
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_name = node.target.id
                if field_name in model_class.model_fields:
                    field_assignments.append({
                        'name': field_name,
                        'lineno': node.lineno,
                        'end_lineno': getattr(node, 'end_lineno', node.lineno)
                    })

        if not field_assignments:
            return [{"title": "Configuration Options", "fields": list(model_class.model_fields.keys())}]

        # Sort by line number
        field_assignments.sort(key=lambda x: x['lineno'])

        # Group fields based on blank lines and comments
        for i, field_info in enumerate(field_assignments):
            field_name = field_info['name']
            current_line = field_info['lineno']
            
            # Check if this starts a new group (blank line before or significant gap)
            is_new_group = False
            
            if i == 0:
                is_new_group = True
            else:
                prev_end_line = field_assignments[i-1]['end_lineno']
                
                # Check for blank lines or comments between fields
                lines_between = source_lines[prev_end_line:current_line-1]
                has_blank_line = any(line.strip() == '' for line in lines_between)
                has_comment = any(line.strip().startswith('#') for line in lines_between)
                
                # Start new group if there's a blank line or comment, or significant gap
                if has_blank_line or has_comment or (current_line - prev_end_line > 3):
                    is_new_group = True

            if is_new_group and current_group_fields:
                # Save the previous group
                title = current_group_title or f"Group {len(groups) + 1}"
                groups.append({
                    "title": title,
                    "fields": current_group_fields.copy(),
                    "description": current_group_comment
                })
                current_group_fields = []
                current_group_title = None
                current_group_comment = None

            # Look for a comment that might serve as a group title
            if is_new_group:
                # Check lines before this field for comments
                start_check = max(0, current_line - 5)
                for line_idx in range(start_check, current_line - 1):
                    if line_idx < len(source_lines):
                        line = source_lines[line_idx].strip()
                        if line.startswith('#') and not line.startswith('# '):
                            # This might be a section comment
                            comment_text = line.lstrip('#').strip()
                            if len(comment_text) > 0 and not comment_text.lower().startswith(('todo', 'fixme', 'note')):
                                current_group_title = comment_text.title()
                                current_group_comment = comment_text

            current_group_fields.append(field_name)

        # Add the final group
        if current_group_fields:
            title = current_group_title or f"Group {len(groups) + 1}"
            groups.append({
                "title": title,
                "fields": current_group_fields,
                "description": current_group_comment
            })

        # If no groups were created, create a default one
        if not groups:
            groups.append({
                "title": "Configuration Options",
                "fields": list(model_class.model_fields.keys())
            })

        return groups

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

    def generate_qmd(
        self, model_class: type[BaseModel], title: str | None = None
    ) -> str:
        """Auto-generate config reference documentation."""

        if title is None:
            title = f"{model_class.__name__} Reference"

        schema = model_class.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Extract field groups from source code
        field_groups = self._extract_field_groups_from_source(model_class)

        # Start building QMD content
        qmd_lines = [
            "---",
            f"title: {title}",
            "description: A complete list of all configuration options.",
            "---",
            "",
        ]

        # Generate one big code block with all fields
        qmd_lines.append("```yaml")

        for i, group in enumerate(field_groups):
            if not group["fields"]:
                continue

            # Add blank line between groups (except before first group)
            if i > 0:
                qmd_lines.append("")

            # Process fields in the order they appear in source
            for field_name in group["fields"]:
                if field_name not in properties:
                    continue

                field_info = properties[field_name]
                field_type = self._format_type(field_info)
                is_required = field_name in required

                description = field_info.get("description", "")
                default = field_info.get("default")

                # Add wrapped comment for description
                if description:
                    wrapped_lines = self._wrap_comment(description)
                    qmd_lines.extend(wrapped_lines)
                    
                line = f"{field_name}: {field_type}"
                if default is not None:
                    line += f" = {default}"
                if is_required:
                    line += " (required)"
                qmd_lines.append(line)

        qmd_lines.append("```")

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
