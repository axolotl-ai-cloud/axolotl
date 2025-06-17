# type: ignore

"""
Quarto documentation generation from Pydantic models. Uses Pydantic model source code
to automatically group fields, including inherited fields from parent classes.
"""

import ast
import inspect
import textwrap
from functools import lru_cache
from typing import Any, FrozenSet, Type

import yaml
from pydantic import BaseModel

from axolotl.utils.schemas.config import AxolotlInputConfig


class QuartoGenerator:
    """Generate Quarto documentation from Pydantic models."""

    def __init__(self):
        self._class_fields_cache = {}
        self._inheritance_map_cache = {}
        self._nested_models_cache = {}

    @lru_cache(maxsize=128)
    def _get_direct_fields(self, cls: Type[BaseModel]) -> FrozenSet[str]:
        """Get fields defined directly in a single class (not inherited)."""
        if cls in self._class_fields_cache:
            return self._class_fields_cache[cls]
        
        fields = set()
        
        # Get annotated fields
        if hasattr(cls, '__annotations__'):
            fields.update(cls.__annotations__.keys())
        
        # Also check for non-annotated assignments via AST (fallback)
        try:
            source = inspect.getsource(cls)
            tree = ast.parse(source)
            
            # Find the class definition
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                    for body_node in node.body:
                        if isinstance(body_node, ast.Assign):
                            for target in body_node.targets:
                                if isinstance(target, ast.Name):
                                    fields.add(target.id)
                    break
        except (OSError, TypeError):
            pass
        
        # Filter out private/special methods
        fields = {f for f in fields if not f.startswith('_')}
        
        result = frozenset(fields)
        self._class_fields_cache[cls] = result
        return result

    def _is_pydantic_model(self, type_obj) -> bool:
        """Check if a type is a Pydantic BaseModel."""
        try:
            return (hasattr(type_obj, '__bases__') and 
                    any(base.__name__ == 'BaseModel' for base in type_obj.__bases__))
        except (AttributeError, TypeError):
            return False

    def _extract_nested_type(self, field_type) -> tuple[Any, bool]:
        """Extract the actual type from Union types like PeftConfig | None."""
        is_optional = False
        
        # Handle Union types (like PeftConfig | None)
        if hasattr(field_type, '__origin__'):
            if field_type.__origin__ is Union:
                # Get non-None types from the Union
                non_none_types = [arg for arg in field_type.__args__ if arg is not type(None)]
                if len(non_none_types) == 1:
                    is_optional = type(None) in field_type.__args__
                    return non_none_types[0], is_optional
                elif len(non_none_types) > 1:
                    # Multiple non-None types, keep as Union
                    return field_type, is_optional
            # Handle other generic types like list[str], dict[str, Any], etc.
            elif hasattr(field_type, '__args__'):
                return field_type, is_optional
        
        return field_type, is_optional

    def _get_nested_models(self, model_class: type[BaseModel], visited=None) -> dict[str, type[BaseModel]]:
        """Get all nested Pydantic models from a model class."""
        if visited is None:
            visited = set()
        
        # Avoid infinite recursion
        if model_class.__name__ in visited:
            return {}
        
        if model_class in self._nested_models_cache:
            return self._nested_models_cache[model_class]
        
        visited.add(model_class.__name__)
        nested_models = {}
        
        # Check all fields in the model
        for field_name, field_info in model_class.model_fields.items():
            field_type, is_optional = self._extract_nested_type(field_info.annotation)
            
            if self._is_pydantic_model(field_type):
                nested_models[field_type.__name__] = field_type
                # Recursively get nested models from this nested model
                deeper_nested = self._get_nested_models(field_type, visited.copy())
                nested_models.update(deeper_nested)
        
        self._nested_models_cache[model_class] = nested_models
        return nested_models

    def _build_inheritance_map(self, child_class: Type[BaseModel]):
        """Build inheritance map for a class and all its parents."""
        if child_class in self._inheritance_map_cache:
            return self._inheritance_map_cache[child_class]
        
        inheritance_map = {}
        
        # Get MRO and filter out BaseModel and object
        mro_classes = [
            cls for cls in child_class.__mro__ 
            if cls not in (BaseModel, object) and hasattr(cls, '__annotations__')
        ]
        
        # Process each class in the MRO
        for cls in mro_classes:
            inheritance_map[cls] = self._get_direct_fields(cls)
        
        self._inheritance_map_cache[child_class] = inheritance_map
        return inheritance_map

    def _wrap_comment(self, text: str, width: int = 88) -> list[str]:
        """Wrap a comment to specified width, accounting for '# ' prefix."""
        if not text.strip():
            return ["#"]

        # Account for "# " prefix (2 characters)
        content_width = width - 2
        wrapped_lines = textwrap.wrap(text, width=content_width)
        return [f"# {line}" for line in wrapped_lines]

    def _extract_type_from_source(
        self, model_class: type[BaseModel], field_name: str
    ) -> str:
        """Extract the actual type annotation text from source code, checking inheritance chain."""
        # Use inheritance map to check classes efficiently
        inheritance_map = self._build_inheritance_map(model_class)
        
        # Check classes in MRO order
        for cls in model_class.__mro__:
            if cls in inheritance_map and field_name in inheritance_map[cls]:
                type_annotation = self._get_type_from_class_source(cls, field_name)
                if type_annotation != "unknown":
                    return type_annotation
        
        return "unknown"

    def _get_type_from_class_source(self, class_obj: type, field_name: str) -> str:
        """Extract type annotation from a specific class's source code."""
        try:
            source = inspect.getsource(class_obj)
            tree = ast.parse(source)
        except (OSError, TypeError):
            return "unknown"

        # Find the class definition
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_obj.__name__:
                # Find the field assignment
                for body_node in node.body:
                    if isinstance(body_node, ast.AnnAssign) and isinstance(body_node.target, ast.Name):
                        if body_node.target.id == field_name and body_node.annotation:
                            return ast.unparse(body_node.annotation)
                break

        return "unknown"

    def _extract_field_groups_from_all_classes(
        self, model_class: type[BaseModel]
    ) -> list[dict]:
        """Extract field groups from all classes in the inheritance hierarchy."""
        all_groups = []
        inheritance_map = self._build_inheritance_map(model_class)
        
        # Get all Pydantic base classes in reverse MRO order (most specific first)
        pydantic_classes = [
            cls for cls in reversed(model_class.__mro__)
            if cls in inheritance_map and inheritance_map[cls]
        ]
        
        # Extract groups from each class
        for cls in pydantic_classes:
            class_groups = self._extract_field_groups_from_source(cls, model_class)
            for group in class_groups:
                # Add class name to group title for context
                if len(pydantic_classes) > 1:
                    group['title'] = f"{cls.__name__}: {group['title']}"
                    group['class_name'] = cls.__name__
                all_groups.append(group)
        
        # If no groups found, create a default grouping by class
        if not all_groups:
            for cls in pydantic_classes:
                fields_in_class = inheritance_map[cls]
                if fields_in_class:
                    all_groups.append({
                        'title': cls.__name__,
                        'fields': list(fields_in_class),
                        'class_name': cls.__name__
                    })
        
        return all_groups

    def _extract_field_groups_from_source(
        self, model_class: type[BaseModel], child_class: type[BaseModel] = None
    ) -> list[dict]:
        """Extract field groups from source code based on blank lines and comments."""
        try:
            source = inspect.getsource(model_class)
            tree = ast.parse(source)
        except (OSError, TypeError):
            # Fallback if we can't get source code
            fields_in_class = self._get_direct_fields(model_class)
            if fields_in_class:
                return [
                    {
                        "title": "Configuration Options",
                        "fields": list(fields_in_class),
                    }
                ]
            return []

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
            fields_in_class = self._get_direct_fields(model_class)
            if fields_in_class:
                return [
                    {
                        "title": "Configuration Options", 
                        "fields": list(fields_in_class),
                    }
                ]
            return []

        # Parse the source lines to detect groupings
        source_lines = source.split("\n")

        # Get fields that are actually defined in this specific class
        fields_in_class = self._get_direct_fields(model_class)

        # Find assignments that correspond to model fields for THIS class only
        field_assignments = []
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_name = node.target.id
                if field_name in fields_in_class:
                    field_assignments.append(
                        {
                            "name": field_name,
                            "lineno": node.lineno,
                            "end_lineno": getattr(node, "end_lineno", node.lineno),
                        }
                    )

        if not field_assignments:
            if fields_in_class:
                return [
                    {
                        "title": "Configuration Options",
                        "fields": list(fields_in_class),
                    }
                ]
            return []

        # Sort by line number
        field_assignments.sort(key=lambda x: x["lineno"])

        # Group fields based on blank lines and comments
        for i, field_info in enumerate(field_assignments):
            field_name = field_info["name"]
            current_line = field_info["lineno"]

            # Check if this starts a new group (blank line before or significant gap)
            is_new_group = False

            if i == 0:
                is_new_group = True
            else:
                prev_end_line = field_assignments[i - 1]["end_lineno"]

                # Check for blank lines or comments between fields
                lines_between = source_lines[prev_end_line : current_line - 1]
                has_blank_line = any(line.strip() == "" for line in lines_between)
                has_comment = any(
                    line.strip().startswith("#") for line in lines_between
                )

                # Start new group if there's a blank line or comment, or significant gap
                if has_blank_line or has_comment or (current_line - prev_end_line > 3):
                    is_new_group = True

            if is_new_group and current_group_fields:
                # Save the previous group
                title = current_group_title or f"Group {len(groups) + 1}"
                groups.append(
                    {
                        "title": title,
                        "fields": current_group_fields.copy(),
                        "description": current_group_comment,
                    }
                )
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
                        if line.startswith("#") and not line.startswith("# "):
                            # This might be a section comment
                            comment_text = line.lstrip("#").strip()
                            if len(
                                comment_text
                            ) > 0 and not comment_text.lower().startswith(
                                ("todo", "fixme", "note")
                            ):
                                current_group_title = comment_text.title()
                                current_group_comment = comment_text

            current_group_fields.append(field_name)

        # Add the final group
        if current_group_fields:
            title = current_group_title or f"Group {len(groups) + 1}"
            groups.append(
                {
                    "title": title,
                    "fields": current_group_fields,
                    "description": current_group_comment,
                }
            )

        return groups

    def _generate_nested_model_sections(self, nested_models: dict[str, type[BaseModel]]) -> list[str]:
        """Generate documentation sections for nested models."""
        sections = []
        
        for model_name, model_class in nested_models.items():
            # Generate schema for the nested model
            try:
                schema = model_class.model_json_schema()
                properties = schema.get("properties", {})
                required = schema.get("required", [])
            except Exception:
                # Fallback: use model fields directly
                properties = {}
                required = []
                for field_name, field_info in model_class.model_fields.items():
                    description = ""
                    if (hasattr(field_info, "json_schema_extra") and 
                        field_info.json_schema_extra):
                        description = field_info.json_schema_extra.get("description", "")
                    elif hasattr(field_info, "description") and field_info.description:
                        description = field_info.description

                    default_val = None
                    if hasattr(field_info, "default") and field_info.default is not None:
                        if str(field_info.default) != "PydanticUndefined":
                            default_val = field_info.default

                    properties[field_name] = {
                        "type": "unknown",
                        "description": description,
                        "default": default_val,
                    }

                    if field_info.is_required():
                        required.append(field_name)

            # Start the section
            section_lines = [
                f"## {model_name}",
                "",
                f"*{model_class.__doc__ or f'Configuration options for {model_name}'}*",
                "",
                "```yaml"
            ]

            # Get field groups for this nested model
            field_groups = self._extract_field_groups_from_all_classes(model_class)

            for i, group in enumerate(field_groups):
                if not group["fields"]:
                    continue

                # Add blank line between groups (except before first group)
                if i > 0:
                    section_lines.append("")

                # Process fields in the order they appear in source
                for field_name in group["fields"]:
                    if field_name not in properties:
                        continue

                    field_info = properties[field_name]
                    field_type = self._extract_type_from_source(model_class, field_name)
                    is_required = field_name in required

                    description = field_info.get("description", "")
                    default = field_info.get("default")

                    # Add wrapped comment for description
                    if description:
                        wrapped_lines = self._wrap_comment(description)
                        section_lines.extend(wrapped_lines)

                    line = f"{field_name}: {field_type}"
                    if default is not None:
                        line += f" = {default}"
                    if is_required:
                        line += " (required)"
                    section_lines.append(line)

            section_lines.extend(["```", ""])
            sections.append("\n".join(section_lines))

        return sections

    def generate_qmd(
        self, model_class: type[BaseModel], title: str | None = None, 
        expand_nested: bool = True
    ) -> str:
        """Auto-generate config reference documentation including inherited fields."""

        if title is None:
            title = f"{model_class.__name__} Reference"

        # Try to get JSON schema, with fallback for serialization issues
        try:
            schema = model_class.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])
        except Exception as e:
            print(
                f"Warning: Could not generate JSON schema ({e}). Using model fields instead."
            )
            # Fallback: use model fields directly
            properties = {}
            required = []
            for field_name, field_info in model_class.model_fields.items():
                # Extract description from json_schema_extra or field info
                description = ""
                if (
                    hasattr(field_info, "json_schema_extra")
                    and field_info.json_schema_extra
                ):
                    description = field_info.json_schema_extra.get("description", "")
                elif hasattr(field_info, "description") and field_info.description:
                    description = field_info.description

                # Get default value
                default_val = None
                if hasattr(field_info, "default") and field_info.default is not None:
                    # Handle special Pydantic default markers
                    if str(field_info.default) != "PydanticUndefined":
                        default_val = field_info.default

                properties[field_name] = {
                    "type": "unknown",
                    "description": description,
                    "default": default_val,
                }

                if field_info.is_required():
                    required.append(field_name)

        # Extract field groups from all classes in inheritance hierarchy
        field_groups = self._extract_field_groups_from_all_classes(model_class)

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
                field_type = self._extract_type_from_source(model_class, field_name)
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

        # Add nested model sections if requested
        if expand_nested:
            nested_models = self._get_nested_models(model_class)
            if nested_models:
                qmd_lines.extend(["", "# Nested Configuration Objects", ""])
                nested_sections = self._generate_nested_model_sections(nested_models)
                qmd_lines.extend(nested_sections)

        return "\n".join(qmd_lines)

    def clear_cache(self):
        """Clear all caches for memory management."""
        self._class_fields_cache.clear()
        self._inheritance_map_cache.clear()
        self._nested_models_cache.clear()
        self._get_direct_fields.cache_clear()


def main():
    generator = QuartoGenerator()

    print("Generating config reference content...")
    qmd_content = generator.generate_qmd(AxolotlInputConfig, "Config Reference")

    print("Writing to file...")
    with open("docs/config-reference.qmd", "w", encoding="utf-8") as f:
        f.write(qmd_content)
    print("Done!")


if __name__ == "__main__":
    main()
