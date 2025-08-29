# type: ignore

"""
Quarto documentation generation from Pydantic models. Uses Pydantic model source code
to automatically group fields, including inherited fields from parent classes.
"""

import ast
import inspect
import textwrap
import types
import typing
from typing import Any, FrozenSet, Type, Union

from pydantic import BaseModel

from axolotl.utils.schemas.config import AxolotlInputConfig


class QuartoGenerator:
    """Generate Quarto documentation from Pydantic models."""

    def __init__(self):
        self._class_fields_cache = {}
        self._inheritance_map_cache = {}
        self._nested_models_cache = {}

    def _get_direct_fields(self, cls: Type[BaseModel]) -> FrozenSet[str]:
        """Get fields defined directly in a single class (not inherited)."""
        if cls in self._class_fields_cache:
            return self._class_fields_cache[cls]

        fields = set()

        # Get annotated fields
        if hasattr(cls, "__annotations__"):
            fields.update(cls.__annotations__.keys())

        # Filter out private/special methods
        fields = {f for f in fields if not f.startswith("_")}

        result = frozenset(fields)
        self._class_fields_cache[cls] = result
        return result

    def _is_pydantic_model(self, type_obj) -> bool:
        """Check if a type is a Pydantic BaseModel."""
        return inspect.isclass(type_obj) and issubclass(type_obj, BaseModel)

    def _extract_nested_type(self, field_type) -> Any:
        """Extract the actual type from complex type annotations."""
        # Handle Annotated types (Python 3.9+)
        if hasattr(typing, "get_origin") and hasattr(typing, "get_args"):
            origin = typing.get_origin(field_type)
            args = typing.get_args(field_type)

            if origin is not None:
                # Handle Annotated[SomeType, ...] - extract the first argument
                if hasattr(typing, "Annotated") and origin is typing.Annotated:
                    if args:
                        return self._extract_nested_type(
                            args[0]
                        )  # Recursively process the actual type

                # Handle list[SomeType], List[SomeType], etc.
                elif origin in (list, typing.List):
                    if args:
                        return self._extract_nested_type(
                            args[0]
                        )  # Extract element type

                # Handle Union types (including | syntax)
                elif origin is typing.Union:
                    # Get non-None types from the Union
                    non_none_types = [arg for arg in args if arg is not type(None)]
                    if len(non_none_types) >= 1:
                        # Prioritize Pydantic models over primitive types
                        pydantic_models = [
                            arg
                            for arg in non_none_types
                            if self._is_pydantic_model(arg)
                        ]
                        if pydantic_models:
                            # Return the first Pydantic model found
                            return self._extract_nested_type(pydantic_models[0])

                        # No Pydantic models, return the first non-None type
                        return self._extract_nested_type(non_none_types[0])

        # Handle new Python 3.10+ union syntax (PeftConfig | None)
        if hasattr(field_type, "__class__") and field_type.__class__ is types.UnionType:
            # Get non-None types from the Union
            non_none_types = [
                arg for arg in field_type.__args__ if arg is not type(None)
            ]
            if len(non_none_types) >= 1:
                # Prioritize Pydantic models over primitive types
                pydantic_models = [
                    arg for arg in non_none_types if self._is_pydantic_model(arg)
                ]
                if pydantic_models:
                    return self._extract_nested_type(pydantic_models[0])
                return self._extract_nested_type(non_none_types[0])

        # Handle old typing.Union syntax (fallback)
        if hasattr(field_type, "__origin__"):
            if field_type.__origin__ is Union:
                # Get non-None types from the Union
                non_none_types = [
                    arg for arg in field_type.__args__ if arg is not type(None)
                ]
                if len(non_none_types) >= 1:
                    # Prioritize Pydantic models over primitive types
                    pydantic_models = [
                        arg for arg in non_none_types if self._is_pydantic_model(arg)
                    ]
                    if pydantic_models:
                        return self._extract_nested_type(pydantic_models[0])
                    return self._extract_nested_type(non_none_types[0])
            # Handle other generic types like dict[str, Any], etc.
            elif hasattr(field_type, "__args__"):
                return field_type

        return field_type

    def _extract_all_pydantic_models_from_type(
        self, field_type
    ) -> list[type[BaseModel]]:
        """Extract all Pydantic models from a type annotation, including from Unions."""
        models = []

        if field_type is None:
            return models

        # Handle Annotated types
        if hasattr(typing, "get_origin") and hasattr(typing, "get_args"):
            origin = typing.get_origin(field_type)
            args = typing.get_args(field_type)

            if origin is not None:
                # Handle Annotated[SomeType, ...] - extract from the first argument
                if hasattr(typing, "Annotated") and origin is typing.Annotated:
                    if args:
                        models.extend(
                            self._extract_all_pydantic_models_from_type(args[0])
                        )
                    return models

                # Handle list[SomeType], List[SomeType], etc.
                if origin in (list, typing.List):
                    if args:
                        models.extend(
                            self._extract_all_pydantic_models_from_type(args[0])
                        )
                    return models

                # Handle Union types
                if origin is typing.Union:
                    for arg in args:
                        if arg is not type(None):  # Skip None type
                            models.extend(
                                self._extract_all_pydantic_models_from_type(arg)
                            )
                    return models

        # Handle new Python 3.10+ union syntax
        if hasattr(field_type, "__class__") and field_type.__class__ is types.UnionType:
            for arg in field_type.__args__:
                if arg is not type(None):  # Skip None type
                    models.extend(self._extract_all_pydantic_models_from_type(arg))
            return models

        # Handle old typing.Union syntax (fallback)
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            for arg in field_type.__args__:
                if arg is not type(None):  # Skip None type
                    models.extend(self._extract_all_pydantic_models_from_type(arg))
            return models

        # Check if this type itself is a Pydantic model
        if self._is_pydantic_model(field_type):
            models.append(field_type)

        return models

    def _get_nested_models(
        self, model_class: type[BaseModel], visited=None
    ) -> dict[str, type[BaseModel]]:
        """Get all nested Pydantic models from a model class."""
        if visited is None:
            visited = set()

        # Avoid infinite recursion
        if model_class in visited:
            return {}

        if model_class in self._nested_models_cache:
            return self._nested_models_cache[model_class]

        visited.add(model_class)
        nested_models = {}

        # Check all fields in the model
        for field_info in model_class.model_fields.values():
            field_type = self._extract_nested_type(field_info.annotation)

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
            cls
            for cls in child_class.__mro__
            if cls not in (BaseModel, object) and hasattr(cls, "__annotations__")
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
                    if isinstance(body_node, ast.AnnAssign) and isinstance(
                        body_node.target, ast.Name
                    ):
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

        # Get all Pydantic base classes in MRO order (most specific first)
        # This puts AxolotlInputConfig fields first, then parent class fields
        pydantic_classes = [
            cls
            for cls in model_class.__mro__
            if cls in inheritance_map and inheritance_map[cls]
        ]

        # Extract groups from each class
        for cls in pydantic_classes:
            class_groups = self._extract_field_groups_from_source(cls)
            for group in class_groups:
                all_groups.append(group)

        # If no groups found, create a default grouping by class
        if not all_groups:
            for cls in pydantic_classes:
                fields_in_class = inheritance_map[cls]
                if fields_in_class:
                    all_groups.append(
                        {
                            "fields": list(fields_in_class),
                        }
                    )

        return all_groups

    def _extract_field_groups_from_source(
        self, model_class: type[BaseModel]
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
                        "fields": list(fields_in_class),
                    }
                ]
            return []

        groups = []
        current_group_fields = []
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
                groups.append(
                    {
                        "fields": current_group_fields.copy(),
                        "description": current_group_comment,
                    }
                )
                current_group_fields = []
                current_group_comment = None

            current_group_fields.append(field_name)

        # Add the final group
        if current_group_fields:
            groups.append(
                {
                    "fields": current_group_fields,
                    "description": current_group_comment,
                }
            )

        return groups

    def _generate_field_documentation(
        self,
        model_class: type[BaseModel],
        field_name: str,
        field_info: dict,
        field_type_str: str,
        is_required: bool,
        indent_level: int = 0,
        visited_models: set = None,
    ) -> list[str]:
        """Generate documentation for a single field, expanding nested models inline."""
        if visited_models is None:
            visited_models = set()

        lines = []
        indent = "  " * indent_level

        # Get the actual field type for nested model detection
        if field_name in model_class.model_fields:
            pydantic_field_info = model_class.model_fields[field_name]
            actual_field_type = pydantic_field_info.annotation
        else:
            actual_field_type = None

        # Add description comment if available
        description = field_info.get("description", "")
        if description:
            wrapped_lines = self._wrap_comment(description, width=88 - len(indent))
            for line in wrapped_lines:
                lines.append(f"{indent}{line}")

        # Extract nested Pydantic models from the type annotation
        nested_models = self._extract_all_pydantic_models_from_type(actual_field_type)

        # Filter out already visited models to prevent infinite recursion
        expandable_models = [
            model for model in nested_models if model not in visited_models
        ]

        if expandable_models:
            # This field contains Pydantic models that can be expanded

            # Show the field with its full type annotation
            field_line = f"{indent}{field_name}: {field_type_str}"
            if field_info.get("default") is not None:
                field_line += f" = {field_info['default']}"
            if is_required:
                field_line += " (required)"
            lines.append(field_line)

            # Add to visited to prevent infinite recursion
            new_visited = visited_models.copy()
            new_visited.update(expandable_models)

            # Expand each nested Pydantic model
            for i, nested_model in enumerate(expandable_models):
                if i > 0:
                    lines.append("\n")
                lines.append(f"{indent}  # For {nested_model.__name__}:")

                # Get nested model schema
                try:
                    nested_schema = nested_model.model_json_schema()
                    nested_properties = nested_schema.get("properties", {})
                    nested_required = nested_schema.get("required", [])
                except Exception:
                    # Fallback: use model fields directly
                    nested_properties = {}
                    nested_required = []
                    for (
                        nested_field_name,
                        nested_field_info,
                    ) in nested_model.model_fields.items():
                        nested_description = ""
                        if (
                            hasattr(nested_field_info, "json_schema_extra")
                            and nested_field_info.json_schema_extra
                        ):
                            nested_description = (
                                nested_field_info.json_schema_extra.get(
                                    "description", ""
                                )
                            )
                        elif (
                            hasattr(nested_field_info, "description")
                            and nested_field_info.description
                        ):
                            nested_description = nested_field_info.description

                        nested_default_val = None
                        if (
                            hasattr(nested_field_info, "default")
                            and nested_field_info.default is not None
                        ):
                            if str(nested_field_info.default) != "PydanticUndefined":
                                nested_default_val = nested_field_info.default

                        nested_properties[nested_field_name] = {
                            "type": "unknown",
                            "description": nested_description,
                            "default": nested_default_val,
                        }

                        if nested_field_info.is_required():
                            nested_required.append(nested_field_name)

                # Get field groups for the nested model
                nested_field_groups = self._extract_field_groups_from_all_classes(
                    nested_model
                )

                # Generate nested fields with increased indentation
                for i, group in enumerate(nested_field_groups):
                    if not group["fields"]:
                        continue

                    # Add blank line between groups (except before first group)
                    if i > 0:
                        lines.append("")

                    # Process nested fields
                    for nested_field_name in group["fields"]:
                        if nested_field_name not in nested_properties:
                            continue

                        nested_field_info = nested_properties[nested_field_name]
                        nested_field_type = self._extract_type_from_source(
                            nested_model, nested_field_name
                        )
                        nested_is_required = nested_field_name in nested_required

                        # Recursively generate documentation for nested field
                        nested_lines = self._generate_field_documentation(
                            nested_model,
                            nested_field_name,
                            nested_field_info,
                            nested_field_type,
                            nested_is_required,
                            indent_level + 1,
                            new_visited,
                        )
                        lines.extend(nested_lines)
        else:
            # Regular field (no expandable nested models)
            field_line = f"{indent}{field_name}: {field_type_str}"
            if field_info.get("default") is not None:
                field_line += f" = {field_info['default']}"
            if is_required:
                field_line += " (required)"
            lines.append(field_line)

        return lines

    def generate_qmd(
        self,
        model_class: type[BaseModel],
        title: str | None = None,
        expand_nested: bool = True,
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

        # Generate one big code block with all fields (inline nested expansion)
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

                if expand_nested:
                    # Check if this field has nested models
                    if field_name in model_class.model_fields:
                        pydantic_field_info = model_class.model_fields[field_name]
                        nested_models = self._extract_all_pydantic_models_from_type(
                            pydantic_field_info.annotation
                        )
                        has_nested = bool(nested_models)
                    else:
                        has_nested = False

                    # Add blank line before nested config
                    if has_nested:
                        qmd_lines.append("")

                    # Use the new inline generation method
                    field_lines = self._generate_field_documentation(
                        model_class,
                        field_name,
                        field_info,
                        field_type,
                        is_required,
                        indent_level=0,
                        visited_models=set(),
                    )
                    qmd_lines.extend(field_lines)

                    # Add blank line after nested config
                    if has_nested:
                        qmd_lines.append("")
                else:
                    # Original simple approach
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

        # Join all lines and clean up any double newlines
        content = "\n".join(qmd_lines)

        # Replace multiple consecutive newlines with just two newlines (one blank line)
        import re

        content = re.sub(r"\n{3,}", "\n\n", content)

        # Ensure single newline at the very end
        content = content.rstrip("\n") + "\n"

        return content


def main():
    generator = QuartoGenerator()

    print("Generating config reference content...")
    qmd_content = generator.generate_qmd(AxolotlInputConfig, "Config Reference", True)

    print("Writing to file...")
    with open("docs/config-reference.qmd", "w", encoding="utf-8") as f:
        f.write(qmd_content)
    print("Done!")


if __name__ == "__main__":
    main()
