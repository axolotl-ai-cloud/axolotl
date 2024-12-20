"""Utilities for YAML files."""

from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple, Union

import yaml


class YAMLOrderTracker:
    """Tracks the order of keys and section breaks in YAML files."""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.structure, self.needs_break = self._parse_yaml_structure()

    def _get_indentation_level(self, line: str) -> int:
        """Get the indentation level of a line."""
        return len(line) - len(line.lstrip())

    def _parse_yaml_structure(
        self,
    ) -> Tuple[Dict[str, Union[List[str], Dict]], Set[str]]:
        """Parse the YAML file to extract structure and identify section breaks."""
        with open(self.yaml_path, "r", encoding="utf-8") as file:
            contents = file.readlines()

        structure: OrderedDict = OrderedDict()
        needs_break = set()  # Track which keys should have a break before them
        current_path = []
        last_indentation = -1
        had_empty_line = False

        for line in contents:
            # Track empty lines and comments
            if not line.strip() or line.strip().startswith("#"):
                had_empty_line = True
                continue

            # Get indentation level and content
            indentation = self._get_indentation_level(line)
            content = line.strip()

            # Skip lines that don't define keys
            if ":" not in content:
                continue

            # Extract key
            key = content.split(":")[0].strip()

            # If this is a top-level key and we had an empty line, mark it
            if indentation == 0:
                if had_empty_line:
                    needs_break.add(key)
                had_empty_line = False

            # Handle indentation changes
            if indentation > last_indentation:
                current_path.append(key)
            elif indentation < last_indentation:
                levels_up = (last_indentation - indentation) // 2
                current_path = current_path[:-levels_up]
                current_path[-1] = key
            else:
                if current_path:
                    current_path[-1] = key

            # Update structure
            current_dict = structure
            for path_key in current_path[:-1]:
                if path_key not in current_dict:
                    current_dict[path_key] = OrderedDict()
                current_dict = current_dict[path_key]

            if current_path:
                if current_path[-1] not in current_dict:
                    current_dict[current_path[-1]] = OrderedDict()

            last_indentation = indentation

        return structure, needs_break


class OrderedDumper(yaml.SafeDumper):
    """Custom YAML dumper that maintains dictionary order."""


def ordered_dict_representer(dumper: OrderedDumper, data: Dict) -> Any:
    """Custom representer for dictionaries that maintains order."""
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


def reorder_dict(data: Dict, reference_structure: Dict) -> OrderedDict:
    """Reorder a dictionary based on a reference structure."""
    ordered = OrderedDict()

    # First add keys that are in the reference order
    for key in reference_structure:
        if key in data:
            if isinstance(reference_structure[key], dict) and isinstance(
                data[key], dict
            ):
                ordered[key] = reorder_dict(data[key], reference_structure[key])
            else:
                ordered[key] = data[key]

    # Then add any remaining keys that weren't in the reference
    for key in data:
        if key not in ordered:
            ordered[key] = data[key]

    return ordered


def dump_yaml_preserved_order(
    data: Dict, reference_yaml_path: str, output_path: str
) -> None:
    """Dump YAML file while preserving nested order and normalized spacing."""
    # Get reference structure and spacing
    tracker = YAMLOrderTracker(reference_yaml_path)

    # Reorder the data
    ordered_data = reorder_dict(data, tracker.structure)

    # Register the custom representer
    OrderedDumper.add_representer(dict, ordered_dict_representer)
    OrderedDumper.add_representer(OrderedDict, ordered_dict_representer)

    # First dump to string
    yaml_str = yaml.dump(
        ordered_data, Dumper=OrderedDumper, sort_keys=False, default_flow_style=False
    )

    # Add spacing according to reference
    lines = yaml_str.split("\n")
    result_lines: List[str] = []
    current_line = 0

    while current_line < len(lines):
        line = lines[current_line]
        if line.strip() and ":" in line and not line.startswith(" "):  # Top-level key
            key = line.split(":")[0].strip()
            if key in tracker.needs_break:
                # Add single empty line before this key
                if result_lines and result_lines[-1] != "":
                    result_lines.append("")
        result_lines.append(line)
        current_line += 1

    # Write the final result
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(result_lines))
