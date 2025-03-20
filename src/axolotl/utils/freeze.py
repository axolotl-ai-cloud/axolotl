"""
module to freeze/unfreeze parameters by name
"""

import logging
import re
from typing import Callable, List, Tuple, Union

from axolotl.utils.distributed import is_main_process

LOG = logging.getLogger("axolotl.utils.freeze")


def freeze_layers_except(model, regex_patterns):
    """
    Freezes all layers of the given model except for the layers that match given regex patterns.
    Periods in the patterns are treated as literal periods, not as wildcard characters.

    Parameters:
    - model (nn.Module): The PyTorch model to be modified.
    - regex_patterns (list of str): List of regex patterns to match layer names to keep unfrozen.
      Note that you cannot use a dot as a wildcard character in the patterns since it is reserved for separating layer names.
      Also, to match the entire layer name, the pattern should start with "^" and end with "$", otherwise it will match any part of the layer name.
      The range pattern part is optional and it is not compiled as a regex pattern which means you must put "$" before the range pattern if you want to match the entire layer name.
      E.g., ["^model.embed_tokens.weight$[:32000]", "layers.2[0-9]+.block_sparse_moe.gate.[a-z]+$"]

    Returns:
    None; the model is modified in place.
    """
    if isinstance(regex_patterns, str):
        regex_patterns = [regex_patterns]

    patterns = [LayerNamePattern(pattern) for pattern in regex_patterns]

    # Unfreeze layers that match the regex patterns
    for name, param in model.named_parameters():
        param.requires_grad = False
        unfrozen_ranges = []
        for pattern in patterns:
            if not pattern.match(name):
                continue

            param.requires_grad = True

            if pattern.range is not None:
                unfrozen_ranges.append(pattern.range)

        merged_unfrozen_ranges = _merge_ranges(unfrozen_ranges, len(param))

        if param.requires_grad and is_main_process():
            unfrozen_ranges = (
                f" with ranges {merged_unfrozen_ranges}"
                if merged_unfrozen_ranges
                else ""
            )
            LOG.debug(f"Unfrozen {name}{unfrozen_ranges}")

        if not merged_unfrozen_ranges:
            continue

        # The range list we need is actually the inverted of the merged ranges
        ranges_to_freeze = _invert_ranges(merged_unfrozen_ranges, len(param))

        param.register_hook(_create_freeze_parameters_hook(ranges_to_freeze))

    if is_main_process() and all(
        not param.requires_grad for param in model.parameters()
    ):
        LOG.warning("All parameters are frozen. Model will not be trained.")


def _invert_ranges(
    given_ranges: List[Tuple[int, int]], layer_size: int
) -> List[Tuple[int, int]]:
    """
    Inverts a list of ranges to obtain the ranges not covered by the given ranges.

    Parameters:
    - given_ranges (List[Tuple[int, int]]): List of ranges to invert. Each range is represented as a tuple of start (inclusive) and end (exclusive) indices.
    - layer_size (int): The length of the layer. E.g., len(model.layer.weight)
    Returns:
    - List[Tuple[int, int]]: List of inverted ranges, where each range is represented as a tuple of start (inclusive) and end (exclusive) indices.
    """
    if not given_ranges:
        return [(0, layer_size)]

    inverted_ranges = []
    current_start = 0

    for start, end in sorted(given_ranges):
        if start > current_start:
            inverted_ranges.append((current_start, start))
        current_start = max(current_start, end)

    # Handle the case where the last given range does not reach the end of the total_size
    if current_start < layer_size:
        inverted_ranges.append((current_start, layer_size))

    return inverted_ranges


def _merge_ranges(
    given_ranges: List[Tuple[int, Union[int, None]]], layer_size: int
) -> List[Tuple[int, int]]:
    """
    Merges overlapping ranges and sorts the given ranges.

    This function takes a list of ranges and merges any overlapping ranges. The ranges are represented
    as tuples, where the first element is the start index (inclusive) and the second element is the end
    index (exclusive). The end index can be None, indicating that the range extends to the end of the
    sequence.

    Parameters:
    - given_ranges (List[Tuple[int, int | None]]): List of ranges to merge.
    - layer_size (int): The length of the layer. E.g., len(model.layer.weight)

    Returns:
    - List[Tuple[int, int]]: List of merged ranges, as start (inclusive) and end (exclusive) indices.
    """
    # End of each range can be determined now since we have the total size
    processed_ranges = [
        (start, end if end is not None else layer_size) for start, end in given_ranges
    ]
    for start, end in processed_ranges:
        if start < 0 or end > layer_size > 0 or start >= end:
            raise ValueError(f"invalid unfreeze range: start={start}, end={end}")

    # No need to merge if there's only one or no ranges
    if len(processed_ranges) <= 1:
        return processed_ranges

    sorted_ranges = sorted(processed_ranges)

    merged_ranges = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged_ranges[-1]
        if start <= prev_end:
            merged_ranges[-1] = (prev_start, max(prev_end, end))
        else:
            merged_ranges.append((start, end))

    return merged_ranges


def _create_freeze_parameters_hook(ranges_to_freeze: List[Tuple[int, int]]) -> Callable:
    """
    Create a hook to freeze parameters in specified ranges by setting their gradients to zero.

    This function takes a list of tuples representing the ranges of indices to freeze. Each tuple should contain
    two integers representing the start and end indices of the range.

    Parameters:
    - ranges_to_freeze (List[Tuple[int, int]]): Ranges of indices to freeze.

    Returns:
    - Callable: A hook function to be used with `register_hook` on parameters.

    Example usage:
    ```
    ranges_to_freeze = [(0, 10), (20, 30)]
    hook = _create_freeze_parameters_hook(ranges_to_freeze)
    model.register_hook(hook)
    ```
    """

    def freeze_parameters_hook(gradients):
        for start, end in ranges_to_freeze:
            gradients[start:end].zero_()

    return freeze_parameters_hook


class LayerNamePattern:
    """
    Represents a regex pattern for layer names, potentially including a parameter index range.
    """

    def __init__(self, pattern: str):
        """
        Initializes a new instance of the LayerNamePattern class.

        Parameters:
        - pattern (str): The regex pattern for layer names, potentially including a parameter index range.
        """
        self.raw_pattern = pattern
        name_pattern, self.range = self._parse_pattern(pattern)
        self.name_regex = re.compile(name_pattern.replace(".", "\\."))

    def match(self, name: str) -> bool:
        """
        Checks if the given layer name matches the regex pattern.

        Parameters:
        - name (str): The layer name to check.

        Returns:
        - bool: True if the layer name matches the pattern, False otherwise.
        """
        return self.name_regex.match(name) is not None

    def _parse_pattern(
        self, pattern: str
    ) -> Tuple[str, Union[Tuple[int, Union[int, None]], None]]:
        """
        Extracts the range pattern from the given pattern.

        Parameters:
        - pattern (str): The pattern to extract the range from.

        Returns:
        - Tuple[str, Tuple[int, int | None] | None]: A tuple containing the regex pattern to match the layer name without the range pattern and the range of layer indices to match, if specified.
        """
        match = re.match(r"^(.+)\[([0-9]*)(?::([0-9]*))?\]$", pattern)
        if not match:
            return pattern, None

        base_pattern, start_part, end_part = match.groups()

        if end_part is None and start_part.isdecimal():
            index = int(start_part)
            return base_pattern, (index, index + 1)

        # [:end] or [start:] or [start:end]
        start = int(start_part) if start_part else 0
        end = int(end_part) if end_part else None

        if end is not None and start >= end:
            raise ValueError(
                f"Invalid range in layer name pattern: {pattern}."
                "End of range must be greater than start."
            )
        return base_pattern, (start, end)
