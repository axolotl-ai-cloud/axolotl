"""Atomic text dataset export for inspecting rewritten responses."""

import json
import tempfile
from pathlib import Path

from filelock import FileLock


def export_dataset(cache: Path, output_dir: str, *, seed: int) -> Path:
    """Export readable rows while preserving exact tokens in the sampling cache."""
    destination = Path(output_dir).resolve() / "projection-sampling" / "rewritten.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    with FileLock(str(destination) + ".lock"):
        try:
            with (
                cache.open(encoding="utf-8") as source,
                tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=destination.parent,
                    suffix=".jsonl",
                    delete=False,
                    encoding="utf-8",
                ) as output,
            ):
                temporary = Path(output.name)
                for line in source:
                    record = json.loads(line)
                    fields = (
                        ("messages", "tools")
                        if "messages" in record
                        else ("prompt", "response", "expert_response")
                    )
                    readable = {key: record[key] for key in fields if key in record}
                    metadata = record["sampling"]
                    if isinstance(metadata, list):
                        metadata = [
                            {
                                key: value
                                for key, value in turn.items()
                                if key != "sampled_token_ids"
                            }
                            for turn in metadata
                        ]
                    readable["sampling"] = metadata
                    readable["sampling_seed"] = seed
                    output.write(
                        json.dumps(readable, ensure_ascii=False, allow_nan=False) + "\n"
                    )
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination
