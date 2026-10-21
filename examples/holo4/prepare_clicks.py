"""Convert one AgentNet clicks shard to Holo4's element-localization format."""

import json

from datasets import load_dataset
from pydantic import BaseModel, Field


class VisualLocalizerOutput(BaseModel):
    x: int = Field(ge=0, le=1000, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=1000, description="Y coordinate as integer in [0, 1000]")


PROMPT = (
    "Localize an element on the GUI image according to the provided target "
    "and output a click position.\n"
    f" * You must output a valid JSON following the format: {VisualLocalizerOutput.model_json_schema()}\n"
    " Your target is:\n"
)


def to_holo4(row):
    x, y = row["click_point"]
    point = {
        "x": round(x / row["image_width"] * 1000),
        "y": round(y / row["image_height"] * 1000),
    }
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT + row["action_text"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(point)}],
            },
        ]
    }


if __name__ == "__main__":
    ds = load_dataset(
        "mlfoundations-cua-dev/agentnet-clicks",
        data_files="data/train-00000-of-00162.parquet",
        split="train",
    )
    ds = ds.map(
        to_holo4,
        remove_columns=[c for c in ds.column_names if c not in ("messages", "images")],
    )
    ds.to_parquet("holo4-clicks.parquet")
