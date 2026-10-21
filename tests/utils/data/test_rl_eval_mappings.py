"""Preference evaluation mappings must follow the source dataset, not its group index."""

import pytest
from datasets import Dataset

from axolotl.utils.data.evaluation import load_eval_datasets
from axolotl.utils.data.rl import _load_or_create_dataset_split, _load_split
from axolotl.utils.data.shared import generate_dataset_hash_from_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.enums import RLType


class MappingTokenizer:
    name_or_path = "local-mapping-test"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1]}

    def apply_chat_template(self, messages, **kwargs):
        return "".join(message["content"] for message in messages)


@pytest.mark.parametrize("strategy", ["dpo", "kto", "chat_template"])
@pytest.mark.parametrize("expand_names", [False, True])
def test_eval_groups_use_their_own_mappings(monkeypatch, strategy, expand_names):
    from axolotl.prompt_strategies.dpo import chat_template
    from axolotl.utils.data import rl

    def entry(name, group):
        fields = {
            "field_prompt": f"prompt_{name}",
            "field_chosen": f"chosen_{name}",
            "field_rejected": f"rejected_{name}",
            "field_completion": f"chosen_{name}",
            "field_label": f"label_{name}",
        }
        return {
            "path": name,
            "eval_name": group,
            "type": "chat_template" if strategy == "chat_template" else fields,
            "field_messages": f"prompt_{name}",
            "field_chosen": f"chosen_{name}",
            "field_rejected": f"rejected_{name}",
        }

    cfg = DictDefault(
        {
            "rl": RLType.KTO if strategy == "kto" else RLType.DPO,
            "datasets": [entry("a", None)],
            "test_datasets": [
                entry("a", "shared"),
                entry("b", "solo"),
                entry("c", "shared"),
            ],
            "skip_prepare_dataset": True,
        }
    )
    if expand_names:
        cfg.test_datasets[0].name = ["one", "two"]
    tokenizer = MappingTokenizer()

    def raw_dataset(ds, *args, **kwargs):
        row = {}
        for name in "abc":
            row.update(
                {
                    f"prompt_{name}": f"prompt-{ds.path}-{name}",
                    f"chosen_{name}": f"chosen-{ds.path}-{name}",
                    f"rejected_{name}": f"rejected-{ds.path}-{name}",
                    f"label_{name}": name != "b",
                }
            )
        return Dataset.from_list([row])

    monkeypatch.setattr(rl, "load_dataset_with_config", raw_dataset)
    monkeypatch.setattr(rl, "load_tokenizer", lambda cfg: tokenizer)
    monkeypatch.setattr(chat_template, "get_chat_template", lambda **kwargs: "test")
    grouped = load_eval_datasets(cfg, lambda c: _load_split(c, "test"))
    names = ["a", "a", "b", "c"] if expand_names else ["a", "b", "c"]
    expected = [f"prompt-{name}-{name}" for name in names]
    assert grouped[""]["prompt"] == expected
    assert grouped["shared"]["prompt"] == [
        value for value in expected if value != "prompt-b-b"
    ]
    assert grouped["solo"]["prompt"] == ["prompt-b-b"]
    completion = "completion" if strategy == "kto" else "chosen"
    assert grouped["solo"][completion] == ["chosen-b-b"]
    if strategy == "kto":
        assert grouped["solo"]["label"] == [False]
    else:
        assert grouped["solo"]["rejected"] == ["rejected-b-b"]
    assert _load_split(cfg, "test")["prompt"] == expected
    assert _load_split(cfg, "train")["prompt"] == ["prompt-a-a"]
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].path == "a"


def test_preference_cache_invalidates_old_transformations(monkeypatch):
    from axolotl.utils.data import rl

    cfg = DictDefault(
        {
            "rl": RLType.DPO,
            "datasets": [{"path": "training"}],
            "test_datasets": [{"path": "evaluation"}],
            "sequence_len": 128,
        }
    )
    tokenizer = MappingTokenizer()
    stale_hash = generate_dataset_hash_from_config(
        cfg, cfg.test_datasets, tokenizer.name_or_path
    )
    stale = Dataset.from_dict({"prompt": ["wrong mapping"]})
    fresh = Dataset.from_dict({"prompt": ["p"], "chosen": ["c"], "rejected": ["r"]})
    cache = {stale_hash: stale}
    reads = []
    saves = []

    def read(cfg, key):
        reads.append(key)
        return cache.get(key)

    def save(cfg, dataset, key, split):
        saves.append(key)
        cache[key] = dataset

    monkeypatch.setattr(rl, "load_preprocessed_dataset", read)
    monkeypatch.setattr(rl, "save_preprocessed_dataset", save)
    monkeypatch.setattr(rl, "load_dataset_with_config", lambda *args, **kwargs: fresh)
    monkeypatch.setattr(rl, "load_tokenizer", lambda cfg: tokenizer)
    first = _load_or_create_dataset_split(cfg, tokenizer, "test")
    monkeypatch.setattr(
        rl, "_load_split", lambda *args, **kwargs: pytest.fail("cache miss")
    )
    second = _load_or_create_dataset_split(cfg, tokenizer, "test")
    assert first.to_dict() == second.to_dict() == fresh.to_dict()
    assert reads == [saves[0], saves[0]]
    assert stale_hash != saves[0]
