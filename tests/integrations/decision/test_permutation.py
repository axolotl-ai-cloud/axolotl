"""Option permutations preserve the meanings of all typed targets."""

from copy import deepcopy

import pytest

from axolotl.integrations.decision.permutation import permute_record


def record(target):
    return {
        "source": "fixture",
        "group": "group",
        "id": "row",
        "state": {"seen": "objects"},
        "questions": {
            "choice": {
                "type": "choice",
                "options": [
                    {"name": name, "description": f"description of {name}"}
                    for name in ("apple", "pear", "plum", "orange")
                ],
            },
            "binary": {"type": "noul"},
        },
        "labels": {"choice": target, "binary": {"kind": "hard", "gold_idx": 1}},
    }


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "hard", "gold_idx": 2},
        {"kind": "dist", "probs": [0.1, 0.2, 0.3, 0.4]},
        {
            "kind": "dist",
            "candidate_indices": [0, 2],
            "candidate_ids": ["apple", "plum"],
            "probs": [0.1, 0.6],
            "other_probability": 0.3,
        },
        {"kind": "set", "allowed_set": [0, 2]},
    ],
)
def test_semantic_targets_preserved_for_every_seed(target):
    original = record(target)
    untouched = deepcopy(original)
    orders = set()
    question_orders = set()
    for seed in range(20):
        result = permute_record(original, seed=seed)
        assert result == permute_record(original, seed=seed)
        options = result["questions"]["choice"]["options"]
        names = [option["name"] for option in options]
        assert all(
            option["description"] == f"description of {option['name']}"
            for option in options
        )
        remapped = result["labels"]["choice"]
        if target["kind"] == "hard":
            assert names[remapped["gold_idx"]] == "plum"
        elif target["kind"] == "dist":
            if "candidate_indices" in target:
                assert {
                    names[index]: probability
                    for index, probability in zip(
                        remapped["candidate_indices"], remapped["probs"], strict=True
                    )
                } == {"apple": 0.1, "plum": 0.6}
                assert remapped["other_probability"] == 0.3
                assert {
                    names[index]: identity
                    for index, identity in zip(
                        remapped["candidate_indices"],
                        remapped["candidate_ids"],
                        strict=True,
                    )
                } == {"apple": "apple", "plum": "plum"}
            else:
                assert dict(zip(names, remapped["probs"], strict=True)) == {
                    "apple": 0.1,
                    "pear": 0.2,
                    "plum": 0.3,
                    "orange": 0.4,
                }
        else:
            assert {names[index] for index in remapped["allowed_set"]} == {
                "apple",
                "plum",
            }
        assert result["labels"]["binary"] == untouched["labels"]["binary"]
        assert result["state"] == untouched["state"]
        orders.add(tuple(names))
        question_orders.add(tuple(result["questions"]))
    assert original == untouched
    assert len(orders) > 1
    assert len(question_orders) == 2


def test_question_input_order_does_not_change_seeded_result():
    first = record({"kind": "hard", "gold_idx": 0})
    second = deepcopy(first)
    second["questions"] = dict(reversed(list(second["questions"].items())))
    assert permute_record(first, seed=5) == permute_record(second, seed=5)


def test_expanded_codebook_is_preserved_during_permutation():
    names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZab")
    original = {
        "source": "fixture",
        "group": "group",
        "id": "expanded-choice",
        "state": {"seen": "objects"},
        "questions": {
            "choice": {
                "type": "choice",
                "options": [
                    {"name": name, "description": f"description of {name}"}
                    for name in names
                ],
            }
        },
        "labels": {"choice": {"kind": "dist", "probs": [1 / 28] * 28}},
    }

    result = permute_record(original, seed=7, codebook="expanded52")

    assert len(result["questions"]["choice"]["options"]) == 28
    assert result["labels"]["choice"]["probs"] == [1 / 28] * 28
