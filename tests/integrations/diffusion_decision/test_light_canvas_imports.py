"""The canvas path imports and runs without axolotl's training stack (serving in vLLM)."""

import json
import os
import subprocess  # nosec B404 - runs this interpreter on fixed test code
import sys
import textwrap
from pathlib import Path

import axolotl.integrations.diffusion_decision as package

SRC = Path(package.__file__).resolve().parents[3]

BLOCKER = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = {blocked!r}


    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if any(name == b or name.startswith(b + ".") for b in BLOCKED):
                raise ModuleNotFoundError(f"blocked import: {{name}}", name=name)
            return None


    sys.meta_path.insert(0, Block())
    """
)

TRAINING_STACK = (
    "peft",
    "accelerate",
    "datasets",
    "pandas",
    "addict",
    "multiprocess",
    "transformers.trainer",
    "axolotl.core",
)

COMMON = textwrap.dedent(
    """
    from axolotl.model_support import (
        DiffusionLayout,
        DiffusionNoise,
        DiffusionSpec,
        EosHandling,
        FirstPositionAlignment,
        GenerationAdapter,
        LogitAlignment,
        MaskTokenPolicy,
        ObjectiveReduction,
        ReductionScope,
        TimeWeighting,
    )

    SPEC = DiffusionSpec(
        noise=DiffusionNoise.ABSORBING,
        layout=DiffusionLayout.FULL_SEQUENCE,
        logit_alignment=LogitAlignment.ALIGNED,
        first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
        self_conditioning=False,
        max_canvas=128,
        max_context=1024,
        eos_handling=EosHandling.INDEPENDENT,
        mask_token_policy=MaskTokenPolicy.MODEL,
        default_time_weighting=TimeWeighting.NONE,
        objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
        generation_adapter=GenerationAdapter.FULL_SEQUENCE,
        reduction_scope=ReductionScope.MICROBATCH,
    )
    """
)


def _run(code: str, blocked: tuple[str, ...]) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), *filter(None, [env.get("PYTHONPATH")])]
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(  # nosec B603 - fixed arguments
        [sys.executable, "-c", BLOCKER.format(blocked=blocked) + code],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_canvas_builds_without_the_training_stack():
    code = COMMON + textwrap.dedent(
        """
        import json

        import axolotl.integrations.diffusion_decision as package
        from axolotl.integrations.diffusion_decision import canvas
        from axolotl.integrations.diffusion_decision.adapters import normalize_record
        from axolotl.integrations.diffusion_decision.grouping import group_records
        from axolotl.integrations.diffusion_decision.readers.base import (
            restricted_probabilities,
        )
        from axolotl.integrations.diffusion_decision.template import (
            SchemaError,
            codebook_alphabet,
        )


        class Tokenizer:
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 11
            unk_token_id = 2

            def encode(self, text, add_special_tokens=False):
                return [ord(character) % 256 for character in text]

            def apply_chat_template(self, messages, **kwargs):
                return [99, *self.encode(messages[-1]["content"])][:64]


        record = {
            "source": "light.test",
            "family": "f",
            "group": "g",
            "id": "r",
            "state": "The color is red.",
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "Which color is named?",
                    "options": [
                        {"name": "red", "description": "red"},
                        {"name": "blue", "description": "blue"},
                    ],
                },
                "sure": {
                    "type": "choice",
                    "instructions": "Is it certain?",
                    "options": ["yes", "no"],
                },
            },
            "labels": {
                "color": {"kind": "hard", "gold_idx": 0, "weight": 1.0},
                "sure": {"kind": "hard", "gold_idx": 1, "weight": 1.0},
            },
        }
        cfg = {
            "seed": 23,
            "sequence_len": 4096,
            "model_config": {"vocab_size": 256, "mask_token_id": 9},
            "diffusion_lm": {"canvas_width": 128, "mask_token_id": 9},
            "diffusion_decision": {"max_questions_per_canvas": 20},
        }
        codebook_alphabet("vendored26")
        normalized = [
            normalize_record(
                "diffusion_decision.jsonl",
                record,
                training=False,
                source_split="dev",
                codebook="vendored26",
            )
        ]
        groups = group_records(normalized, max_questions=20)
        rows = [
            canvas._canvas_row(Tokenizer(), dict(group), cfg, 1.0, spec=SPEC, vocab_size=256)
            for group in groups
        ]
        kept, drops = canvas._filter_budget_rows(rows, cfg, SPEC, is_eval=True)
        example = canvas._decision_example(rows[0]["canvas"], rows[0])
        built = rows[0]["canvas"]
        assert issubclass(SchemaError, Exception) and callable(restricted_probabilities)
        assert "DiffusionDecisionPlugin" in dir(package)
        print(
            json.dumps(
                {
                    "rows": len(rows),
                    "kept": len(kept),
                    "drops": drops,
                    "questions": list(built.question_ids),
                    "label_positions": list(built.label_positions),
                    "mask_at_labels": [built.canvas_ids[p] for p in built.label_positions],
                    "allowed": [list(q.allowed_token_ids) for q in example.questions],
                    "loaded": sorted(
                        name
                        for name in sys.modules
                        if any(name == b or name.startswith(b + ".") for b in BLOCKED)
                        or name == "axolotl.integrations.diffusion_decision.plugin"
                        or name == "axolotl.integrations.diffusion_decision.datasets"
                    ),
                }
            )
        )
        """
    )
    result = _run(code, TRAINING_STACK)
    assert result["loaded"] == []
    assert result["rows"] == result["kept"] == 1
    assert result["drops"] == {"logical": 0, "physical": 0}
    assert sorted(result["questions"]) == ["color", "sure"]  # permute_record reorders
    assert len(result["label_positions"]) == 2
    assert result["mask_at_labels"] == [9, 9]
    assert result["allowed"] == [[ord("A"), ord("B")]] * 2


def test_packed_budget_filter_needs_only_addict():
    code = COMMON + textwrap.dedent(
        """
        import json

        from axolotl.integrations.diffusion_decision import canvas

        cfg = {
            "sequence_len": 4096,
            "batch_flattening": True,
            "eval_batch_size": 2,
            "model_config_type": "nemotron_labs_diffusion",
            "diffusion_lm": {},
        }
        limits = canvas._budget_limits(cfg, is_eval=True)
        print(
            json.dumps(
                {
                    "limits": [limits.logical, limits.physical],
                    "loaded": sorted(
                        name
                        for name in sys.modules
                        if any(name == b or name.startswith(b + ".") for b in BLOCKED)
                    ),
                }
            )
        )
        """
    )
    blocked = tuple(name for name in TRAINING_STACK if name != "addict")
    assert _run(code, blocked) == {"limits": [4096, 8192], "loaded": []}


def test_blocker_rejects_the_training_plugin():
    code = textwrap.dedent(
        """
        import json

        try:
            from axolotl.integrations.diffusion_decision import DiffusionDecisionPlugin
        except ModuleNotFoundError as error:
            print(json.dumps({"blocked": error.name}))
        else:
            print(json.dumps({"blocked": None}))
        """
    )
    assert _run(code, TRAINING_STACK)["blocked"] in TRAINING_STACK


def test_segment_assembly_without_the_training_stack():
    code = COMMON + textwrap.dedent(
        """
        import json

        from axolotl.integrations.diffusion_decision import canvas, segments


        class Tokenizer:
            # One token per character, so any segmentation concatenates.
            decision_segments_safe = True
            pad_token_id = 0
            eos_token_id = 11

            def encode(self, text, add_special_tokens=False):
                return [ord(character) % 256 for character in text]

            def apply_chat_template(self, messages, *, tokenize, **kwargs):
                text = "<s>" + messages[0]["content"] + "|" + messages[1]["content"] + ">"
                return self.encode(text) if tokenize else text


        record = {
            "source": "light.test",
            "group": "g",
            "id": "r",
            "state": "The color is red.",
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "Which color?",
                    "options": ["red", "blue", "green"],
                },
                "warm": {"type": "choice", "instructions": "Warm?", "options": ["x", "y"]},
            },
            "labels": {
                "color": {"kind": "hard", "gold_idx": 0},
                "warm": {"kind": "hard", "gold_idx": 1},
            },
        }
        cfg = {
            "seed": 23,
            "model_config": {"vocab_size": 256, "mask_token_id": 9},
            "diffusion_lm": {"canvas_width": 128, "mask_token_id": 9},
            "diffusion_decision": {"max_questions_per_canvas": 20},
        }
        kwargs = dict(
            spec=SPEC, vocab_size=256, slot_plan=None, thought_open_ids=(), thought_close_ids=()
        )
        tokenizer = Tokenizer()
        built, reason = canvas._reserve_segments(tokenizer, record, cfg, **kwargs)
        same = []
        for draw in [(0, 1), (1, 0), (3, 2)]:
            full = canvas._canvas_row(tokenizer, dict(record), cfg, 1.0, train=True, draw=draw, **kwargs)
            fast = canvas._canvas_row(
                tokenizer, dict(record), cfg, 1.0, train=True, draw=draw, segments=built, **kwargs
            )
            same.append(full == fast)
        print(
            json.dumps(
                {
                    "reason": reason,
                    "same": same,
                    "assembled": segments.SEGMENT_STATS["assembled"],
                    "loaded": sorted(
                        name
                        for name in sys.modules
                        if any(name == b or name.startswith(b + ".") for b in BLOCKED)
                    ),
                }
            )
        )
        """
    )
    result = _run(code, TRAINING_STACK)
    assert result == {"reason": None, "same": [True] * 3, "assembled": 3, "loaded": []}


def test_light_image_tokens_match_processing_strategies():
    from axolotl import processing_strategies as ps
    from axolotl.integrations.diffusion_decision import _image_tokens as light

    for name in (
        "NEMOTRON_VLM_IMAGE_START_ID",
        "NEMOTRON_VLM_IMAGE_PAD_ID",
        "NEMOTRON_VLM_IMAGE_BREAK_ID",
        "NEMOTRON_VLM_IMAGE_END_ID",
        "NEMOTRON_VLM_IMAGE_START_MARKER",
        "NEMOTRON_VLM_TOKEN_SIDE",
    ):
        assert getattr(light, name) == getattr(ps, name), name
    for size in ((28, 28), (56, 84), (1400, 28)):
        assert light.merged_patch_count(
            size
        ) == ps.NemotronDiffusionVLMProcessingStrategy.merged_patch_count(size)
