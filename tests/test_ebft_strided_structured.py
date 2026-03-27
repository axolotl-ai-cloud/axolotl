"""Tests for the EBFT strided structured dataset transform and data loading."""

import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from axolotl.prompt_strategies.ebft import load as load_ebft
from axolotl.utils.dict import DictDefault


@pytest.fixture
def tokenizer():
    """Create a simple word-level tokenizer — no network access needed."""
    # Build a tiny vocab covering common test words
    vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    words = (
        "what is 2 + the answer 4 hello world goodbye bye hi short prompt "
        "x write code print test some string metadata noise ok good python "
        "sampling abc 123 def solve return this that"
    ).split()
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()

    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    return tok


@pytest.fixture
def cfg():
    return DictDefault({"sequence_len": 64})


@pytest.fixture
def transform_fn_and_kwargs(cfg):
    result = load_ebft("ebft_strided_structured.transform", cfg)
    assert result is not None, "Failed to load ebft_strided_structured transform"
    transform_fn, map_kwargs = result
    return transform_fn, map_kwargs


class TestEBFTStridedStructuredTransform:
    """Tests for the dataset transform function itself."""

    def test_transform_loads(self, transform_fn_and_kwargs):
        transform_fn, map_kwargs = transform_fn_and_kwargs
        assert callable(transform_fn)
        assert "remove_columns" in map_kwargs

    def test_remove_columns_sentinel(self, transform_fn_and_kwargs):
        """Transform should signal removal of all original columns."""
        _, map_kwargs = transform_fn_and_kwargs
        assert map_kwargs["remove_columns"] == "__all__"

    def test_prompt_completion_tokenization(self, transform_fn_and_kwargs, tokenizer):
        """Prompt tokens get labels=-100, completion tokens get real labels."""
        transform_fn, _ = transform_fn_and_kwargs
        example = {"input": "what is 2 + 2", "output": "the answer is 4"}
        result = transform_fn(example, tokenizer=tokenizer)

        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert "prompt_length" in result

        prompt_length = result["prompt_length"]
        labels = result["labels"]
        seq_len = len(result["input_ids"])

        assert seq_len == 64, "Should be padded to sequence_len"
        assert len(labels) == seq_len
        assert prompt_length > 0

        # Prompt tokens should be masked
        for i in range(prompt_length):
            assert labels[i] == -100, f"Prompt token at {i} should be -100"

        # At least one completion token should have a real label
        completion_labels = [lab for lab in labels[prompt_length:] if lab != -100]
        assert len(completion_labels) > 0, "Should have non-masked completion tokens"

    def test_prompt_length_matches_boundary(self, transform_fn_and_kwargs, tokenizer):
        """prompt_length should be the exact boundary between -100 and real labels."""
        transform_fn, _ = transform_fn_and_kwargs
        example = {"input": "hello world", "output": "goodbye world"}
        result = transform_fn(example, tokenizer=tokenizer)

        prompt_length = result["prompt_length"]
        labels = result["labels"]

        assert labels[prompt_length - 1] == -100, "Last prompt token should be masked"
        assert labels[prompt_length] != -100, (
            "First completion token should not be masked"
        )

    def test_padding_tokens_masked(self, transform_fn_and_kwargs, tokenizer):
        """Padding tokens should have labels=-100 and attention_mask=0."""
        transform_fn, _ = transform_fn_and_kwargs
        example = {"input": "hi", "output": "bye"}
        result = transform_fn(example, tokenizer=tokenizer)

        attention_mask = result["attention_mask"]
        labels = result["labels"]

        real_len = sum(attention_mask)
        assert real_len < 64, "Short example should have padding"

        for i in range(real_len, 64):
            assert attention_mask[i] == 0, (
                f"Pad position {i} should have attention_mask=0"
            )
            assert labels[i] == -100, f"Pad position {i} should have labels=-100"

    def test_truncation_long_completion(self, transform_fn_and_kwargs, tokenizer):
        """Long completions should be truncated to fit sequence_len."""
        transform_fn, _ = transform_fn_and_kwargs
        example = {
            "input": "short prompt",
            "output": "x " * 500,
        }
        result = transform_fn(example, tokenizer=tokenizer)
        assert len(result["input_ids"]) == 64

    def test_alternative_field_names(self, transform_fn_and_kwargs, tokenizer):
        """Transform should handle different field name conventions."""
        transform_fn, _ = transform_fn_and_kwargs

        result = transform_fn(
            {"prompt": "what", "completion": "this"}, tokenizer=tokenizer
        )
        assert result["prompt_length"] > 0

        result = transform_fn(
            {"question": "what", "answer": "this"}, tokenizer=tokenizer
        )
        assert result["prompt_length"] > 0

    def test_without_tokenizer_returns_prompt(self, transform_fn_and_kwargs):
        """Without tokenizer, should return a prompt string."""
        transform_fn, _ = transform_fn_and_kwargs
        result = transform_fn({"input": "hello", "output": "world"})
        assert "prompt" in result
        assert result["prompt"] == "hello"


class TestEBFTColumnRemoval:
    """Tests for the __all__ column removal logic in the RL data path."""

    def _filter_remove_columns(self, map_kwargs, dataset):
        """Reproduce the filtering logic from rl.py _load_split."""
        if "remove_columns" in map_kwargs:
            ds_columns = dataset.column_names
            if map_kwargs["remove_columns"] == "__all__":
                map_kwargs["remove_columns"] = list(ds_columns)
            else:
                map_kwargs["remove_columns"] = [
                    c for c in map_kwargs["remove_columns"] if c in ds_columns
                ]
        return map_kwargs

    def test_all_original_columns_removed(self, transform_fn_and_kwargs, tokenizer):
        """After mapping, only tokenized columns should remain."""
        transform_fn, map_kwargs = transform_fn_and_kwargs
        map_kwargs = dict(map_kwargs)  # copy

        ds = Dataset.from_list(
            [
                {"input": "what is 2 + 2", "output": "4", "extra_field": "noise"},
            ]
        )

        map_kwargs = self._filter_remove_columns(map_kwargs, ds)
        assert "input" in map_kwargs["remove_columns"]
        assert "output" in map_kwargs["remove_columns"]
        assert "extra_field" in map_kwargs["remove_columns"]

        from functools import partial

        mapped = ds.map(partial(transform_fn, tokenizer=tokenizer), **map_kwargs)
        assert "input_ids" in mapped.column_names
        assert "labels" in mapped.column_names
        assert "prompt_length" in mapped.column_names
        assert "input" not in mapped.column_names
        assert "output" not in mapped.column_names
        assert "extra_field" not in mapped.column_names

    def test_extra_metadata_columns_removed(self, transform_fn_and_kwargs, tokenizer):
        """Datasets with many extra metadata columns should all be cleaned up."""
        transform_fn, map_kwargs = transform_fn_and_kwargs
        map_kwargs = dict(map_kwargs)

        ds = Dataset.from_list(
            [
                {
                    "input": "write hello world",
                    "output": "print hello",
                    "id": "abc 123",
                    "domain": "python",
                    "generation_algorithm": "sampling",
                    "llm_judgement": "good",
                    "unit_tests": "test",
                    "tests_execution_status": "ok",
                    "average_test_score": 0.95,
                },
            ]
        )

        map_kwargs = self._filter_remove_columns(map_kwargs, ds)
        assert len(map_kwargs["remove_columns"]) == 9

        from functools import partial

        mapped = ds.map(partial(transform_fn, tokenizer=tokenizer), **map_kwargs)

        expected_columns = {"input_ids", "attention_mask", "labels", "prompt_length"}
        assert set(mapped.column_names) == expected_columns

    def test_no_string_columns_remain(self, transform_fn_and_kwargs, tokenizer):
        """No string-typed columns should remain (would crash the DataLoader)."""
        transform_fn, map_kwargs = transform_fn_and_kwargs
        map_kwargs = dict(map_kwargs)

        ds = Dataset.from_list(
            [
                {"input": "test", "output": "test", "notes": "some string metadata"},
            ]
        )

        map_kwargs = self._filter_remove_columns(map_kwargs, ds)

        from functools import partial

        mapped = ds.map(partial(transform_fn, tokenizer=tokenizer), **map_kwargs)

        for col in mapped.column_names:
            feat = mapped.features[col]
            assert str(feat) != "string", (
                f"Column '{col}' is still a string — would crash DataLoader"
            )

    def test_filter_preserves_explicit_list(self):
        """When remove_columns is an explicit list, only existing columns are kept."""
        ds = Dataset.from_list([{"a": 1, "b": "text", "c": 3}])
        map_kwargs = {"remove_columns": ["a", "b", "missing_col"]}

        ds_columns = ds.column_names
        map_kwargs["remove_columns"] = [
            c for c in map_kwargs["remove_columns"] if c in ds_columns
        ]

        assert map_kwargs["remove_columns"] == ["a", "b"]
        assert "missing_col" not in map_kwargs["remove_columns"]


class TestMultiTurnSeparators:
    """Verify multi-turn transforms and trainer-side GT reconstruction."""

    def test_multiturn_transform_splits_turns(self):
        """Transform should store first turn as GT and remaining turns separately."""
        from axolotl.prompt_strategies.ebft import load as load_ebft
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault({"sequence_len": 512})
        fn, _ = load_ebft("ebft_chat_multiturn.transform", cfg)
        out = fn(
            {
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            }
        )
        # ground_truth is only the first assistant turn
        assert out["ground_truth"] == "A1"
        # remaining_turns carries the rest for trainer-side reconstruction
        assert out["remaining_turns"] == [
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]

    def test_multiturn_gt_reconstruction_via_chat_template(self):
        """Trainer-side GT reconstruction should insert role markers between turns.

        This tests the logic from trainer.py:284-299 that reconstructs multi-turn
        GT using apply_chat_template, ensuring assistant turns are separated by
        role markers rather than concatenated as raw text.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True
        )

        # Simulate the transform output
        prompt_msgs = [{"role": "user", "content": "Q1"}]
        gt = "A1"
        remaining_turns = [
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]

        # --- Reproduce the trainer-side reconstruction (trainer.py:284-299) ---
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        gt_conv = list(prompt_msgs) + [{"role": "assistant", "content": gt}]
        gt_conv.extend(remaining_turns)
        full_gt_text = tokenizer.apply_chat_template(
            gt_conv, tokenize=False, add_generation_prompt=False
        )

        # The full GT text should contain both assistant turns with role markers
        assert "A1" in full_gt_text
        assert "A2" in full_gt_text
        # Raw concatenation "A1A2" should NOT appear — role markers separate them
        assert "A1A2" not in full_gt_text, (
            "GT reconstruction should have role markers between turns, not raw concatenation"
        )
        # The user turn Q2 should appear between A1 and A2
        a1_pos = full_gt_text.index("A1")
        a2_pos = full_gt_text.index("A2")
        q2_pos = full_gt_text.index("Q2")
        assert a1_pos < q2_pos < a2_pos, (
            "Turn order should be A1 -> Q2 -> A2 in rendered GT"
        )
        # The GT should start with the prompt
        assert full_gt_text.startswith(prompt_text), (
            "Full GT should start with the rendered prompt"
        )

    def test_multiturn_gt_reconstruction_fallback_single_turn(self):
        """Single-turn prompts in a multi-turn dataset should use raw concatenation."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True
        )

        prompt_msgs = [{"role": "user", "content": "Q1"}]
        gt = "A1"
        # remaining_turns would be [] for single-turn prompts

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        # With empty remaining_turns, trainer falls through to raw concat
        # (trainer.py:302: gt_texts.append(prompt_text + gt))
        gt_text = prompt_text + gt
        assert gt_text.endswith("A1")
        assert prompt_text in gt_text
