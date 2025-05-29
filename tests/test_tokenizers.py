"""Test cases for tokenizer loading."""

import pytest

from axolotl.loaders import load_tokenizer
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline


class TestTokenizers:
    """Test class for the load_tokenizer fn"""

    @enable_hf_offline
    def test_default_use_fast(self):
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
            }
        )
        tokenizer = load_tokenizer(cfg)
        assert "Fast" in tokenizer.__class__.__name__

    @enable_hf_offline
    def test_dont_use_fast(self):
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "tokenizer_use_fast": False,
            }
        )
        tokenizer = load_tokenizer(cfg)
        assert "Fast" not in tokenizer.__class__.__name__

    @enable_hf_offline
    def test_special_tokens_modules_to_save(self):
        # setting special_tokens to new token
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "adapter": "lora",
                "special_tokens": {"bos_token": "[INST]"},
            }
        )
        with pytest.raises(
            ValueError,
            match=r".*Please set lora_modules_to_save*",
        ):
            load_tokenizer(cfg)

        # setting special_tokens but not changing from default
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "adapter": "lora",
                "special_tokens": {"bos_token": "<s>"},
            }
        )
        load_tokenizer(cfg)

        # non-adapter setting special_tokens
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "special_tokens": {"bos_token": "[INST]"},
            }
        )
        load_tokenizer(cfg)

    @enable_hf_offline
    def test_add_additional_special_tokens(self):
        cfg = DictDefault(
            {
                "tokenizer_config": "huggyllama/llama-7b",
                "special_tokens": {"additional_special_tokens": ["<|im_start|>"]},
            }
        )
        tokenizer = load_tokenizer(cfg)
        assert tokenizer("<|im_start|>user")["input_ids"] == [1, 32000, 1404]
        assert len(tokenizer) == 32001

        # ensure reloading the tokenizer again from cfg results in same vocab length
        tokenizer = load_tokenizer(cfg)
        assert len(tokenizer) == 32001

    @enable_hf_offline
    def test_added_tokens_overrides(self, temp_dir):
        cfg = DictDefault(
            {
                # use with tokenizer that has reserved_tokens in added_tokens
                "tokenizer_config": "NousResearch/Llama-3.2-1B",
                "added_tokens_overrides": {
                    128041: "RANDOM_OVERRIDE_1",
                    128042: "RANDOM_OVERRIDE_2",
                },
                "output_dir": temp_dir,
            }
        )

        tokenizer = load_tokenizer(cfg)
        assert tokenizer.encode("RANDOM_OVERRIDE_1", add_special_tokens=False) == [
            128041
        ]
        assert tokenizer.encode("RANDOM_OVERRIDE_2", add_special_tokens=False) == [
            128042
        ]
        assert (
            tokenizer.decode([128041, 128042]) == "RANDOM_OVERRIDE_1RANDOM_OVERRIDE_2"
        )

    @enable_hf_offline
    def test_added_tokens_overrides_gemma3(self, temp_dir):
        cfg = DictDefault(
            {
                # use with tokenizer that has reserved_tokens in added_tokens
                "tokenizer_config": "mlx-community/gemma-3-4b-it-8bit",
                "added_tokens_overrides": {
                    256001: "RANDOM_OVERRIDE_1",
                    256002: "RANDOM_OVERRIDE_2",
                },
                "output_dir": temp_dir,
            }
        )

        tokenizer = load_tokenizer(cfg)
        assert tokenizer.encode("RANDOM_OVERRIDE_1", add_special_tokens=False) == [
            256001
        ]
        assert tokenizer.encode("RANDOM_OVERRIDE_2", add_special_tokens=False) == [
            256002
        ]
        assert (
            tokenizer.decode([256001, 256002]) == "RANDOM_OVERRIDE_1RANDOM_OVERRIDE_2"
        )

    @enable_hf_offline
    def test_added_tokens_overrides_with_toolargeid(self, temp_dir):
        cfg = DictDefault(
            {
                # use with tokenizer that has reserved_tokens in added_tokens
                "tokenizer_config": "HuggingFaceTB/SmolLM2-135M",
                "added_tokens_overrides": {1000000: "BROKEN_RANDOM_OVERRIDE_1"},
                "output_dir": temp_dir,
            }
        )

        with pytest.raises(
            ValueError, match=r".*Token ID 1000000 not found in added_tokens.*"
        ):
            load_tokenizer(cfg)

    def test_mistral_tokenizer_auto_detection(self):
        """Test that Mistral models are auto-detected and use MistralTokenizerWrapper"""
        cfg = DictDefault(
            {
                "base_model": "adamo1139/Mistral-Small-24B-Instruct-2501-ungated",
                "tokenizer_config": "adamo1139/Mistral-Small-24B-Instruct-2501-ungated",
            }
        )
        tokenizer = load_tokenizer(cfg)
        assert tokenizer.__class__.__name__ == "MistralTokenizerWrapper"

    def test_mixtral_tokenizer_auto_detection(self):
        """Test that Mixtral models are auto-detected and use MistralTokenizerWrapper"""
        cfg = DictDefault(
            {
                "base_model": "model-hub/Mixtral-8x7B-v0.1",
                "tokenizer_config": "model-hub/Mixtral-8x7B-v0.1",
            }
        )
        tokenizer = load_tokenizer(cfg)
        assert tokenizer.__class__.__name__ == "MistralTokenizerWrapper"

    def test_mistral_tokenizer_basic_functionality(self):
        """Test basic encode/decode functionality of MistralTokenizerWrapper"""
        cfg = DictDefault(
            {
                "base_model": "adamo1139/Mistral-Small-24B-Instruct-2501-ungated",
                "tokenizer_config": "adamo1139/Mistral-Small-24B-Instruct-2501-ungated",
            }
        )
        tokenizer = load_tokenizer(cfg)

        # Test basic encoding
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Test basic decoding
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)

        # Test token properties are accessible
        assert hasattr(tokenizer, "eos_token_id")
        assert hasattr(tokenizer, "bos_token_id")
        assert isinstance(tokenizer.eos_token_id, int)
        assert isinstance(tokenizer.bos_token_id, int)
