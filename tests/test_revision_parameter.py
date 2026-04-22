"""Tests for revision_of_model being passed to tokenizer and processor loaders."""

from unittest.mock import MagicMock, patch

from transformers import PreTrainedTokenizerBase

from axolotl.utils.dict import DictDefault


class TestRevisionParameter:
    """Tests for revision_of_model being passed to tokenizer and processor loaders."""

    @patch("axolotl.loaders.tokenizer.load_model_config")
    @patch("axolotl.loaders.tokenizer.AutoTokenizer")
    @patch(
        "axolotl.loaders.patch_manager.PatchManager.apply_pre_tokenizer_load_patches"
    )
    def test_load_tokenizer_passes_revision(
        self, _mock_patches, mock_auto_tokenizer, _mock_load_config
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.__class__.__name__ = "MockTokenizer"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = DictDefault(
            {
                "tokenizer_config": "some-model",
                "revision_of_model": "abc123",
            }
        )
        from axolotl.loaders.tokenizer import load_tokenizer

        load_tokenizer(cfg)

        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args
        assert call_kwargs.kwargs.get("revision") == "abc123"

    @patch("axolotl.loaders.tokenizer.load_model_config")
    @patch("axolotl.loaders.tokenizer.AutoTokenizer")
    @patch(
        "axolotl.loaders.patch_manager.PatchManager.apply_pre_tokenizer_load_patches"
    )
    def test_load_tokenizer_omits_revision_when_unset(
        self, _mock_patches, mock_auto_tokenizer, _mock_load_config
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.__class__.__name__ = "MockTokenizer"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = DictDefault(
            {
                "tokenizer_config": "some-model",
            }
        )
        from axolotl.loaders.tokenizer import load_tokenizer

        load_tokenizer(cfg)

        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args
        assert "revision" not in call_kwargs.kwargs

    @patch("axolotl.loaders.tokenizer.AutoTokenizer")
    @patch("axolotl.loaders.tokenizer.is_local_main_process", return_value=True)
    @patch("axolotl.loaders.tokenizer.barrier")
    def test_modify_tokenizer_files_passes_revision(
        self, _mock_barrier, _mock_main, mock_auto_tokenizer, temp_dir
    ):
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        from axolotl.loaders.tokenizer import modify_tokenizer_files

        modify_tokenizer_files("some-model", {}, output_dir=temp_dir, revision="abc123")

        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args
        assert call_kwargs.kwargs.get("revision") == "abc123"

    @patch("axolotl.loaders.tokenizer.AutoTokenizer")
    @patch("axolotl.loaders.tokenizer.is_local_main_process", return_value=True)
    @patch("axolotl.loaders.tokenizer.barrier")
    def test_modify_tokenizer_files_defaults_revision_to_main(
        self, _mock_barrier, _mock_main, mock_auto_tokenizer, temp_dir
    ):
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        from axolotl.loaders.tokenizer import modify_tokenizer_files

        modify_tokenizer_files("some-model", {}, output_dir=temp_dir)

        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args
        assert call_kwargs.kwargs.get("revision") == "main"

    @patch("axolotl.loaders.processor.AutoProcessor")
    def test_load_processor_passes_revision(self, mock_auto_processor):
        mock_processor = MagicMock()
        mock_processor.size = {}
        mock_auto_processor.from_pretrained.return_value = mock_processor

        cfg = DictDefault(
            {
                "processor_config": "some-model",
                "revision_of_model": "abc123",
                "trust_remote_code": False,
            }
        )
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

        from axolotl.loaders.processor import load_processor

        load_processor(cfg, tokenizer)

        call_kwargs = mock_auto_processor.from_pretrained.call_args
        assert call_kwargs.kwargs.get("revision") == "abc123"

    @patch("axolotl.loaders.processor.AutoProcessor")
    def test_load_processor_omits_revision_when_unset(self, mock_auto_processor):
        mock_processor = MagicMock()
        mock_processor.size = {}
        mock_auto_processor.from_pretrained.return_value = mock_processor

        cfg = DictDefault(
            {
                "processor_config": "some-model",
                "trust_remote_code": False,
            }
        )
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

        from axolotl.loaders.processor import load_processor

        load_processor(cfg, tokenizer)

        call_kwargs = mock_auto_processor.from_pretrained.call_args
        assert "revision" not in call_kwargs.kwargs

    @patch("axolotl.loaders.processor.AutoProcessor")
    def test_load_processor_forwards_processor_kwargs(self, mock_auto_processor):
        mock_processor = MagicMock()
        mock_processor.size = {}
        mock_auto_processor.from_pretrained.return_value = mock_processor

        cfg = DictDefault(
            {
                "processor_config": "some-model",
                "trust_remote_code": False,
                "processor_kwargs": {
                    "image_seq_length": 1120,
                    "max_soft_tokens": 1120,
                },
            }
        )
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

        from axolotl.loaders.processor import load_processor

        load_processor(cfg, tokenizer)

        call_kwargs = mock_auto_processor.from_pretrained.call_args
        assert call_kwargs.kwargs.get("image_seq_length") == 1120
        assert call_kwargs.kwargs.get("max_soft_tokens") == 1120

    @patch("axolotl.loaders.processor.AutoProcessor")
    def test_load_processor_omits_processor_kwargs_when_unset(
        self, mock_auto_processor
    ):
        mock_processor = MagicMock()
        mock_processor.size = {}
        mock_auto_processor.from_pretrained.return_value = mock_processor

        cfg = DictDefault(
            {
                "processor_config": "some-model",
                "trust_remote_code": False,
            }
        )
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

        from axolotl.loaders.processor import load_processor

        load_processor(cfg, tokenizer)

        call_kwargs = mock_auto_processor.from_pretrained.call_args
        assert "image_seq_length" not in call_kwargs.kwargs
        assert "max_soft_tokens" not in call_kwargs.kwargs

    def test_processor_kwargs_schema_rejects_revision(self):
        import pytest

        from axolotl.utils.schemas.model import ModelInputConfig

        with pytest.raises(ValueError, match="revision"):
            ModelInputConfig(
                base_model="some-model",
                processor_kwargs={"revision": "abc123"},
            )

    def test_processor_kwargs_schema_rejects_trust_remote_code(self):
        import pytest

        from axolotl.utils.schemas.model import ModelInputConfig

        with pytest.raises(ValueError, match="trust_remote_code"):
            ModelInputConfig(
                base_model="some-model",
                processor_kwargs={"trust_remote_code": True},
            )

    def test_processor_kwargs_schema_accepts_valid_keys(self):
        from axolotl.utils.schemas.model import ModelInputConfig

        cfg = ModelInputConfig(
            base_model="some-model",
            processor_kwargs={"image_seq_length": 1120, "max_soft_tokens": 1120},
        )
        assert cfg.processor_kwargs == {
            "image_seq_length": 1120,
            "max_soft_tokens": 1120,
        }

    def test_processor_kwargs_schema_accepts_none_and_empty(self):
        from axolotl.utils.schemas.model import ModelInputConfig

        assert ModelInputConfig(base_model="x").processor_kwargs is None
        assert (
            ModelInputConfig(base_model="x", processor_kwargs={}).processor_kwargs == {}
        )
