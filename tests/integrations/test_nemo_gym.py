"""Unit tests for NeMo Gym integration.

Tests the reward, weight-sync and plugin wiring logic without requiring a running
NeMo Gym server or GPU. The multi-turn rollout loop is tested in
test_nemo_gym_rollout.py.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestRewardEnv(unittest.TestCase):
    """Tests for reward_env passthrough function."""

    def test_with_list_rewards(self):
        from axolotl.integrations.nemo_gym.rewards import reward_env

        result = reward_env([["comp1"], ["comp2"]], env_reward=[0.5, 0.8])
        assert result == [0.5, 0.8]

    def test_with_scalar_reward(self):
        from axolotl.integrations.nemo_gym.rewards import reward_env

        result = reward_env([["comp1"], ["comp2"]], env_reward=0.7)
        assert result == [0.7, 0.7]

    def test_missing_reward_returns_zeros(self):
        from axolotl.integrations.nemo_gym.rewards import reward_env

        result = reward_env([["comp1"], ["comp2"]])
        assert result == [0.0, 0.0]


class TestRewardNemoGymVerify(unittest.TestCase):
    """Tests for reward_nemo_gym_verify with mocked HTTP."""

    @patch("axolotl.integrations.nemo_gym.rewards._get_verify_urls")
    @patch("axolotl.integrations.nemo_gym.rewards.requests")
    def test_calls_verify_endpoint(self, mock_requests, mock_get_urls):
        from axolotl.integrations.nemo_gym.rewards import reward_nemo_gym_verify

        mock_get_urls.return_value = {"wordle": "http://localhost:9999/verify"}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"reward": 0.75}
        mock_requests.post.return_value = mock_resp

        result = reward_nemo_gym_verify(
            completions=[[{"role": "assistant", "content": "crane"}]],
            prompts=[[{"role": "user", "content": "Guess a word"}]],
            resources_server_ref=[{"name": "wordle"}],
            verify_extra=[{}],
        )

        assert result == [0.75]
        mock_requests.post.assert_called_once()

    @patch("axolotl.integrations.nemo_gym.rewards._get_verify_urls")
    def test_missing_server_returns_zero(self, mock_get_urls):
        from axolotl.integrations.nemo_gym.rewards import reward_nemo_gym_verify

        mock_get_urls.return_value = {}

        result = reward_nemo_gym_verify(
            completions=[[{"role": "assistant", "content": "crane"}]],
            prompts=[[{"role": "user", "content": "Guess"}]],
            resources_server_ref=[{"name": "unknown_server"}],
            verify_extra=[{}],
        )
        assert result == [0.0]


class TestNormalizeHost(unittest.TestCase):
    """Tests for server.py _normalize_host helper."""

    def test_zero_addr_normalized(self):
        from axolotl.integrations.nemo_gym.server import _normalize_host

        assert _normalize_host("0.0.0.0") == "127.0.0.1"

    def test_localhost_normalized(self):
        from axolotl.integrations.nemo_gym.server import _normalize_host

        assert _normalize_host("localhost") == "127.0.0.1"

    def test_loopback_passthrough(self):
        from axolotl.integrations.nemo_gym.server import _normalize_host

        assert _normalize_host("127.0.0.1") == "127.0.0.1"

    def test_custom_fallback(self):
        from axolotl.integrations.nemo_gym.server import _normalize_host

        assert _normalize_host("0.0.0.0", fallback="10.0.0.1") == "10.0.0.1"

    def test_real_ip_passthrough(self):
        from axolotl.integrations.nemo_gym.server import _normalize_host

        assert _normalize_host("192.168.1.50") == "192.168.1.50"


class TestSelectWeightSyncTransport(unittest.TestCase):
    """Pure-logic table tests for ``select_weight_sync_transport``."""

    def _caps(self, **kwargs):
        from axolotl.integrations.nemo_gym.plugin import VLLMWeightSyncCapabilities

        c = VLLMWeightSyncCapabilities(probed=True)
        for k, v in kwargs.items():
            setattr(c, k, v)
        return c

    def test_lora_with_native_endpoint(self):
        from axolotl.integrations.nemo_gym.plugin import select_weight_sync_transport

        caps = self._caps(lora_filesystem=True)
        assert (
            select_weight_sync_transport(caps, has_lora=True, vllm_lora_sync_pref=True)
            == "lora_filesystem"
        )

    def test_lora_pref_off_with_both_routes_uses_nccl(self):
        from axolotl.integrations.nemo_gym.plugin import select_weight_sync_transport

        caps = self._caps(lora_filesystem=True, nccl=True)
        assert (
            select_weight_sync_transport(caps, has_lora=True, vllm_lora_sync_pref=False)
            == "nccl"
        )
        assert (
            select_weight_sync_transport(caps, has_lora=True, vllm_lora_sync_pref=True)
            == "lora_filesystem"
        )

    def test_lora_falls_back_to_nccl_when_no_lora_endpoint(self):
        from axolotl.integrations.nemo_gym.plugin import select_weight_sync_transport

        caps = self._caps(nccl=True)
        assert (
            select_weight_sync_transport(caps, has_lora=True, vllm_lora_sync_pref=False)
            == "nccl"
        )

    def test_full_param_prefers_nccl(self):
        from axolotl.integrations.nemo_gym.plugin import select_weight_sync_transport

        caps = self._caps(nccl=True, lora_filesystem=True)
        assert (
            select_weight_sync_transport(
                caps, has_lora=False, vllm_lora_sync_pref=False
            )
            == "nccl"
        )

    def test_full_param_with_only_lora_endpoint_returns_none(self):
        from axolotl.integrations.nemo_gym.plugin import select_weight_sync_transport

        caps = self._caps(lora_filesystem=True)
        assert (
            select_weight_sync_transport(
                caps, has_lora=False, vllm_lora_sync_pref=False
            )
            == "none"
        )

    def test_full_param_no_routes_returns_none(self):
        from axolotl.integrations.nemo_gym.plugin import select_weight_sync_transport

        caps = self._caps()  # all False
        assert (
            select_weight_sync_transport(
                caps, has_lora=False, vllm_lora_sync_pref=False
            )
            == "none"
        )

    def test_lora_no_routes_returns_none(self):
        from axolotl.integrations.nemo_gym.plugin import select_weight_sync_transport

        caps = self._caps()
        assert (
            select_weight_sync_transport(caps, has_lora=True, vllm_lora_sync_pref=True)
            == "none"
        )


class TestProbeVllmWeightSync(unittest.TestCase):
    """``probe_vllm_weight_sync`` reads a vLLM ``/openapi.json`` and reports caps."""

    @staticmethod
    def _probe(paths):
        from unittest.mock import patch

        from axolotl.integrations.nemo_gym.plugin import probe_vllm_weight_sync

        spec = {"paths": {p: {"post": {}} for p in paths}}
        with patch("requests.get") as mock_get:
            mock_get.return_value.raise_for_status = lambda: None
            mock_get.return_value.json = lambda: spec
            return probe_vllm_weight_sync("http://localhost:8000")

    def test_stock_vllm_with_lora_enabled(self):
        """Stock ``vllm serve`` with runtime LoRA updating exposes only LoRA routes."""
        caps = self._probe(
            [
                "/v1/models",
                "/v1/load_lora_adapter",
                "/v1/unload_lora_adapter",
                "/v1/completions",
            ]
        )
        assert caps.probed is True
        assert caps.lora_filesystem is True
        assert caps.nccl is False

    def test_stock_vllm_with_nothing(self):
        caps = self._probe(["/v1/models", "/v1/completions"])
        assert caps.probed is True
        assert caps.lora_filesystem is False
        assert caps.nccl is False

    def test_native_dev_mode_server_full_capabilities(self):
        """``axolotl vllm-serve`` exposes native NCCL weight-transfer + LoRA routes."""
        caps = self._probe(
            [
                "/init_weight_transfer_engine",
                "/start_weight_update",
                "/update_weights",
                "/finish_weight_update",
                "/get_world_size",
                "/v1/load_lora_adapter",
            ]
        )
        assert caps.probed is True
        assert caps.nccl is True
        assert caps.lora_filesystem is True

    def test_dev_mode_without_lora(self):
        caps = self._probe(["/init_weight_transfer_engine", "/update_weights"])
        assert caps.nccl is True
        assert caps.lora_filesystem is False

    def test_legacy_axolotl_routes_not_recognised(self):
        caps = self._probe(["/init_communicator/", "/update_named_param/"])
        assert caps.nccl is False

    def test_unreachable_server_records_error(self):
        from unittest.mock import patch

        from axolotl.integrations.nemo_gym.plugin import probe_vllm_weight_sync

        with patch("requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Connection refused")
            caps = probe_vllm_weight_sync("http://localhost:9999")

        assert caps.probed is False
        assert caps.probe_error is not None
        assert "ConnectionError" in caps.probe_error
        assert caps.nccl is False
        assert caps.lora_filesystem is False


class TestPluginWeightSyncEnforcement(unittest.TestCase):
    """End-to-end test of post_trainer_create's transport-selection branch.

    The plugin used to silently no-op weight sync when ``vllm_lora_sync: false``,
    leaving the trainer learning in isolation while vLLM kept serving the
    unmodified base model. After the fix:

      - LoRA + LoRA-loading endpoint   → installs filesystem LoRA sync
      - LoRA + only NCCL endpoint      → uses NCCL broadcast
      - Full FT + NCCL endpoint        → uses NCCL broadcast (standard TRL flow)
      - No usable transport            → raises ValueError with a precise diagnosis
    """

    @staticmethod
    def _fake_cfg(adapter, vllm_lora_sync):
        class FakeTRL:
            pass

        class FakeCfg:
            pass

        trl = FakeTRL()
        trl.vllm_lora_sync = vllm_lora_sync
        trl.vllm_server_host = "127.0.0.1"
        trl.vllm_server_port = 8000

        cfg = FakeCfg()
        cfg.nemo_gym_enabled = True
        cfg.nemo_gym_model_name = None
        cfg.base_model = "test/model"
        cfg.nemo_gym_verify_timeout = 30
        cfg.nemo_gym_multi_turn = False
        cfg.adapter = adapter
        cfg.trl = trl
        return cfg

    @staticmethod
    def _fake_trainer():
        class FakeVLLMGen:
            sync_weights = staticmethod(lambda: None)

        class FakeTrainer:
            vllm_generation = FakeVLLMGen()

        return FakeTrainer()

    @staticmethod
    def _caps(**kwargs):
        from axolotl.integrations.nemo_gym.plugin import VLLMWeightSyncCapabilities

        c = VLLMWeightSyncCapabilities(probed=True)
        for k, v in kwargs.items():
            setattr(c, k, v)
        return c

    def test_lora_with_lora_endpoint_installs_filesystem_sync(self):
        from unittest.mock import patch

        from axolotl.integrations.nemo_gym.plugin import NemoGymPlugin

        plugin = NemoGymPlugin()
        plugin._vllm_caps = self._caps(lora_filesystem=True)
        cfg = self._fake_cfg(adapter="lora", vllm_lora_sync=True)
        trainer = self._fake_trainer()

        with (
            patch.object(plugin, "_setup_lora_sync") as setup,
            patch.object(plugin, "_check_lora_endpoint") as check,
        ):
            plugin.post_trainer_create(cfg, trainer)
            setup.assert_called_once()
            check.assert_called_once()

    def test_lora_with_no_routes_raises_with_lora_specific_message(self):
        from axolotl.integrations.nemo_gym.plugin import NemoGymPlugin

        plugin = NemoGymPlugin()
        plugin._vllm_caps = self._caps()  # all False, but probed
        cfg = self._fake_cfg(adapter="lora", vllm_lora_sync=False)
        trainer = self._fake_trainer()

        with self.assertRaises(ValueError) as ctx:
            plugin.post_trainer_create(cfg, trainer)
        msg = str(ctx.exception)
        assert "no-op trainer" in msg
        assert "load_lora_adapter" in msg
        assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" in msg

    def test_full_finetune_with_nccl_endpoint_uses_nccl(self):
        from axolotl.integrations.nemo_gym.plugin import NemoGymPlugin

        plugin = NemoGymPlugin()
        plugin._vllm_caps = self._caps(nccl=True)
        cfg = self._fake_cfg(adapter=None, vllm_lora_sync=False)
        trainer = self._fake_trainer()
        sync_weights = trainer.vllm_generation.sync_weights

        plugin.post_trainer_create(cfg, trainer)
        assert trainer.vllm_generation.sync_weights is sync_weights

    def test_full_finetune_with_no_routes_raises_with_full_param_message(self):
        from axolotl.integrations.nemo_gym.plugin import NemoGymPlugin

        plugin = NemoGymPlugin()
        plugin._vllm_caps = self._caps()
        cfg = self._fake_cfg(adapter=None, vllm_lora_sync=False)
        trainer = self._fake_trainer()
        with self.assertRaises(ValueError) as ctx:
            plugin.post_trainer_create(cfg, trainer)
        msg = str(ctx.exception)
        assert "no-op trainer" in msg
        assert "init_weight_transfer_engine" in msg
        assert "axolotl vllm-serve" in msg
        assert "VLLM_SERVER_DEV_MODE" in msg

    def test_unprobed_caps_raises_with_probe_failure_message(self):
        from axolotl.integrations.nemo_gym.plugin import NemoGymPlugin

        plugin = NemoGymPlugin()
        # Plugin._vllm_caps left as default-None: the post_trainer_create
        # branch falls back to a fresh VLLMWeightSyncCapabilities() with
        # probed=False, so the error path should mention probing.
        cfg = self._fake_cfg(adapter="lora", vllm_lora_sync=True)
        trainer = self._fake_trainer()
        with self.assertRaises(ValueError) as ctx:
            plugin.post_trainer_create(cfg, trainer)
        assert "could not probe" in str(ctx.exception)


class TestLoRASyncSetup(unittest.TestCase):
    """Tests for _setup_lora_sync delegation logic."""

    def test_delegates_to_async_trainer(self):
        """When trainer has _sync_lora_adapter, the closure should delegate."""
        from axolotl.integrations.nemo_gym.plugin import NemoGymPlugin

        plugin = NemoGymPlugin.__new__(NemoGymPlugin)

        trainer = MagicMock()
        trainer._sync_lora_adapter = MagicMock()
        trainer.vllm_generation = MagicMock()

        plugin._setup_lora_sync(trainer)

        # The closure should be installed
        trainer.vllm_generation.sync_weights()
        trainer._sync_lora_adapter.assert_called_once()

    def test_check_lora_endpoint_skips_non_main_rank(self):
        """_check_lora_endpoint should not crash when vllm_client is absent (rank 1)."""
        from axolotl.integrations.nemo_gym.plugin import NemoGymPlugin

        vllm_gen = MagicMock(spec=[])  # No attributes at all
        # Should not raise
        NemoGymPlugin._check_lora_endpoint(vllm_gen)


if __name__ == "__main__":
    unittest.main()
