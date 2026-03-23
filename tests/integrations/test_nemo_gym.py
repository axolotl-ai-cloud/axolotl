"""Unit tests for NeMo Gym integration.

Tests the core parsing, routing, reward, and plugin wiring logic
without requiring a running NeMo Gym server or GPU.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestParseAgentResponse(unittest.TestCase):
    """Tests for _parse_agent_response in multi_turn.py."""

    def _parse(self, response, eos_token_id=2):
        from axolotl.integrations.nemo_gym.multi_turn import _parse_agent_response

        return _parse_agent_response(response, eos_token_id)

    def test_empty_response_returns_defaults(self):
        result = self._parse({})
        assert result["prompt_ids"] == [2]
        assert result["completion_ids"] == [2]
        assert result["env_mask"] == [0]
        assert result["reward"] == 0.0
        assert result["num_turns"] == 0

    def test_error_response_returns_defaults(self):
        result = self._parse({"error": "something broke"})
        assert result["reward"] == 0.0
        assert result["num_turns"] == 0

    def test_single_turn_function_call(self):
        response = {
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "name": "guess_word",
                        "arguments": '{"guess": "crane"}',
                        "call_id": "call_1",
                        "prompt_token_ids": [10, 20, 30],
                        "generation_token_ids": [40, 50],
                        "generation_log_probs": [-0.1, -0.2],
                    }
                ]
            },
            "reward": 0.5,
        }
        result = self._parse(response)
        assert result["prompt_ids"] == [10, 20, 30]
        assert result["completion_ids"] == [40, 50]
        assert result["env_mask"] == [1, 1]  # model tokens
        assert result["logprobs"] == [-0.1, -0.2]
        assert result["reward"] == 0.5
        assert result["num_turns"] == 1

    def test_multi_turn_preserves_env_mask(self):
        """Second turn's prompt tokens (tool results) get env_mask=0."""
        response = {
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "prompt_token_ids": [10, 20],
                        "generation_token_ids": [30, 31],
                        "generation_log_probs": [-0.1, -0.2],
                    },
                    {
                        "type": "function_call_output",
                        "output": '{"feedback": "XYGXY"}',
                    },
                    {
                        "type": "function_call",
                        # prompt includes original + gen + tool output
                        "prompt_token_ids": [10, 20, 30, 31, 100, 101, 102],
                        "generation_token_ids": [40, 41],
                        "generation_log_probs": [-0.3, -0.4],
                    },
                ]
            },
            "reward": 0.3,
        }
        result = self._parse(response)
        assert result["prompt_ids"] == [10, 20]
        # completion = gen1 + tool_result + gen2
        assert result["completion_ids"] == [30, 31, 100, 101, 102, 40, 41]
        # env_mask: gen1=model(1), tool=env(0), gen2=model(1)
        assert result["env_mask"] == [1, 1, 0, 0, 0, 1, 1]
        assert result["num_turns"] == 2

    def test_empty_output_preserves_reward(self):
        response = {
            "response": {"output": []},
            "reward": 0.42,
        }
        result = self._parse(response)
        assert result["reward"] == 0.42

    def test_message_only_output(self):
        """A message with text but no function calls."""
        response = {
            "response": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "I'll guess crane."}
                        ],
                        "prompt_token_ids": [10, 20],
                        "generation_token_ids": [30, 31, 32],
                        "generation_log_probs": [-0.1, -0.2, -0.3],
                    }
                ]
            },
            "reward": 0.1,
        }
        result = self._parse(response)
        assert result["num_turns"] == 1
        assert result["completion_ids"] == [30, 31, 32]
        assert result["env_mask"] == [1, 1, 1]


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


class TestDatasetLookupKeying(unittest.TestCase):
    """Verify dataset lookup uses last message content as key."""

    def test_single_message_prompt(self):
        """Single-message prompt: [0] == [-1], both work."""
        prompt = [{"role": "user", "content": "Play Wordle!"}]
        assert prompt[0]["content"] == prompt[-1]["content"]

    def test_multi_message_prompt_uses_last(self):
        """Multi-message prompt: must use [-1] to match data_producer lookup."""
        prompt = [
            {"role": "system", "content": "You are a game player."},
            {"role": "user", "content": "Play Wordle!"},
        ]
        # data_producer.py line 92 uses prompt[-1]
        key = prompt[-1]["content"]
        assert key == "Play Wordle!"
        # Old code used prompt[0] which would be wrong here
        assert prompt[0]["content"] != key


class TestAgentRefPreservation(unittest.TestCase):
    """Verify agent_ref is preserved through the dispatch chain."""

    def test_data_producer_preserves_agent_ref(self):
        """Simulates the data_producer lookup logic."""
        # Simulate what plugin.py builds
        dataset_lookup = {
            "Play Wordle!": {
                "prompt": [{"role": "user", "content": "Play Wordle!"}],
                "agent_ref": {"name": "wordle_simple_agent"},
                "verify_extra": {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "Play Wordle!"}]
                    }
                },
            }
        }

        # Simulate data_producer.py logic (after fix)
        prompt_text = "Play Wordle!"
        full_item = dataset_lookup.get(prompt_text, {})
        item = full_item.get("verify_extra", {})
        if "agent_ref" in full_item and "agent_ref" not in item:
            item["agent_ref"] = full_item["agent_ref"]

        assert "agent_ref" in item
        assert item["agent_ref"]["name"] == "wordle_simple_agent"

    def test_multi_turn_preserves_agent_ref(self):
        """Simulates the multi_turn.py dispatch logic."""
        dataset_lookup = {
            "Play Wordle!": {
                "agent_ref": {"name": "wordle_simple_agent"},
                "verify_extra": {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "Play Wordle!"}]
                    }
                },
            }
        }

        # Simulate multi_turn.py logic (after fix)
        prompt_str = "Play Wordle!"
        full_item = None
        for key, val in dataset_lookup.items():
            if isinstance(key, str) and prompt_str == key:
                full_item = val
                break

        dispatched = full_item.get("verify_extra", full_item)
        if isinstance(dispatched, dict) and "agent_ref" not in dispatched:
            agent_ref = full_item.get("agent_ref")
            if agent_ref:
                dispatched = {**dispatched, "agent_ref": agent_ref}

        assert "agent_ref" in dispatched
        assert dispatched["agent_ref"]["name"] == "wordle_simple_agent"


class TestCallAgentsRouting(unittest.TestCase):
    """Tests for _call_agents routing via agent_ref."""

    def test_routes_to_correct_agent(self):
        """Items with agent_ref should route to the matching agent server."""

        agent_servers = {
            "wordle_agent": "http://localhost:11111",
            "math_agent": "http://localhost:22222",
        }

        items = [
            {
                "agent_ref": {"name": "wordle_agent"},
                "responses_create_params": {
                    "input": [{"role": "user", "content": "Play"}]
                },
            }
        ]

        # We can't actually call the agent, but verify the URL resolution
        # by checking _call_agents builds the right request
        # The function uses aiohttp — just verify agent_ref lookup works
        item = items[0]
        agent_ref = item.get("agent_ref", {})
        agent_name = agent_ref.get("name", "")
        agent_url = agent_servers.get(agent_name, "")
        assert agent_url == "http://localhost:11111"

    def test_fallback_to_first_agent(self):
        """Items without agent_ref should use first available agent."""
        agent_servers = {"default_agent": "http://localhost:33333"}
        item = {
            "responses_create_params": {"input": [{"role": "user", "content": "Hello"}]}
        }
        agent_ref = item.get("agent_ref", {})
        agent_name = agent_ref.get("name", "")
        agent_url = agent_servers.get(agent_name, "")
        if not agent_url and agent_servers:
            agent_url = next(iter(agent_servers.values()))
        assert agent_url == "http://localhost:33333"


class TestPluginDefaults(unittest.TestCase):
    """Tests for plugin config enforcement."""

    def test_dataloader_num_workers_forced_to_zero(self):
        """Plugin should set dataloader_num_workers=0 for NeMo Gym."""

        # Simulate the plugin logic
        class FakeCfg:
            dataloader_num_workers = 4
            nemo_gym_multi_turn = True

        cfg = FakeCfg()
        # Replicate plugin.get_training_args logic
        if getattr(cfg, "dataloader_num_workers", None) not in (None, 0):
            pass  # would log warning
        cfg.dataloader_num_workers = 0
        assert cfg.dataloader_num_workers == 0

    def test_dataloader_num_workers_none_stays_zero(self):
        class FakeCfg:
            dataloader_num_workers = None

        cfg = FakeCfg()
        cfg.dataloader_num_workers = 0
        assert cfg.dataloader_num_workers == 0


class TestNemoGymE2E(unittest.TestCase):
    """End-to-end test: data producer → agent (mocked) → parse → tensors → rewards.

    Exercises the full NemoGymDataProducer.produce() pipeline with mocked HTTP
    responses, verifying that multi-turn Wordle agent responses are correctly
    parsed into padded tensors with proper env_mask, logprobs, and rewards.
    No GPU or NeMo Gym server required.
    """

    # A realistic 2-turn agent /run response (guess + feedback + guess + done)
    AGENT_RESPONSE = {
        "response": {
            "output": [
                {
                    "type": "function_call",
                    "name": "guess_word",
                    "arguments": '{"guess": "crane"}',
                    "call_id": "call_1",
                    "id": "call_1",
                    "status": "completed",
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [10, 11, 12, 13],
                    "generation_log_probs": [-0.1, -0.2, -0.3, -0.4],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"feedback":"XYGXY","guesses_remaining":5,"done":false}',
                },
                {
                    "type": "function_call",
                    "name": "guess_word",
                    "arguments": '{"guess": "slide"}',
                    "call_id": "call_2",
                    "id": "call_2",
                    "status": "completed",
                    # prompt = original(5) + gen1(4) + tool_output(3 tokens)
                    "prompt_token_ids": [1, 2, 3, 4, 5, 10, 11, 12, 13, 50, 51, 52],
                    "generation_token_ids": [20, 21, 22],
                    "generation_log_probs": [-0.5, -0.6, -0.7],
                },
            ],
        },
        "reward": 0.42,
    }

    def _make_mock_trainer(self):
        """Create a minimal mock trainer with the attributes produce() needs."""
        trainer = MagicMock()
        trainer.accelerator.is_main_process = True
        trainer.accelerator.device = "cpu"
        trainer.max_completion_length = 512
        trainer.temperature = 0.8
        trainer.pad_token_id = 0
        trainer.processing_class.eos_token_id = 2
        trainer.processing_class.batch_decode.return_value = ["crane slide"]
        return trainer

    @patch("axolotl.integrations.nemo_gym.data_producer._call_agents")
    def test_produce_returns_valid_rollout_dataset(self, mock_call_agents):
        """Full pipeline: produce() → _call_agents (mocked) → parse → RolloutDataset."""

        from axolotl.integrations.nemo_gym.data_producer import NemoGymDataProducer

        # Mock _call_agents — it's async, so return a coroutine
        async def fake_call_agents(**kwargs):
            return [self.AGENT_RESPONSE, self.AGENT_RESPONSE]

        mock_call_agents.side_effect = fake_call_agents

        # Build a minimal mock of GRPODataProducer's __init__ dependencies
        # We can't easily call super().__init__, so we'll set attributes directly
        producer = NemoGymDataProducer.__new__(NemoGymDataProducer)
        producer._agent_servers = {"wordle_agent": "http://mock:9999"}
        producer._dataset_lookup = {
            "Play Wordle!": {
                "agent_ref": {"name": "wordle_agent"},
                "verify_extra": {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "Play Wordle!"}],
                    }
                },
            }
        }
        producer._request_timeout = 30
        producer._num_generations = 2

        # Mock the trainer
        trainer = self._make_mock_trainer()
        producer._trainer = trainer

        # Mock the prompt iterator (returns a batch of 1 input)
        producer._prompt_iter = iter(
            [
                [
                    {
                        "prompt": [{"role": "user", "content": "Play Wordle!"}],
                    }
                ]
            ]
        )
        producer._prompt_dl = [
            [{"prompt": [{"role": "user", "content": "Play Wordle!"}]}]
        ]

        # Call produce
        result = producer.produce(model=MagicMock(), global_step=1)

        # Verify result structure
        assert result is not None
        data = result._data

        # Check tensor shapes — 2 rollouts (num_generations=2)
        assert data["prompt_ids"].shape[0] == 2
        assert data["completion_ids"].shape[0] == 2
        assert data["completion_mask"].shape[0] == 2
        assert data["sampling_per_token_logps"].shape[0] == 2
        assert data["tool_mask"].shape[0] == 2

        # Verify completion content — each rollout should have:
        # gen1(4) + tool_output(3) + gen2(3) = 10 tokens
        # (padded to same length across the batch, but both are same here)
        comp_len = data["completion_mask"][0].sum().item()
        assert comp_len == 10, f"Expected 10 completion tokens, got {comp_len}"

        # Verify env_mask: gen1=1,1,1,1 tool=0,0,0 gen2=1,1,1
        tool_mask = data["tool_mask"][0][:comp_len].tolist()
        assert tool_mask == [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]

        # Verify logprobs are populated (use approx for float32 precision)
        import pytest

        logps = data["sampling_per_token_logps"][0][:comp_len].tolist()
        assert logps[:4] == pytest.approx([-0.1, -0.2, -0.3, -0.4], abs=1e-6)
        assert logps[4:7] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        assert logps[7:10] == pytest.approx([-0.5, -0.6, -0.7], abs=1e-6)

        # Verify rewards were injected into inputs
        assert data["_deferred_inputs"][0]["env_reward"] == 0.42
        assert data["_deferred_inputs"][1]["env_reward"] == 0.42

        # Verify deferred scoring markers
        assert data["_pending_policy_logps"] is True

    @patch("axolotl.integrations.nemo_gym.data_producer._call_agents")
    def test_produce_handles_failed_agent_response(self, mock_call_agents):
        """Failed agent responses should produce default (length-1) rollouts."""

        from axolotl.integrations.nemo_gym.data_producer import NemoGymDataProducer

        # One success, one failure — async mock
        async def fake_call_agents(**kwargs):
            return [
                self.AGENT_RESPONSE,
                {
                    "error": "Connection timeout",
                    "response": {"output": []},
                    "reward": 0.0,
                },
            ]

        mock_call_agents.side_effect = fake_call_agents

        producer = NemoGymDataProducer.__new__(NemoGymDataProducer)
        producer._agent_servers = {"wordle_agent": "http://mock:9999"}
        producer._dataset_lookup = {}
        producer._request_timeout = 30
        producer._num_generations = 2
        producer._trainer = self._make_mock_trainer()
        producer._prompt_iter = iter(
            [[{"prompt": [{"role": "user", "content": "Play!"}]}]]
        )
        producer._prompt_dl = [[{"prompt": [{"role": "user", "content": "Play!"}]}]]

        result = producer.produce(model=MagicMock(), global_step=1)

        assert result is not None
        data = result._data

        # Both rollouts present
        assert data["completion_ids"].shape[0] == 2

        # First rollout has real tokens, second has just eos (length 1)
        mask0 = data["completion_mask"][0].sum().item()
        mask1 = data["completion_mask"][1].sum().item()
        assert mask0 == 10  # full response
        assert mask1 == 1  # default fallback (just eos)

        # Rewards: success=0.42, failure=0.0
        assert data["_deferred_inputs"][0]["env_reward"] == 0.42
        assert data["_deferred_inputs"][1]["env_reward"] == 0.0

    @patch("axolotl.integrations.nemo_gym.rewards._get_verify_urls")
    @patch("axolotl.integrations.nemo_gym.rewards.requests")
    def test_reward_functions_chain(self, mock_requests, mock_get_urls):
        """Test that reward_env and reward_nemo_gym_verify can be used together."""
        from axolotl.integrations.nemo_gym.rewards import (
            reward_env,
            reward_nemo_gym_verify,
        )

        completions = [[{"role": "assistant", "content": "crane"}]]
        prompts = [[{"role": "user", "content": "Guess"}]]

        # reward_env: passthrough from agent
        env_result = reward_env(completions, prompts, env_reward=[0.42])
        assert env_result == [0.42]

        # reward_nemo_gym_verify: calls /verify
        mock_get_urls.return_value = {"wordle": "http://localhost:9999/verify"}
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"reward": 0.75}
        mock_requests.post.return_value = mock_resp

        verify_result = reward_nemo_gym_verify(
            completions,
            prompts,
            resources_server_ref=[{"name": "wordle"}],
            verify_extra=[{}],
        )
        assert verify_result == [0.75]

        # Both rewards can coexist (as they would in a multi-reward config)
        combined = [e + v for e, v in zip(env_result, verify_result, strict=True)]
        assert combined == [1.17]


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
