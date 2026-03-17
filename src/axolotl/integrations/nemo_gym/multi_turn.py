# Copyright 2024 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.

"""
Multi-turn rollout function for NeMo Gym environments.

Implements TRL's rollout_func interface to orchestrate multi-step interactions
between the model and NeMo Gym resource servers. Follows the same pattern as
TRL's OpenEnv Wordle example:
  - prompt_ids = initial prompt tokens
  - completion_ids = all subsequent tokens (model + environment interleaved)
  - env_mask = 1 for model-generated tokens, 0 for environment tokens
  - logprobs = per-token log probabilities (0.0 for env tokens)
  - reward = scalar from /verify endpoint

The multi-turn loop:
  1. /seed_session → get available tools + initial observation
  2. Generate model response via generate_rollout_completions()
  3. Parse tool calls from model output
  4. Execute tool calls against resource server endpoints
  5. Append tool results to conversation
  6. Repeat until no tool calls, done, or max_turns reached
  7. /verify → get final reward
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def create_multi_turn_rollout_func(
    global_config: dict,
    verify_endpoints: dict[str, str],
    server_base_urls: dict[str, str],
    dataset_lookup: dict[str, dict],
    model_name: str = "axolotl-model",
    max_turns: int = 10,
    verify_timeout: int = 30,
    tool_timeout: int = 30,
):
    """Create a TRL-compatible rollout_func for multi-turn NeMo Gym interactions.

    Args:
        global_config: NeMo Gym global config dict.
        verify_endpoints: Mapping of server_name → /verify URL.
        server_base_urls: Mapping of server_name → base URL (for /seed_session and tools).
        dataset_lookup: Mapping of prompt text → dataset row dict (for verify_extra, server_name).
        model_name: Model name for verify requests.
        max_turns: Maximum turns per rollout.
        verify_timeout: Timeout for /verify requests.
        tool_timeout: Timeout for tool execution requests.

    Returns:
        A rollout_func with signature (prompts: list[str], trainer) → dict.
    """

    def rollout_func(prompts: list[str], trainer) -> dict[str, Any]:
        print(f"\n[ROLLOUT_FUNC CALLED] num_prompts={len(prompts)}, first_prompt_len={len(prompts[0]) if prompts else 0}")
        from trl.experimental.openenv import generate_rollout_completions

        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_env_masks = []
        all_rewards = []
        all_tool_used = []

        tokenizer = trainer.processing_class

        for prompt_text in prompts:
            # TRL may pass already-templated strings (colocate mode applies chat template
            # before calling rollout_func). Extract the raw user message for dataset lookup.
            raw_prompt = prompt_text
            for key in dataset_lookup:
                if key in prompt_text:
                    raw_prompt = key
                    break
            episode = _rollout_once(
                trainer=trainer,
                tokenizer=tokenizer,
                prompt_text=raw_prompt,
                global_config=global_config,
                verify_endpoints=verify_endpoints,
                server_base_urls=server_base_urls,
                dataset_lookup=dataset_lookup,
                model_name=model_name,
                max_turns=max_turns,
                verify_timeout=verify_timeout,
                tool_timeout=tool_timeout,
                generate_fn=generate_rollout_completions,
            )
            all_prompt_ids.append(episode["prompt_ids"])
            all_completion_ids.append(episode["completion_ids"])
            all_logprobs.append(episode["logprobs"])
            all_env_masks.append(episode["env_mask"])
            all_rewards.append(episode["reward"])
            # Track tool usage (env_mask has 0s when tools were executed)
            has_tool = any(m == 0 for m in episode["env_mask"])
            all_tool_used.append(1.0 if has_tool else 0.0)

        # TRL expects logprobs as list[list[list[float]]] (batch x seq x top-k)
        # Wrap each scalar logprob in a single-element list for top-1 format
        wrapped_logprobs = [[[lp] for lp in seq] for seq in all_logprobs]

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": wrapped_logprobs,
            "env_mask": all_env_masks,
            "nemo_gym_reward": all_rewards,
            "tool_used": all_tool_used,
        }

    return rollout_func


def _rollout_once(
    trainer,
    tokenizer,
    prompt_text: str,
    global_config: dict,
    verify_endpoints: dict[str, str],
    server_base_urls: dict[str, str],
    dataset_lookup: dict[str, dict],
    model_name: str,
    max_turns: int,
    verify_timeout: int,
    tool_timeout: int,
    generate_fn,
) -> dict[str, Any]:
    """Run one multi-turn rollout episode.

    Returns dict with prompt_ids, completion_ids, logprobs, env_mask, reward.
    """
    # Look up dataset row for this prompt
    row_data = dataset_lookup.get(prompt_text, {})
    server_ref = row_data.get("resources_server_ref", {})
    server_name = server_ref.get("name") if server_ref else None
    verify_extra = row_data.get("verify_extra", {})

    base_url = server_base_urls.get(server_name, "") if server_name else ""
    verify_url = verify_endpoints.get(server_name, "") if server_name else ""

    # Step 1: Seed session to get tools and initial observation
    tools, initial_observation = _seed_session(
        base_url, verify_extra, timeout=tool_timeout
    )

    # Also extract tools from the dataset row if seed_session didn't provide them
    # NeMo Gym datasets often have tools in responses_create_params.tools
    if not tools and verify_extra:
        dataset_tools = verify_extra.get("responses_create_params", {}).get("tools", [])
        if dataset_tools:
            for tool_def in dataset_tools:
                if isinstance(tool_def, dict) and "function" in tool_def:
                    # Already in OpenAI format
                    tools.append(tool_def)
                elif isinstance(tool_def, dict) and "name" in tool_def:
                    # NeMo Gym flat format — wrap in OpenAI function tool format
                    # Strip non-standard fields that confuse chat templates
                    func_def = {
                        "name": tool_def["name"],
                        "parameters": tool_def.get("parameters", {}),
                    }
                    if tool_def.get("description"):
                        func_def["description"] = tool_def["description"]
                    tools.append({
                        "type": "function",
                        "function": func_def,
                    })

    # Convert NeMo Gym tool format to the format expected by chat templates
    # NeMo Gym: [{"type": "function", "function": {"name": ..., "parameters": ...}}]
    # Some tokenizers want tools directly, others want functions
    chat_tools = tools if tools else None

    # Build initial prompt messages with tool-use instruction
    system_content = ""
    if initial_observation:
        system_content = initial_observation
    if chat_tools:
        # Add explicit instruction to use tools — critical for models like Qwen3
        # that default to thinking mode instead of tool calling
        tool_instruction = (
            "You MUST use the provided tools to answer the user's question. "
            "Call the appropriate tool first, then provide your final answer "
            "based on the tool's response. Do not answer without using a tool."
        )
        if system_content:
            system_content = tool_instruction + "\n\n" + system_content
        else:
            system_content = tool_instruction

    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt_text})

    # Build template kwargs — pass tools if the tokenizer supports it
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if chat_tools:
        template_kwargs["tools"] = chat_tools
    # Disable thinking mode for models that support it (e.g., Qwen3)
    # Thinking wastes tokens and prevents tool calling
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            enable_thinking=False, tokenize=False,
        )
        template_kwargs["enable_thinking"] = False
    except TypeError:
        pass  # Tokenizer doesn't support enable_thinking

    try:
        initial_prompt_text = tokenizer.apply_chat_template(
            messages, **template_kwargs
        )
    except Exception:
        # Fallback: some tokenizers don't support tools kwarg
        # Include tool descriptions in the system message instead
        if chat_tools:
            tool_desc = "Available tools:\n" + json.dumps(chat_tools, indent=2)
            if messages[0]["role"] == "system":
                messages[0]["content"] += "\n\n" + tool_desc
            else:
                messages.insert(0, {"role": "system", "content": tool_desc})
        initial_prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    prompt_ids = list(tokenizer.encode(initial_prompt_text, add_special_tokens=False))
    # Debug: print what we built
    print(f"[ROLLOUT_ONCE] tools={len(tools)}, msgs={len(messages)}, "
          f"prompt_tokens={len(prompt_ids)}, "
          f"msg_roles={[m['role'] for m in messages]}, "
          f"sys_len={len(messages[0]['content']) if messages and messages[0]['role']=='system' else 0}")

    completion_ids: list[int] = []
    logprobs_list: list[float] = []
    env_mask: list[int] = []
    accumulated_messages = list(messages)

    reward = 0.0
    done = False

    for turn in range(max_turns):
        if done:
            break

        # Build the full conversation for generation (with tools)
        try:
            gen_prompt = tokenizer.apply_chat_template(
                accumulated_messages, **template_kwargs
            )
        except Exception:
            gen_prompt = tokenizer.apply_chat_template(
                accumulated_messages, add_generation_prompt=True, tokenize=False
            )

        # Generate model response
        # Pass as_chat=False since we already applied the chat template with tools
        if turn == 0 and not done:
            tok_len = len(tokenizer.encode(gen_prompt))
            print(f"\n[MULTI_TURN] <tools>: {'<tools>' in gen_prompt}, "
                  f"get_weather: {'get_weather' in gen_prompt}, "
                  f"tokens: {tok_len}, "
                  f"ends_with_think: {gen_prompt.rstrip().endswith('</think>')}, "
                  f"last_50: ...{repr(gen_prompt[-50:])}")
        try:
            outputs = generate_fn(trainer, [gen_prompt], as_chat=False)[0]
        except RuntimeError as exc:
            LOG.warning(f"Generation failed at turn {turn}: {exc}")
            break

        gen_ids = list(outputs["completion_ids"])
        gen_logprobs = list(outputs["logprobs"])
        gen_text = outputs.get("text") or tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Also decode WITHOUT skip_special_tokens to see if tool_call tokens are being stripped
        gen_text_raw = tokenizer.decode(gen_ids, skip_special_tokens=False)

        if turn == 0:
            print(f"[MULTI_TURN GEN] Turn {turn}, text (200): {gen_text[:200]}")
            print(f"[MULTI_TURN GEN] raw (200): {gen_text_raw[:200]}")
            print(f"[MULTI_TURN GEN] tool_call: {'<tool_call>' in gen_text}, ids[:10]: {gen_ids[:10]}")

        # Add model tokens to episode (these are trainable)
        completion_ids.extend(gen_ids)
        logprobs_list.extend(gen_logprobs)
        env_mask.extend([1] * len(gen_ids))

        # Parse tool calls from the model output
        tool_calls = _parse_tool_calls(gen_text, tools)

        if tool_calls and base_url:
            # Execute tool calls and get results
            tool_results = _execute_tool_calls(
                base_url, tool_calls, timeout=tool_timeout
            )

            # Build tool result as JSON string (matches chat template expectations)
            tool_result_text = _format_tool_results(tool_results)

            # Build the conversation with proper role for tool results
            # Use "tool" role which chat templates map to <tool_response> blocks
            next_messages = list(accumulated_messages)
            next_messages.append({"role": "assistant", "content": gen_text})
            for result in tool_results:
                next_messages.append({
                    "role": "tool",
                    "content": json.dumps(result.get("output", "")),
                    "name": result.get("name", ""),
                })

            # Compute the env tokens as the delta between full template and current
            try:
                next_prompt = tokenizer.apply_chat_template(
                    next_messages, **template_kwargs
                )
                current_prompt = tokenizer.apply_chat_template(
                    accumulated_messages + [{"role": "assistant", "content": gen_text}],
                    add_generation_prompt=False, tokenize=False,
                )
                # The env tokens are the difference between next and current prompts
                # (the tool response block + generation prompt)
                env_text = next_prompt[len(current_prompt):]
                env_tokens = tokenizer.encode(env_text, add_special_tokens=False)
            except Exception:
                # Fallback: use plain text formatting
                env_tokens = tokenizer.encode(tool_result_text, add_special_tokens=False)

            completion_ids.extend(env_tokens)
            logprobs_list.extend([0.0] * len(env_tokens))
            env_mask.extend([0] * len(env_tokens))

            # Update conversation history with proper roles
            accumulated_messages = next_messages
        else:
            # No tool calls - this is the final response
            accumulated_messages.append({"role": "assistant", "content": gen_text})
            done = True

    # Step 7: Verify and get reward
    if verify_url:
        reward = _verify_completion(
            verify_url=verify_url,
            verify_extra=verify_extra,
            prompt_text=prompt_text,
            accumulated_messages=accumulated_messages,
            model_name=model_name,
            timeout=verify_timeout,
        )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs_list,
        "env_mask": env_mask,
        "reward": reward,
    }


def _seed_session(
    base_url: str,
    verify_extra: dict,
    timeout: int = 30,
) -> tuple[list[dict], str]:
    """Call /seed_session to initialize the environment and get available tools.

    Returns:
        (tools, initial_observation) where tools is a list of OpenAI function tool
        defs and initial_observation is a string observation (may be empty).
    """
    if not base_url:
        return [], ""

    seed_url = f"{base_url}/seed_session"
    try:
        # Build seed request from the dataset row
        seed_request = {}
        if "responses_create_params" in verify_extra:
            seed_request["responses_create_params"] = verify_extra[
                "responses_create_params"
            ]
        if "metadata" in verify_extra:
            seed_request["metadata"] = verify_extra["metadata"]
        if "task_idx" in verify_extra:
            seed_request["task_idx"] = verify_extra["task_idx"]

        resp = requests.post(seed_url, json=seed_request, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            tools = data.get("tools", [])
            # Extract observation from response
            obs_items = data.get("obs", [])
            obs_text = ""
            if obs_items:
                obs_parts = []
                for item in obs_items:
                    if isinstance(item, dict):
                        content = item.get("content", "")
                        if content:
                            obs_parts.append(content)
                    elif isinstance(item, str):
                        obs_parts.append(item)
                obs_text = "\n".join(obs_parts)
            return tools, obs_text
        elif resp.status_code == 404:
            # /seed_session not supported - single-turn environment
            return [], ""
        else:
            LOG.warning(
                f"seed_session returned {resp.status_code}: {resp.text[:200]}"
            )
            return [], ""
    except requests.exceptions.RequestException as exc:
        LOG.debug(f"seed_session failed (may not be supported): {exc}")
        return [], ""


def _parse_tool_calls(
    model_output: str, available_tools: list[dict]
) -> list[dict[str, Any]]:
    """Parse tool/function calls from model output.

    Supports two formats:
    1. OpenAI function_call JSON format: {"name": "...", "arguments": {...}}
    2. XML-style tool calls: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    Returns list of dicts with 'name' and 'arguments' keys.
    """
    if not available_tools:
        return []

    tool_names = set()
    for tool in available_tools:
        if isinstance(tool, dict):
            func = tool.get("function", tool)
            name = func.get("name", "")
            if name:
                tool_names.add(name)

    if not tool_names:
        return []

    calls = []

    # Try XML-style tool calls: <tool_call>...</tool_call>
    xml_pattern = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
    )
    for match in xml_pattern.finditer(model_output):
        try:
            call_data = json.loads(match.group(1))
            name = call_data.get("name", "")
            if name in tool_names:
                calls.append(
                    {
                        "name": name,
                        "arguments": call_data.get("arguments", {}),
                    }
                )
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Try JSON blocks with function call structure (supports nested braces)
    json_pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*(?:\{[^{}]*\}[^{}]*)?\}')
    for match in json_pattern.finditer(model_output):
        try:
            call_data = json.loads(match.group())
            name = call_data.get("name", "")
            if name in tool_names:
                calls.append(
                    {
                        "name": name,
                        "arguments": call_data.get("arguments", {}),
                    }
                )
        except json.JSONDecodeError:
            continue

    return calls


def _execute_tool_calls(
    base_url: str,
    tool_calls: list[dict[str, Any]],
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Execute tool calls against the resource server.

    Each tool call is POSTed to base_url/{tool_name}.

    Returns list of dicts with 'name', 'call_id', and 'output' keys.
    """
    results = []
    for i, call in enumerate(tool_calls):
        name = call["name"]
        arguments = call.get("arguments", {})
        call_id = f"call_{i}"

        tool_url = f"{base_url}/{name}"
        try:
            resp = requests.post(tool_url, json=arguments, timeout=timeout)
            if resp.status_code == 200:
                output = resp.text
                # Try to extract a meaningful string from JSON response
                try:
                    resp_data = resp.json()
                    if isinstance(resp_data, dict):
                        # Use the first string value or the full JSON
                        output = resp_data.get(
                            "output",
                            resp_data.get("result", json.dumps(resp_data)),
                        )
                    elif isinstance(resp_data, str):
                        output = resp_data
                except (json.JSONDecodeError, ValueError):
                    pass
                results.append(
                    {"name": name, "call_id": call_id, "output": str(output)}
                )
            else:
                LOG.warning(
                    f"Tool {name} returned {resp.status_code}: {resp.text[:200]}"
                )
                results.append(
                    {
                        "name": name,
                        "call_id": call_id,
                        "output": f"Error: HTTP {resp.status_code}",
                    }
                )
        except requests.exceptions.RequestException as exc:
            LOG.warning(f"Tool execution failed for {name}: {exc}")
            results.append(
                {
                    "name": name,
                    "call_id": call_id,
                    "output": f"Error: {exc}",
                }
            )

    return results


def _format_tool_results(tool_results: list[dict[str, Any]]) -> str:
    """Format tool execution results as text for the conversation."""
    parts = []
    for result in tool_results:
        name = result["name"]
        output = result["output"]
        parts.append(f"[Tool: {name}]\n{output}")
    return "\n\n".join(parts)


def _verify_completion(
    verify_url: str,
    verify_extra: dict,
    prompt_text: str,
    accumulated_messages: list[dict[str, str]],
    model_name: str,
    timeout: int = 30,
) -> float:
    """Call /verify to get the final reward for a completed rollout.

    Constructs the verify request from the accumulated conversation and
    original dataset row data.
    """
    # Extract the final assistant response(s)
    assistant_outputs = []
    for msg in accumulated_messages:
        if msg["role"] == "assistant":
            assistant_outputs.append(
                {
                    "id": f"msg_{len(assistant_outputs)}",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": msg["content"],
                            "annotations": [],
                        }
                    ],
                }
            )

    verify_request = {k: v for k, v in verify_extra.items() if v is not None}
    verify_request["responses_create_params"] = {
        "input": [{"role": "user", "content": prompt_text}]
    }
    verify_request["response"] = {
        "id": "resp",
        "created_at": 0,
        "model": model_name,
        "object": "response",
        "output": assistant_outputs,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }

    try:
        resp = requests.post(verify_url, json=verify_request, timeout=timeout)
        if resp.status_code == 200:
            return float(resp.json().get("reward", 0.0))
        LOG.warning(f"Verify returned {resp.status_code}: {resp.text[:200]}")
        return 0.0
    except requests.exceptions.RequestException as exc:
        LOG.warning(f"Verify failed: {exc}")
        return 0.0
