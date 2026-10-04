"""Projection sampling through Axolotl's configurable chat-template strategy."""

import json
from copy import deepcopy
from typing import Any, Callable

from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy

from .sampler import ProjectionSampler
from .scoring import evaluate_logprob_margin, evaluate_proposal_kl


def sample_chat(
    row: dict[str, Any],
    strategy: ChatTemplateStrategy,
    sampler: ProjectionSampler,
    verifier: Callable | None = None,
) -> dict[str, Any]:
    """Rewrite eligible assistant turns and cache the standard SFT parser's labels."""
    rewritten = deepcopy(row)
    prompter = strategy.prompter
    messages = deepcopy(strategy._get_messages(rewritten))
    if not messages:
        raise ValueError("Projection sampling requires nonempty chat messages")
    rewritten[prompter.field_messages] = messages
    tools = strategy._get_tools(rewritten)
    metadata = []
    legacy = not any(
        (
            strategy.roles_to_train,
            strategy.train_on_eos,
            strategy.train_on_eot,
            prompter.message_field_training,
            prompter.message_field_training_detail,
        )
    )
    for index, message in enumerate(messages):
        prefix_row = {**rewritten, prompter.field_messages: messages[: index + 1]}
        turns = strategy.get_conversation_thread(prefix_row)
        if not turns:
            continue
        turn = turns[-1]
        if turn.get("role") != "assistant":
            continue
        train = turn.get("training")
        if train is None:
            if legacy:
                train = strategy.train_on_inputs or index == len(messages) - 1
            elif (
                turn.get("training_detail") is not None
                or turn.get("reasoning_training_detail") is not None
            ):
                train = bool(turn.get("training_detail")) or bool(
                    turn.get("reasoning_training_detail")
                )
            else:
                train = (
                    strategy.train_on_inputs or "assistant" in strategy.roles_to_train
                )
        if not train:
            continue
        reason = None
        if (
            turn.get("training_detail") is not None
            or turn.get("reasoning_training_detail") is not None
        ):
            reason = "partial_training_mask"
        elif turn.get("tool_calls") or turn.get(prompter.template_thinking_key):
            reason = "structured_assistant_turn"
        elif not isinstance(turn.get("content"), str) or not turn["content"].strip():
            reason = "empty_assistant_content"
        if reason is not None:
            metadata.append({"message_index": index, "skipped": reason})
            continue
        context = list(
            prompter.build_prompt(turns[:-1], add_generation_prompt=True, tools=tools)
        )
        if not context:
            raise ValueError("The chat template produced an empty generation context")
        conversation = [
            {
                key: value
                for key, value in preceding.items()
                if key
                not in ("training", "training_detail", "reasoning_training_detail")
            }
            for preceding in turns[:-1]
        ]
        question = json.dumps(conversation, ensure_ascii=False)
        expert = turn["content"]

        def proposal_prompt(text: str) -> list[int]:
            return list(
                prompter.build_prompt(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tools=tools,
                )
            )

        result = sampler.sample(
            question, expert, target_context=context, prompt_builder=proposal_prompt
        )
        tokens = result.token_ids
        content_tokens = tokens[:-1] if result.finished else tokens
        response = sampler.backend.tokenizer.decode(
            content_tokens, skip_special_tokens=False
        )
        verified = (
            None
            if verifier is None
            else bool(
                verifier(question=question, expert_response=expert, response=response)
            )
        )
        fallback = not result.finished or not response.strip() or verified is False
        fallback_reason = None
        margin_metadata = {}
        kl_metadata = {}
        if not fallback:
            candidate_row = deepcopy(prefix_row)
            candidate_row[prompter.field_messages][-1][
                prompter.message_property_mappings["content"]
            ] = response
            rendered = list(
                prompter.build_prompt(
                    strategy.get_conversation_thread(candidate_row), tools=tools
                )
            )
            # Retokenization must retain the trajectory whose density was scored.
            if (
                rendered[: len(context)] != context
                or rendered[len(context) : len(context) + len(tokens)] != tokens
            ):
                fallback = True
                fallback_reason = "template_token_mismatch"
        if not fallback and sampler.config.min_logprob_improvement is not None:
            original_ids = list(prompter.build_prompt(turns, tools=tools))
            candidate_full = deepcopy(rewritten)
            candidate_full[prompter.field_messages][index][
                prompter.message_property_mappings["content"]
            ] = response
            original_tokenized = strategy.tokenize_prompt(deepcopy(rewritten))
            candidate_tokenized = strategy.tokenize_prompt(candidate_full)
            if (
                original_tokenized["input_ids"][: len(original_ids)] != original_ids
                or candidate_tokenized["input_ids"][: len(rendered)] != rendered
            ):
                fallback = True
                fallback_reason = "template_token_mismatch"
            else:
                turn_index = len(turns) - 1
                original_start, _ = strategy.find_turn(
                    strategy.get_conversation_thread(rewritten), turn_index, tools=tools
                )
                candidate_start, _ = strategy.find_turn(
                    strategy.get_conversation_thread(candidate_full),
                    turn_index,
                    tools=tools,
                )
                # Full-conversation masks retain last-turn EOS/EOT semantics.
                margin_metadata = evaluate_logprob_margin(
                    sampler.backend,
                    {
                        key: original_tokenized[key][: len(original_ids)]
                        for key in ("input_ids", "labels")
                    },
                    {
                        key: candidate_tokenized[key][: len(rendered)]
                        for key in ("input_ids", "labels")
                    },
                    sampler.config.min_logprob_improvement,
                    starts=(
                        original_start if original_start >= 0 else len(original_ids),
                        candidate_start if candidate_start >= 0 else len(rendered),
                    ),
                )
                if not margin_metadata["logprob_margin_passed"]:
                    fallback = True
                    fallback_reason = (
                        "no_labeled_response_tokens"
                        if margin_metadata["logprob_improvement"] is None
                        else "insufficient_logprob_improvement"
                    )
        if not fallback and sampler.config.max_proposal_kl is not None:
            if sampler.config.min_logprob_improvement is None:
                candidate_full = deepcopy(rewritten)
                candidate_full[prompter.field_messages][index][
                    prompter.message_property_mappings["content"]
                ] = response
                candidate_tokenized = strategy.tokenize_prompt(candidate_full)
                if candidate_tokenized["input_ids"][: len(rendered)] != rendered:
                    fallback = True
                    fallback_reason = "template_token_mismatch"
            if not fallback:
                kl_metadata = evaluate_proposal_kl(
                    sampler.backend,
                    context,
                    sampler.proposal_ids(question, expert, [], proposal_prompt),
                    tokens,
                    candidate_tokenized,
                    sampler.config.max_proposal_kl,
                )
                if not kl_metadata["proposal_kl_gate_passed"]:
                    fallback = True
                    fallback_reason = (
                        "no_labeled_response_tokens"
                        if kl_metadata["proposal_to_base_mean_kl"] is None
                        else "excessive_proposal_kl"
                    )
        if not fallback:
            message[prompter.message_property_mappings["content"]] = response
        metadata.append(
            {
                "message_index": index,
                "expert_response": expert,
                "sampled_token_ids": tokens,
                "attempts": result.attempts,
                "accepted": result.accepted,
                "target_logprob": result.target_logprob,
                "finished": result.finished,
                "verified": verified,
                "fallback_to_expert": fallback,
                "fallback_reason": fallback_reason,
                **margin_metadata,
                **kl_metadata,
                **sampler.proposal_statistics(result),
            }
        )
    tokenized = strategy.tokenize_prompt(rewritten)
    return {
        **({"tools": tools} if tools is not None else {}),
        **tokenized,
        "messages": strategy.get_conversation_thread(rewritten),
        "sampling": metadata,
    }
