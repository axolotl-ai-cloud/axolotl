"""
Dataset transform for multi-turn chat data with structured EBFT (vLLM mode).

Three variants:

1. `transform` — Uses the FIRST assistant turn as the generation target.
   Passes remaining turns as `remaining_turns` for sequential rollout.
   The trainer generates turn 1 via GRPO/vLLM, then sequentially generates
   subsequent assistant turns, comparing the full conversation to GT.

2. `transform_last_turn` — Uses the LAST assistant turn as the target.
   Simplest approach: the full conversation history is the prompt.

3. `transform_all_turns` — Explodes each conversation into N examples
   (one per assistant turn). Each turn is an independent training example.
   Use with batched=True.

Supports OpenAI chat format:
  {"messages": [{"role": ..., "content": ...}, ...]}
"""


def transform(cfg, **kwargs):
    """Multi-turn with sequential rollout.

    Returns the first assistant turn as ground_truth, plus remaining_turns
    for the trainer to do sequential rollout generation.
    """

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        if not messages:
            return {"prompt": [], "ground_truth": ""}

        # Split at first assistant turn
        prompt_msgs = []
        first_gt = None
        remaining = []

        found_first = False
        for msg in messages:
            if msg["role"] == "assistant" and not found_first:
                first_gt = msg["content"]
                found_first = True
            elif found_first:
                remaining.append(msg)
            else:
                prompt_msgs.append(msg)

        if first_gt is None:
            return {"prompt": prompt_msgs, "ground_truth": ""}

        # Store only the first assistant turn as ground_truth. The full multi-turn
        # GT is reconstructed in the reward function via chat template rendering
        # (using remaining_turns), which preserves role markers between turns.
        return {
            "prompt": prompt_msgs,
            "ground_truth": first_gt,
            "remaining_turns": remaining,
        }

    return transform_fn, {
        "remove_columns": "__all__",
    }


def transform_last_turn(cfg, **kwargs):
    """Single-turn: use the last assistant turn as the generation target."""

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        if not messages:
            return {"prompt": [], "ground_truth": ""}

        # Find all assistant turns
        history = []
        last_prompt = []
        last_gt = ""
        for msg in messages:
            if msg["role"] == "assistant":
                last_prompt = list(history)
                last_gt = msg["content"]
            history.append(msg)

        return {
            "prompt": last_prompt,
            "ground_truth": last_gt,
        }

    return transform_fn, {
        "remove_columns": "__all__",
    }


def transform_all_turns(cfg, **kwargs):
    """Explode: one example per assistant turn.

    Use with datasets.map(batched=True) to produce N examples from
    each N-turn conversation.

    Usage in YAML:
        type: ebft_chat_multiturn.transform_all_turns
    """

    def transform_fn(examples, tokenizer=None):
        all_prompts = []
        all_ground_truths = []

        messages_list = examples.get("messages", examples.get("conversations", []))

        for messages in messages_list:
            history = []
            for msg in messages:
                if msg["role"] == "assistant":
                    all_prompts.append(list(history))
                    all_ground_truths.append(msg["content"])
                history.append(msg)

        return {
            "prompt": all_prompts,
            "ground_truth": all_ground_truths,
        }

    return transform_fn, {
        "remove_columns": "__all__",
        "batched": True,
    }
