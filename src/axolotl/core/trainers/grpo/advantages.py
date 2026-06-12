"""Selectable advantage estimators for GRPO-style RL trainers.

The estimator only swaps the baseline (``group_mean`` = GRPO, ``rloo`` =
leave-one-out per https://huggingface.co/papers/2402.14740,
``reinforce_plus_plus`` = global batch per
https://huggingface.co/papers/2501.03262); reward scaling stays governed by
``scale_rewards`` so the two remain orthogonal, mirroring TRL's semantics.
"""

import torch

ADVANTAGE_ESTIMATORS = ("group_mean", "rloo", "reinforce_plus_plus")


def compute_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    advantage_estimator: str = "group_mean",
    scale_rewards: str | bool = "group",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute advantages from a flat tensor of rewards.

    Args:
        rewards: Flat tensor of shape ``(B,)`` where ``B`` is a multiple of
            ``num_generations`` and completions of the same prompt are
            contiguous (the layout produced by GRPO rollouts after
            ``gather()``).
        num_generations: Number of completions sampled per prompt.
        advantage_estimator: One of ``"group_mean"``, ``"rloo"``, or
            ``"reinforce_plus_plus"``.
        scale_rewards: ``"group"`` (or ``True``) scales by the per-group
            standard deviation, ``"batch"`` by the standard deviation of the
            whole batch, ``"none"`` (or ``False``) applies no scaling.

    Returns:
        Tuple of ``(advantages, std_rewards, is_std_zero)`` where
        ``advantages`` has the same shape as ``rewards``, ``std_rewards`` is
        the (per-sample) standard deviation used for scaling/logging, and
        ``is_std_zero`` is a boolean tensor flagging samples whose scaling
        std is (numerically) zero.

    Raises:
        ValueError: If ``advantage_estimator`` or ``scale_rewards`` is
            invalid, or if ``rloo`` is requested with fewer than two
            generations per prompt.
    """
    if advantage_estimator not in ADVANTAGE_ESTIMATORS:
        raise ValueError(
            f"Invalid advantage_estimator: {advantage_estimator!r}. Must be one of "
            f"{ADVANTAGE_ESTIMATORS}."
        )
    # GRPOConfig.__post_init__ maps bools to strings, but accept bools here
    # too so the helper can be used with raw config values.
    scale_rewards = {True: "group", False: "none"}.get(scale_rewards, scale_rewards)
    if scale_rewards not in ("group", "batch", "none"):
        raise ValueError(
            f"Invalid scale_rewards: {scale_rewards!r}. Must be one of 'group', "
            "'batch', or 'none'."
        )

    grouped = rewards.view(-1, num_generations)

    if advantage_estimator == "group_mean":
        baseline = grouped.mean(dim=1, keepdim=True).expand_as(grouped)
    elif advantage_estimator == "rloo":
        if num_generations < 2:
            raise ValueError(
                "advantage_estimator 'rloo' requires num_generations >= 2, got "
                f"{num_generations}."
            )
        # Leave-one-out baseline: mean of the other (G - 1) rewards in the group
        baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (num_generations - 1)
    else:  # reinforce_plus_plus
        baseline = rewards.mean().expand_as(grouped)

    advantages = (grouped - baseline).reshape(-1)

    if scale_rewards == "batch":
        if rewards.numel() > 1:
            std_rewards = rewards.std().expand_as(rewards)
        else:
            std_rewards = torch.zeros_like(rewards)
    else:
        # For scale_rewards == "none", std_rewards is only used for logging
        if num_generations > 1:
            std_rewards = grouped.std(dim=1).repeat_interleave(num_generations, dim=0)
        else:
            std_rewards = torch.zeros_like(rewards)

    if scale_rewards != "none":
        advantages = advantages / (std_rewards + 1e-4)
    is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

    return advantages, std_rewards, is_std_zero
