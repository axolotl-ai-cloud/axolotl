"""Advantage estimators for GRPO-style RL trainers.

Implements the selectable baseline/advantage computations requested in
https://github.com/axolotl-ai-cloud/axolotl/issues/3676:

- ``group_mean`` (default): the standard GRPO baseline — the mean reward of
  the ``num_generations`` completions sampled for the same prompt.
- ``rloo``: REINFORCE Leave-One-Out — each completion is baselined against
  the mean reward of the *other* completions in its group, which keeps the
  baseline unbiased w.r.t. the sample it is applied to. See `Back to Basics:
  Revisiting REINFORCE-Style Optimization for Learning from Human Feedback
  in LLMs <https://huggingface.co/papers/2402.14740>`_.
- ``reinforce_plus_plus``: a global-batch baseline — every completion is
  baselined against the mean reward of the entire (gathered) batch, ignoring
  group structure. See `REINFORCE++: A Simple and Efficient Approach for
  Aligning Large Language Models <https://huggingface.co/papers/2501.03262>`_.
  The paper normalizes advantages with global batch statistics, so this
  estimator pairs naturally with ``scale_rewards: batch``.

The estimator only controls the *baseline*; reward scaling stays controlled
by ``scale_rewards`` (``group`` / ``batch`` / ``none``) so the two remain
orthogonal, mirroring TRL's GRPO semantics.
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
