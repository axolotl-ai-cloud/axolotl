# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Bounded cost refinement with protected optimizer-step label and token variance."""


def balance_microbatches(
    batches,
    lengths,
    counts,
    starts,
    batches_per_optimizer_step,
    padding_multiple=1,
    capacity=None,
    window_steps=1,
):
    """Swap samples within complete updates without increasing padding or label variance.

    Batches contain rows of sample indices. Fixed-count padded callers represent
    each sample as a row; flattened callers represent a microbatch as one row.
    Larger windows permit cross-step swaps only if step token and label variance
    do not increase. Incomplete windows are unchanged.
    Row cardinalities and causal row-leading target masks are preserved. Four
    paired passes consider at most 32 samples from each microbatch per swap.
    """
    if batches_per_optimizer_step < 1 or padding_multiple < 1 or window_steps < 1:
        raise ValueError("Update width and padding multiple must be positive")
    step_width = batches_per_optimizer_step
    width = step_width * window_steps
    if width == 1:
        return list(batches)
    lengths = [int(value) for value in lengths]
    counts = [int(value) for value in counts]
    starts = [int(value) for value in starts]
    result = [[list(row) for row in batch] for batch in batches]

    def cost(rows):
        return len(rows) * (
            (max(rows) + padding_multiple - 1) // padding_multiple * padding_multiple
        )

    end = len(result) // width * width
    all_used = [[sum(lengths[i] for i in row) for row in b] for b in result[:end]]
    global_sum = sum(cost(rows) for rows in all_used)
    for offset in range(0, end, width):
        window = result[offset : offset + width]
        used = all_used[offset : offset + width]
        costs = [cost(rows) for rows in used]
        tokens = [sum(rows) for rows in used]
        labels = [
            sum(sum(counts[i] for i in row) - int(starts[row[0]]) for row in b)
            for b in window
        ]
        step_tokens = [
            sum(tokens[i : i + step_width]) for i in range(0, width, step_width)
        ]
        step_labels = [
            sum(labels[i : i + step_width]) for i in range(0, width, step_width)
        ]
        old_sum = sum(costs)
        for pass_index in range(4):
            objective = costs if pass_index % 2 == 0 else tokens
            order = sorted(range(width), key=objective.__getitem__)
            changed = False
            for lo, hi in zip(order[: width // 2], reversed(order), strict=False):
                if costs[hi] == costs[lo] and tokens[hi] == tokens[lo]:
                    continue
                candidates = []
                for index, reverse in ((hi, True), (lo, False)):
                    items = [
                        (r, p, i)
                        for r, row in enumerate(window[index])
                        for p, i in enumerate(row)
                    ]
                    items.sort(key=lambda item: int(lengths[item[2]]), reverse=reverse)
                    candidates.append(items[:32])
                remaining = [
                    [max(rows[:r] + rows[r + 1 :], default=0) for r in range(len(rows))]
                    for rows in (used[hi], used[lo])
                ]
                best = None
                best_score = (max(costs[hi], costs[lo]), 0, 0, 0)
                for hr, hp, a in candidates[0]:
                    for lr, lp, b in candidates[1]:
                        delta = lengths[a] - lengths[b]
                        token_gain = 2 * delta * (tokens[hi] - tokens[lo] - delta)
                        if delta <= 0 or token_gain < 0:
                            continue
                        # A formerly row-leading supervised token must stay masked.
                        if (hp == 0) != (lp == 0) and (starts[a] or starts[b]):
                            continue
                        transfer = counts[a] - counts[b]
                        if hp == 0 and lp == 0:
                            transfer -= starts[a] - starts[b]
                        label_gain = 2 * transfer * (labels[hi] - labels[lo] - transfer)
                        if label_gain < 0:
                            continue
                        hs, ls = hi // step_width, lo // step_width
                        if hs != ls:
                            if (
                                2 * delta * (step_tokens[hs] - step_tokens[ls] - delta)
                                < 0
                            ):
                                continue
                            if (
                                2
                                * transfer
                                * (step_labels[hs] - step_labels[ls] - transfer)
                                < 0
                            ):
                                continue
                        hrow, lrow = used[hi][hr] - delta, used[lo][lr] + delta
                        if capacity is not None and max(hrow, lrow) > capacity:
                            continue
                        hc = len(used[hi]) * (
                            (max(hrow, remaining[0][hr]) + padding_multiple - 1)
                            // padding_multiple
                            * padding_multiple
                        )
                        lc = len(used[lo]) * (
                            (max(lrow, remaining[1][lr]) + padding_multiple - 1)
                            // padding_multiple
                            * padding_multiple
                        )
                        saved = costs[hi] + costs[lo] - hc - lc
                        if saved < 0 or max(hc, lc) > max(costs[hi], costs[lo]):
                            continue
                        variance_gain = width * (
                            costs[hi] ** 2 + costs[lo] ** 2 - hc**2 - lc**2
                        )
                        variance_gain += (old_sum - saved) ** 2 - old_sum**2
                        global_gain = end * (
                            costs[hi] ** 2 + costs[lo] ** 2 - hc**2 - lc**2
                        )
                        global_gain += (global_sum - saved) ** 2 - global_sum**2
                        if variance_gain < 0 or global_gain < 0:
                            continue
                        score = (max(hc, lc), -saved, -variance_gain, -token_gain)
                        if score < best_score:
                            best_score = score
                            best = (hr, hp, lr, lp, delta, transfer, hc, lc)
                if best is not None:
                    hr, hp, lr, lp, delta, transfer, hc, lc = best
                    window[hi][hr][hp], window[lo][lr][lp] = (
                        window[lo][lr][lp],
                        window[hi][hr][hp],
                    )
                    used[hi][hr] -= delta
                    used[lo][lr] += delta
                    hs, ls = hi // step_width, lo // step_width
                    step_tokens[hs] -= delta
                    step_tokens[ls] += delta
                    step_labels[hs] -= transfer
                    step_labels[ls] += transfer
                    tokens[hi] -= delta
                    tokens[lo] += delta
                    labels[hi] -= transfer
                    labels[lo] += transfer
                    saved = costs[hi] + costs[lo] - hc - lc
                    global_sum -= saved
                    old_sum -= saved
                    costs[hi], costs[lo] = hc, lc
                    changed = True
            if not changed and pass_index % 2:
                break
    return result
