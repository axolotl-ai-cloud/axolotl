"""
Tests pinning the KD teacher-target contract shared by the three teacher producers
(vllm online, sglang online, offline chat_template):

    target row j is the teacher distribution over input_ids[j + 1],
    masked by labels[j + 1]

which is the convention the fused loss kernel consumes.
"""

from unittest.mock import MagicMock

import orjson
import pytest
import requests
import torch

from axolotl.integrations.kd.chat_template import (
    ChatTemplateStrategyWithKD,
    ChatTemplateStrategyWithKDv2,
)
from axolotl.integrations.kd.collator_online_teacher import OnlineTeacherCollator
from axolotl.integrations.kd.kernels.liger import LigerFusedLinearKLTopKLogprobLoss
from axolotl.integrations.kd.utils import LOGPROB_PAD_VALUE

TOP_K = 3
INPUT_IDS = [10, 11, 12, 13, 14]
LABELS = [-100, -100, 12, 13, 14]
SEQ_LEN = len(INPUT_IDS)

# teacher distribution over token `tok`: top-1 is the token itself, so every producer's
# alignment is observable from target_token_ids alone
DECOY_LOGPROBS = [-1.0, -2.5]


def teacher_row(token_id: int) -> list[tuple[int, float]]:
    """(token_id, logprob) candidates, highest first, for the distribution over token_id."""
    return [
        (token_id, -0.1),
        (900 + token_id, DECOY_LOGPROBS[0]),
        (800 + token_id, DECOY_LOGPROBS[1]),
    ]


def make_collator(server="vllm", topk=TOP_K, kd_temperature=1.0, normalize=True):
    tokenizer = MagicMock()
    tokenizer.deprecation_warnings = {}
    tokenizer.padding_side = "right"
    return OnlineTeacherCollator(
        tokenizer,
        kd_online_server_base_url="http://teacher:8000",
        kd_online_topk=topk,
        kd_temperature=kd_temperature,
        kd_online_server=server,
        kd_normalize_topk=normalize,
        kd_online_preflight=False,
    )


def fake_response(payload, status_code=200):
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.content = orjson.dumps(payload)
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def vllm_payload(rows_by_position):
    """rows_by_position[i] -> candidates for the distribution over input_ids[i]."""
    prompt_logprobs = [None]
    for i in range(1, SEQ_LEN):
        prompt_logprobs.append(
            {str(token_id): {"logprob": lp} for token_id, lp in rows_by_position[i]}
        )
    return {"choices": [{"prompt_logprobs": prompt_logprobs}]}


def sglang_payload(rows_by_position):
    input_top_logprobs = [None]
    for i in range(1, SEQ_LEN):
        input_top_logprobs.append(
            [[lp, token_id, f"<{token_id}>"] for token_id, lp in rows_by_position[i]]
        )
    return [{"meta_info": {"input_top_logprobs": input_top_logprobs}}]


def offline_sample(rows_by_position, first_position=1):
    """Offline logprobs cover the trailing tokens [first_position, SEQ_LEN)."""
    logprobs = []
    for i in range(first_position, SEQ_LEN):
        logprobs.append(
            [
                {"logprob": lp, "token": f"token_id:{token_id}"}
                for token_id, lp in rows_by_position[i]
            ]
        )
    return {
        "input_ids": list(INPUT_IDS),
        "labels": list(LABELS),
        "logprobs": logprobs,
    }


def offline_v2_sample(rows_by_position, first_position=1):
    """v2 datasets carry logprob values and their token ids in parallel fields."""
    logprobs = []
    target_token_ids = []
    for i in range(first_position, SEQ_LEN):
        logprobs.append([lp for _, lp in rows_by_position[i]])
        target_token_ids.append([token_id for token_id, _ in rows_by_position[i]])
    return {
        "input_ids": list(INPUT_IDS),
        "labels": list(LABELS),
        "logprobs": logprobs,
        "target_token_ids": target_token_ids,
    }


def make_offline_strategy(kd_temperature=1.0, gen_temperature=1.0, cls=None):
    prompter = MagicMock()
    prompter.roles = {"user": "user", "assistant": "assistant"}
    prompter.chat_template_msg_variables = ["role", "content"]
    prompter.chat_template = "{{ messages }}"

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.encode = MagicMock(return_value=[2])

    return (cls or ChatTemplateStrategyWithKD)(
        prompter=prompter,
        tokenizer=tokenizer,
        train_on_inputs=False,
        sequence_len=512,
        logprobs_field="logprobs",
        gen_temperature=gen_temperature,
        kd_temperature=kd_temperature,
    )


def run_online(server, payload, collator=None, labels=None):
    collator = collator or make_collator(server=server)
    collator.http_session.post = MagicMock(return_value=fake_response(payload))
    fetch = (
        collator.fetch_online_logprobs_sglang
        if server == "sglang"
        else collator.fetch_online_logprobs_vllm
    )
    return collator, fetch([list(INPUT_IDS)], [list(labels or LABELS)])


def assert_pinned(target_token_ids, target_mask, covered_from=0):
    """target row j must describe input_ids[j+1] and be masked by labels[j+1]."""
    assert len(target_token_ids) == SEQ_LEN
    assert len(target_mask) == SEQ_LEN

    for j in range(covered_from, SEQ_LEN - 1):
        assert target_token_ids[j][0] == INPUT_IDS[j + 1], (
            f"row {j} describes token {target_token_ids[j][0]}, expected {INPUT_IDS[j + 1]}"
        )
        expected_mask = 0 if LABELS[j + 1] == -100 else 1
        assert target_mask[j] == [expected_mask] * TOP_K, f"row {j} mask mismatch"

    # rows before the teacher's coverage and the final row (which would predict a token
    # outside the sequence) are padding
    for j in list(range(covered_from)) + [SEQ_LEN - 1]:
        assert target_mask[j] == [0] * TOP_K


@pytest.fixture(name="rows")
def rows_fixture():
    return {i: teacher_row(INPUT_IDS[i]) for i in range(SEQ_LEN)}


def test_vllm_producer_alignment(rows):
    _, result = run_online("vllm", vllm_payload(rows))
    assert_pinned(result["target_token_ids"][0], result["target_mask"][0])


def test_sglang_producer_alignment(rows):
    _, result = run_online("sglang", sglang_payload(rows))
    assert_pinned(result["target_token_ids"][0], result["target_mask"][0])


def test_offline_producer_alignment(rows):
    sample = make_offline_strategy().transform_logprobs(offline_sample(rows))
    assert_pinned(sample["target_token_ids"], sample["target_mask"])


def test_offline_producer_alignment_with_prompt_padding(rows):
    """Teacher rows covering only the tail still land one position to the left."""
    sample = make_offline_strategy().transform_logprobs(
        offline_sample(rows, first_position=3)
    )
    assert_pinned(sample["target_token_ids"], sample["target_mask"], covered_from=2)


def test_offline_v2_producer_alignment(rows):
    """v2 is the strategy the `axolotl.integrations.kd.chat_template` type resolves to."""
    strategy = make_offline_strategy(cls=ChatTemplateStrategyWithKDv2)
    sample = strategy.transform_logprobs(offline_v2_sample(rows))
    assert_pinned(sample["target_token_ids"], sample["target_mask"])


def test_offline_v2_matches_v1(rows):
    v1 = make_offline_strategy().transform_logprobs(offline_sample(rows))
    v2 = make_offline_strategy(cls=ChatTemplateStrategyWithKDv2).transform_logprobs(
        offline_v2_sample(rows)
    )

    assert v1["target_token_ids"] == v2["target_token_ids"]
    assert v1["target_mask"] == v2["target_mask"]
    assert torch.allclose(
        torch.tensor(v1["target_logprobs"]), torch.tensor(v2["target_logprobs"])
    )


def test_offline_empty_rows_are_masked(rows):
    """A position the teacher returned nothing for is padded out, not crashed on."""
    sample = offline_sample(rows)
    sample["logprobs"][1] = None

    result = make_offline_strategy().transform_logprobs(sample)

    assert result["target_mask"][1] == [0] * TOP_K
    assert result["target_logprobs"][1] == [LOGPROB_PAD_VALUE] * TOP_K
    assert len(result["target_logprobs"]) == SEQ_LEN


@pytest.mark.parametrize("kd_temperature", [1.0, 2.0])
def test_online_and_offline_producers_agree(rows, kd_temperature):
    """Same teacher distributions must give identical targets online and offline."""
    _, vllm_result = run_online(
        "vllm",
        vllm_payload(rows),
        collator=make_collator(kd_temperature=kd_temperature),
    )
    _, sglang_result = run_online(
        "sglang",
        sglang_payload(rows),
        collator=make_collator(server="sglang", kd_temperature=kd_temperature),
    )
    offline = make_offline_strategy(kd_temperature=kd_temperature).transform_logprobs(
        offline_sample(rows)
    )

    assert vllm_result["target_token_ids"][0] == sglang_result["target_token_ids"][0]
    assert vllm_result["target_token_ids"][0] == offline["target_token_ids"]
    assert vllm_result["target_mask"][0] == offline["target_mask"]

    online_logprobs = torch.tensor(vllm_result["target_logprobs"][0])
    sglang_logprobs = torch.tensor(sglang_result["target_logprobs"][0])
    offline_logprobs = torch.tensor(offline["target_logprobs"])
    assert torch.allclose(online_logprobs, sglang_logprobs, atol=1e-6)
    assert torch.allclose(online_logprobs, offline_logprobs, atol=1e-6)


def test_online_temperature_rescales_teacher(rows):
    """kd_temperature must flatten the online teacher distribution, not just the student."""
    _, cold = run_online("vllm", vllm_payload(rows), collator=make_collator())
    _, warm = run_online(
        "vllm", vllm_payload(rows), collator=make_collator(kd_temperature=4.0)
    )

    cold_probs = torch.tensor(cold["target_logprobs"][0][1]).exp()
    warm_probs = torch.tensor(warm["target_logprobs"][0][1]).exp()
    assert warm_probs.max() < cold_probs.max()
    assert torch.allclose(warm_probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_short_topk_rows_are_padded_and_masked_out(rows):
    """A position with fewer than top-k entries pads with finite logprobs and mask 0."""
    rows[2] = rows[2][:1]
    _, result = run_online("vllm", vllm_payload(rows))

    short_row = 1  # distribution over input_ids[2]
    assert result["target_mask"][0][short_row] == [1, 0, 0]
    assert result["target_logprobs"][0][short_row][1:] == [
        LOGPROB_PAD_VALUE,
        LOGPROB_PAD_VALUE,
    ]
    assert all(
        torch.isfinite(torch.tensor(row)).all() for row in result["target_logprobs"][0]
    )


def test_short_topk_rows_give_finite_loss(rows):
    """The padded rows must not turn into NaN inside the fused kernel."""
    rows[2] = rows[2][:1]
    _, result = run_online("vllm", vllm_payload(rows))

    torch.manual_seed(0)
    hidden = torch.randn(1, SEQ_LEN, 8, requires_grad=True)
    lm_head = torch.randn(1024, 8, requires_grad=True)
    loss_fn = LigerFusedLinearKLTopKLogprobLoss(
        weight_hard_loss=0.0,
        weight_soft_loss=1.0,
        temperature=1.0,
        beta=0.0,
        compiled=False,
        chunk_size=8,
        compute_ce_loss=False,
    )
    loss = loss_fn(
        lm_head,
        hidden,
        torch.tensor([result["target_token_ids"][0]]),
        torch.tensor([result["target_logprobs"][0]]),
        torch.tensor([result["target_mask"][0]], dtype=torch.int32),
        torch.tensor([LABELS]),
    )
    loss.backward()

    assert torch.isfinite(loss).all()
    assert torch.isfinite(hidden.grad).all()


def test_vllm_candidates_are_sorted_before_truncation(rows):
    """vllm returns k+1 entries with the sampled token first; keep the k highest."""
    rows[1] = [(INPUT_IDS[1], -5.0)] + [
        (700, -0.1),
        (701, -0.2),
        (702, -0.3),
    ]
    _, result = run_online("vllm", vllm_payload(rows))

    assert result["target_token_ids"][0][0] == [700, 701, 702]
    logprobs = result["target_logprobs"][0][0]
    assert logprobs == sorted(logprobs, reverse=True)


def test_vllm_payload_requests_no_generation(rows):
    collator, _ = run_online("vllm", vllm_payload(rows))
    _, kwargs = collator.http_session.post.call_args
    payload = kwargs["json"]

    assert payload == {
        "prompt": [list(INPUT_IDS)],
        "max_tokens": 0,
        "echo": True,
        "prompt_logprobs": TOP_K,
    }


def test_sglang_payload_requests_no_generation(rows):
    collator, _ = run_online("sglang", sglang_payload(rows))
    _, kwargs = collator.http_session.post.call_args
    payload = kwargs["json"]

    assert payload["sampling_params"] == {"max_new_tokens": 0}
    assert "temperature" not in payload
    assert "temperature" not in payload["sampling_params"]


def test_teacher_length_mismatch_raises(rows):
    payload = vllm_payload(rows)
    payload["choices"][0]["prompt_logprobs"].pop()
    collator = make_collator()
    collator.http_session.post = MagicMock(return_value=fake_response(payload))

    with pytest.raises(ValueError, match="logprob positions"):
        collator.fetch_online_logprobs_vllm([list(INPUT_IDS)], [list(LABELS)])


def test_teacher_batch_size_mismatch_raises(rows):
    collator = make_collator()
    collator.http_session.post = MagicMock(
        return_value=fake_response(vllm_payload(rows))
    )

    with pytest.raises(ValueError, match="choices"):
        collator.fetch_online_logprobs_vllm(
            [list(INPUT_IDS), list(INPUT_IDS)], [list(LABELS), list(LABELS)]
        )


def test_missing_prompt_logprobs_raises():
    collator = make_collator()
    collator.http_session.post = MagicMock(
        return_value=fake_response({"choices": [{"text": ""}]})
    )

    with pytest.raises(ValueError, match="prompt_logprobs"):
        collator.fetch_online_logprobs_vllm([list(INPUT_IDS)], [list(LABELS)])


def test_features_without_input_ids_raise(rows):
    collator = make_collator()
    collator.http_session.post = MagicMock(
        return_value=fake_response(vllm_payload(rows))
    )

    with pytest.raises(ValueError, match="input_ids and labels"):
        collator._attach_teacher_logprobs([{"labels": list(LABELS)}])


def test_unpacked_features_get_teacher_targets(rows):
    """Without sample packing the collator sees a flat list of features, not sub-batches."""
    collator = make_collator()
    collator.http_session.post = MagicMock(
        return_value=fake_response(vllm_payload(rows))
    )

    features = [{"input_ids": list(INPUT_IDS), "labels": list(LABELS)}]
    collator._attach_teacher_logprobs(features)

    for field in ("target_token_ids", "target_logprobs", "target_mask"):
        assert len(features[0][field]) == SEQ_LEN
    assert_pinned(features[0]["target_token_ids"], features[0]["target_mask"])


def test_teacher_request_stats_are_tracked(rows):
    collator = make_collator()
    collator.http_session.post = MagicMock(
        return_value=fake_response(vllm_payload(rows))
    )

    collator._attach_teacher_logprobs(
        [{"input_ids": list(INPUT_IDS), "labels": list(LABELS)}]
    )

    assert collator._teacher_requests == 1
    assert collator._teacher_attempts == 1
    assert len(collator._latencies) == 1
    # 3 of the 5 rows are unmasked, all top-k slots valid
    assert collator._mask_slots_total == SEQ_LEN * TOP_K
    assert collator._mask_slots_valid == 3 * TOP_K


def test_offline_over_length_truncation_keeps_head(rows):
    """A sequence truncated to sequence_len keeps the head rows, and the row count matches."""
    long_rows = {i: teacher_row(INPUT_IDS[i]) for i in range(SEQ_LEN)}
    sample = offline_sample(long_rows, first_position=0)
    # more teacher rows than input tokens, as when the tokenized sequence was truncated
    sample["logprobs"] = sample["logprobs"] + [teacher_row_entries(999)] * 3

    result = make_offline_strategy().transform_logprobs(sample)

    assert len(result["target_logprobs"]) == SEQ_LEN
    assert len(result["target_token_ids"]) == SEQ_LEN
    assert len(result["target_mask"]) == SEQ_LEN
    assert_pinned(result["target_token_ids"], result["target_mask"])


def teacher_row_entries(token_id):
    return [
        {"logprob": lp, "token": f"token_id:{tok}"} for tok, lp in teacher_row(token_id)
    ]
