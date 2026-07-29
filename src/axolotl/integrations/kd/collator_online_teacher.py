"""
Packed data loader for online teacher training supporting vllm and sglang.
"""

import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import torch
from orjson import orjson

from axolotl.integrations.kd.collator import KDBatchSamplerDataCollatorForSeq2Seq
from axolotl.integrations.kd.utils import LOGPROB_PAD_VALUE, rescale_topk_logprobs
from axolotl.utils.data.utils import retry_on_request_exceptions
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class OnlineTeacherCollator(KDBatchSamplerDataCollatorForSeq2Seq):
    """
    Collator for online teacher training.

    Teacher rows follow the loss convention: row ``j`` is the teacher distribution over
    the token the student predicts at position ``j``, i.e. ``input_ids[j + 1]``.
    """

    DEFAULT_LABEL_PAD_TOKEN_ID: int = -100
    LOG_EVERY_N_REQUESTS: int = 50

    def __init__(
        self,
        *args: Any,
        kd_online_server_base_url: Optional[str] = None,
        kd_online_topk: Optional[int] = None,
        kd_temperature: Optional[float] = 1.0,
        kd_online_server: Optional[str] = "vllm",
        kd_online_timeout: Optional[int] = 120,
        kd_normalize_topk: Optional[bool] = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if kd_online_server_base_url is None:
            raise ValueError(
                "kd_online_server_base_url must be provided for OnlineTeacherDataloader"
            )
        if kd_online_topk is None or kd_online_topk <= 0:
            raise ValueError(
                "kd_online_topk must be a positive integer for OnlineTeacherDataloader"
            )

        self.kd_online_server_base_url = kd_online_server_base_url.rstrip("/")
        self.kd_online_topk = kd_online_topk
        self.kd_temperature = kd_temperature or 1.0
        self.kd_online_server = kd_online_server
        self.http_session = requests.Session()
        self.kd_online_timeout = kd_online_timeout
        self.kd_normalize_topk = kd_normalize_topk

        self._latencies: deque[float] = deque(maxlen=1024)
        self._teacher_attempts = 0
        self._teacher_requests = 0
        self._mask_slots_valid = 0
        self._mask_slots_total = 0

    def _padding_row(self) -> Tuple[List[float], List[int], List[int]]:
        return (
            [LOGPROB_PAD_VALUE] * self.kd_online_topk,
            [0] * self.kd_online_topk,
            [0] * self.kd_online_topk,
        )

    def _build_row(
        self,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        label: int,
    ) -> Tuple[List[float], List[int], List[int]]:
        """
        Build one (logprobs, token_ids, mask) target row: sorted by logprob descending,
        truncated to top-k, rescaled to the KD temperature, padded to top-k with masked
        slots.
        """
        candidates = sorted(
            zip(token_ids, logprobs, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )[: self.kd_online_topk]
        if not candidates:
            return self._padding_row()

        row_token_ids = [int(token_id) for token_id, _ in candidates]
        row_logprobs = rescale_topk_logprobs(
            torch.tensor([logprob for _, logprob in candidates], dtype=torch.float32),
            gen_temperature=1.0,
            kd_temperature=self.kd_temperature,
            normalize=bool(self.kd_normalize_topk),
        ).tolist()

        valid = 0 if label == self.DEFAULT_LABEL_PAD_TOKEN_ID else 1
        row_mask = [valid] * len(row_token_ids)

        pad_len = self.kd_online_topk - len(row_token_ids)
        if pad_len > 0:
            row_logprobs = row_logprobs + [LOGPROB_PAD_VALUE] * pad_len
            row_token_ids = row_token_ids + [0] * pad_len
            row_mask = row_mask + [0] * pad_len

        return row_logprobs, row_token_ids, row_mask

    def _post(self, api_endpoint: str, payload: Dict[str, Any]) -> requests.Response:
        self._teacher_attempts += 1
        headers = {"Accept-Encoding": "deflate, gzip, br, zstd"}
        start = time.perf_counter()
        response = self.http_session.post(
            api_endpoint,
            json=payload,
            headers=headers,
            timeout=self.kd_online_timeout,
        )
        self._latencies.append(time.perf_counter() - start)
        response.raise_for_status()
        self._teacher_requests += 1
        return response

    def _record_coverage(self, target_mask: List[List[List[int]]]) -> None:
        for sequence in target_mask:
            for row in sequence:
                self._mask_slots_valid += sum(row)
                self._mask_slots_total += len(row)

    def _maybe_log_stats(self) -> None:
        if (
            not self._teacher_requests
            or self._teacher_requests % self.LOG_EVERY_N_REQUESTS
        ):
            return
        latencies = sorted(self._latencies)
        p50 = latencies[int(0.50 * (len(latencies) - 1))]
        p99 = latencies[int(0.99 * (len(latencies) - 1))]
        coverage = self._mask_slots_valid / max(1, self._mask_slots_total)
        LOG.info(
            f"kd online teacher: requests={self._teacher_requests} "
            f"retries={self._teacher_attempts - self._teacher_requests} "
            f"latency_p50={p50:.3f}s latency_p99={p99:.3f}s "
            f"target_mask_coverage={coverage:.4f}"
        )

    @retry_on_request_exceptions(max_retries=10, delay=5, retry_client_errors=False)
    def fetch_online_logprobs_sglang(
        self, batch_input_ids: List[List[int]], labels: List[List[int]]
    ):
        """
        Fetches logprobs from an online teacher served by sglang for a batch of input_ids.
        """
        api_endpoint = f"{self.kd_online_server_base_url}/generate"

        payload = {
            "input_ids": batch_input_ids,
            "return_logprob": True,
            "top_logprobs_num": self.kd_online_topk,
            "logprob_start_len": 0,
            "sampling_params": {
                "max_new_tokens": 0,
            },
        }

        ret_data_target_token_ids: List[List[List[int]]] = []
        ret_data_target_logprobs: List[List[List[float]]] = []
        ret_data_target_mask: List[List[List[int]]] = []

        response = self._post(api_endpoint, payload)
        api_data = response.json()

        if not isinstance(api_data, list) or len(api_data) != len(batch_input_ids):
            raise ValueError(
                f"teacher returned {type(api_data).__name__} of length "
                f"{len(api_data) if isinstance(api_data, list) else 'N/A'}, "
                f"expected a list of {len(batch_input_ids)} sequences"
            )

        for sequence_data, seq_input_ids, seq_labels in zip(
            api_data, batch_input_ids, labels, strict=True
        ):
            meta_info = sequence_data.pop("meta_info", {})
            input_top_logprobs = meta_info.pop("input_top_logprobs", None)
            if not isinstance(input_top_logprobs, list):
                raise ValueError(
                    f"teacher response missing list-valued input_top_logprobs, got "
                    f"{type(input_top_logprobs).__name__}"
                )
            if len(input_top_logprobs) != len(seq_input_ids):
                raise ValueError(
                    f"teacher returned {len(input_top_logprobs)} logprob positions for "
                    f"{len(seq_input_ids)} input tokens"
                )

            seq_logprobs: List[List[float]] = []
            seq_token_ids: List[List[int]] = []
            seq_mask: List[List[int]] = []

            # input_top_logprobs[i] is the distribution over token i, which the student
            # predicts at position i - 1
            for i in range(1, len(seq_input_ids)):
                pos_data = input_top_logprobs[i]
                if not isinstance(pos_data, list) or not pos_data:
                    LOG.warning(f"Malformed logprob data at position {i}. Masking it.")
                    row = self._padding_row()
                else:
                    token_ids: List[int] = []
                    token_logprobs: List[float] = []
                    for entry in pos_data:
                        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                            continue
                        logprob, token_id = entry[0], entry[1]
                        if logprob is None or not math.isfinite(logprob):
                            continue
                        token_ids.append(int(token_id))
                        token_logprobs.append(float(logprob))
                    row = self._build_row(token_ids, token_logprobs, seq_labels[i])

                seq_logprobs.append(row[0])
                seq_token_ids.append(row[1])
                seq_mask.append(row[2])

            # the last position predicts a token outside this sequence
            tail = self._padding_row()
            seq_logprobs.append(tail[0])
            seq_token_ids.append(tail[1])
            seq_mask.append(tail[2])

            ret_data_target_token_ids.append(seq_token_ids)
            ret_data_target_logprobs.append(seq_logprobs)
            ret_data_target_mask.append(seq_mask)

        self._record_coverage(ret_data_target_mask)

        return {
            "target_token_ids": ret_data_target_token_ids,
            "target_logprobs": ret_data_target_logprobs,
            "target_mask": ret_data_target_mask,
        }

    @retry_on_request_exceptions(max_retries=10, delay=5, retry_client_errors=False)
    def fetch_online_logprobs_vllm(
        self, batch_input_ids: List[List[int]], labels: List[List[int]]
    ):
        """
        Fetches logprobs from an online teacher served by vllm for a batch of input_ids.
        """
        api_endpoint = f"{self.kd_online_server_base_url}/v1/completions"

        # prompt_logprobs is scored over the prompt itself; max_tokens=0 keeps the
        # teacher from generating anything on top of it
        payload = {
            "prompt": batch_input_ids,
            "max_tokens": 0,
            "echo": True,
            "prompt_logprobs": self.kd_online_topk,
        }

        ret_data_target_token_ids: List[List[List[int]]] = []
        ret_data_target_logprobs: List[List[List[float]]] = []
        ret_data_target_mask: List[List[List[int]]] = []

        response = self._post(api_endpoint, payload)
        api_data: dict = orjson.loads(response.content)
        choices = api_data.get("choices")

        if not isinstance(choices, list) or len(choices) != len(batch_input_ids):
            raise ValueError(
                f"teacher returned {len(choices) if isinstance(choices, list) else 'no'} "
                f"choices, expected {len(batch_input_ids)}"
            )

        for sequence_data, seq_input_ids, seq_labels in zip(
            choices, batch_input_ids, labels, strict=True
        ):
            prompt_logprobs = sequence_data.pop("prompt_logprobs", None)
            if not isinstance(prompt_logprobs, list):
                raise ValueError(
                    "teacher response is missing prompt_logprobs; the server must be an "
                    "OpenAI-compatible vllm serve with prompt_logprobs support"
                )
            if len(prompt_logprobs) != len(seq_input_ids):
                raise ValueError(
                    f"teacher returned {len(prompt_logprobs)} logprob positions for "
                    f"{len(seq_input_ids)} input tokens"
                )

            seq_logprobs: List[List[float]] = []
            seq_token_ids: List[List[int]] = []
            seq_mask: List[List[int]] = []

            # prompt_logprobs[i] is the distribution over token i (None for i == 0),
            # which the student predicts at position i - 1
            for i in range(1, len(seq_input_ids)):
                pos_data = prompt_logprobs[i]
                if not isinstance(pos_data, dict) or not pos_data:
                    LOG.warning(f"Malformed logprob data at position {i}. Masking it.")
                    row = self._padding_row()
                else:
                    token_ids = []
                    token_logprobs = []
                    for token_id, entry in pos_data.items():
                        logprob = (
                            entry.get("logprob") if isinstance(entry, dict) else None
                        )
                        if logprob is None or not math.isfinite(logprob):
                            continue
                        token_ids.append(int(token_id))
                        token_logprobs.append(float(logprob))
                    row = self._build_row(token_ids, token_logprobs, seq_labels[i])

                seq_logprobs.append(row[0])
                seq_token_ids.append(row[1])
                seq_mask.append(row[2])

            # the last position predicts a token outside this sequence
            tail = self._padding_row()
            seq_logprobs.append(tail[0])
            seq_token_ids.append(tail[1])
            seq_mask.append(tail[2])

            ret_data_target_token_ids.append(seq_token_ids)
            ret_data_target_logprobs.append(seq_logprobs)
            ret_data_target_mask.append(seq_mask)

        self._record_coverage(ret_data_target_mask)

        return {
            "target_token_ids": ret_data_target_token_ids,
            "target_logprobs": ret_data_target_logprobs,
            "target_mask": ret_data_target_mask,
        }

    def _attach_teacher_logprobs(self, items: List[Dict[str, Any]]) -> None:
        input_ids_for_api_call: List[List[int]] = []
        labels_for_api_call: List[List[int]] = []

        for item_dict in items:
            if not isinstance(item_dict, dict):
                raise TypeError(
                    f"OnlineTeacherCollator expects dict features, got {type(item_dict).__name__}"
                )

            current_input_ids = item_dict.get("input_ids")
            current_labels = item_dict.get("labels")
            if current_input_ids is None or current_labels is None:
                raise ValueError(
                    "OnlineTeacherCollator requires both input_ids and labels on every "
                    "feature to request teacher logprobs"
                )

            input_ids_for_api_call.append(
                current_input_ids.tolist()
                if hasattr(current_input_ids, "tolist")
                else list(current_input_ids)
            )
            labels_for_api_call.append(
                current_labels.tolist()
                if hasattr(current_labels, "tolist")
                else list(current_labels)
            )

        if self.kd_online_server == "sglang":
            api_response = self.fetch_online_logprobs_sglang(
                input_ids_for_api_call, labels_for_api_call
            )
        else:
            api_response = self.fetch_online_logprobs_vllm(
                input_ids_for_api_call, labels_for_api_call
            )

        if len(api_response["target_token_ids"]) != len(items):
            raise ValueError(
                f"teacher returned {len(api_response['target_token_ids'])} sequences for "
                f"{len(items)} features"
            )

        for i, item_to_update in enumerate(items):
            expected_len = len(input_ids_for_api_call[i])
            for field in ("target_token_ids", "target_logprobs", "target_mask"):
                rows = api_response[field][i]
                if len(rows) != expected_len:
                    raise ValueError(
                        f"teacher {field} has {len(rows)} rows for {expected_len} input tokens"
                    )
                item_to_update[field] = rows

        self._maybe_log_stats()

    def __call__(
        self, features: List[Any], return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        if not features:
            return super().__call__(features, return_tensors=return_tensors)

        # features is either a list of packed sub-batches or a flat list of features
        sub_batches = features if isinstance(features[0], list) else [features]

        for sub_batch_features in sub_batches:
            if not sub_batch_features:
                continue
            self._attach_teacher_logprobs(sub_batch_features)

        return super().__call__(features, return_tensors=return_tensors)
