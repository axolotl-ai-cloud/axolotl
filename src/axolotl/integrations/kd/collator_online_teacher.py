"""
Packed data loader for online teacher training supporting vllm and sglang.
"""

import hashlib
import hmac
import logging
from typing import Any, Dict, List, Optional

import requests
import torch
from orjson import orjson

from axolotl.integrations.kd.collator import KDBatchSamplerDataCollatorForSeq2Seq
from axolotl.integrations.kd.utils import normalize_logprobs
from axolotl.utils.data.utils import retry_on_request_exceptions

LOG = logging.getLogger(__name__)


def hmac_sha_from_int_list(int_list, key, hash_func=hashlib.sha256):
    """
    Create HMAC-SHA hash from a list of integers

    Args:
        int_list: List of integers
        key: Secret key (string or bytes)
        hash_func: Hash function (default: sha256)

    Returns:
        HMAC digest as hex string
    """
    # Convert key to bytes if it's a string
    if isinstance(key, str):
        key = key.encode("utf-8")

    # Convert list of ints to bytes
    # Method 1: Convert each int to bytes and concatenate
    data = b"".join(i.to_bytes(4, byteorder="big") for i in int_list)

    # Create HMAC
    h = hmac.new(key, data, hash_func)
    return h.hexdigest()


class OnlineTeacherCollator(KDBatchSamplerDataCollatorForSeq2Seq):
    """
    Collator for online teacher training.
    """

    DEFAULT_LABEL_PAD_TOKEN_ID: int = -100

    def __init__(
        self,
        *args: Any,
        kd_online_server_base_url: Optional[str] = None,
        kd_online_topk: Optional[int] = None,
        kd_temperature: Optional[float] = 1.0,
        kd_online_server: Optional[str] = "vllm",
        kd_online_timeout: Optional[int] = 120,
        kd_cache_dir: Optional[str] = None,
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
        self.kd_temperature = kd_temperature
        self.kd_online_server = kd_online_server
        self.http_session = requests.Session()
        self.kd_online_timeout = kd_online_timeout
        self.kd_cache_dir = kd_cache_dir
        self.kd_normalize_topk = kd_normalize_topk

    def _normalize_logprobs(self, raw_logprobs: List[float]) -> List[float]:
        """
        Re-normalizes top-k raw logprobs as probabilities, and converts back to logprobs.
        """
        if not raw_logprobs or self.kd_online_topk == 0:
            return (
                [-float("inf")] * self.kd_online_topk if self.kd_online_topk > 0 else []
            )

        raw_logprobs_tensor = torch.tensor(raw_logprobs, dtype=torch.float32)
        return normalize_logprobs(raw_logprobs_tensor, self.kd_online_topk).tolist()

    @retry_on_request_exceptions(max_retries=10, delay=5)
    def fetch_online_logprobs_sglang(
        self, batch_input_ids: List[List[int]], labels: List[List[int]]
    ):
        """
        Fetches logprobs from an online teacher served by sglang for a batch of input_ids.
        Assumes API returns token IDs as strings in logprob dictionary keys.
        """
        api_endpoint = f"{self.kd_online_server_base_url}/generate"

        payload = {
            "input_ids": batch_input_ids,
            "return_logprob": True,
            "top_logprobs_num": self.kd_online_topk,
            "logprob_start_len": 0,
            "return_text_in_logprobs": True,
            "echo": True,
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": self.kd_temperature,
                "skip_special_tokens": False,
            },
        }

        # Initialize with empty lists, so if API call fails, these are returned.
        ret_data_target_token_ids: List[List[List[int]]] = []
        ret_data_target_logprobs: List[List[List[float]]] = []
        ret_data_target_mask: List[List[List[int]]] = []

        try:
            response = self.http_session.post(
                api_endpoint, json=payload, timeout=self.kd_online_timeout
            )
            response.raise_for_status()
            api_data: list[dict] = response.json()

            # Ensure api_data is a list, and its length matches batch_input_ids
            if not isinstance(api_data, list) or len(api_data) != len(batch_input_ids):
                LOG.error(
                    f"API response format error. Expected a list of {len(batch_input_ids)} "
                    f"items, got {type(api_data)} with length {len(api_data) if isinstance(api_data, list) else 'N/A'}."
                )
                # Return empty data; items processed later will get default empty KD fields
                return {
                    "target_token_ids": ret_data_target_token_ids,
                    "target_logprobs": ret_data_target_logprobs,
                    "target_mask": ret_data_target_mask,
                }

            for sequence_data, seq_input_ids, seq_labels in zip(
                api_data, batch_input_ids, labels, strict=False
            ):
                current_target_logprobs = []
                current_target_token_ids = []
                current_target_mask = []

                meta_info = sequence_data.pop("meta_info", {})
                # Ensure input_top_logprobs is a list
                input_top_logprobs: Optional[list[None | list[tuple]]] = meta_info.pop(
                    "input_top_logprobs", []
                )
                if not isinstance(input_top_logprobs, list):
                    LOG.warning(
                        f"Received non-list input_top_logprobs: {input_top_logprobs}. Skipping sequence."
                    )
                    input_top_logprobs = []  # Treat as empty

                # basic check that the logprob data len matches the input len, so no need to handle padding
                assert len(seq_input_ids) == len(input_top_logprobs)

                for i, _, label in zip(
                    range(len(seq_input_ids)), seq_input_ids, seq_labels, strict=False
                ):
                    if i < len(input_top_logprobs) and input_top_logprobs[i] is None:
                        # this is always the case for the first token.
                        # there is never logprob data for the first token since that's a true input
                        # so we replace the None value with padding data
                        current_target_logprobs.append(
                            [-float("inf")] * self.kd_online_topk
                        )
                        current_target_token_ids.append([0] * self.kd_online_topk)
                        current_target_mask.append([0] * self.kd_online_topk)
                    elif (
                        i < len(input_top_logprobs)
                        and input_top_logprobs[i] is not None
                    ):
                        pos_top_logprobs_data = input_top_logprobs[i]
                        # Ensure pos_top_logprobs_data is a list of lists as expected
                        if not (
                            isinstance(pos_top_logprobs_data, list)
                            and all(
                                isinstance(item, list) for item in pos_top_logprobs_data
                            )
                            and len(pos_top_logprobs_data) > 0
                            and len(pos_top_logprobs_data[0]) == 3
                        ):  # [logprob, token_id, token_str]
                            LOG.warning(
                                f"Malformed pos_top_logprobs_data: {pos_top_logprobs_data}. Padding this position."
                            )
                            current_target_logprobs.append(
                                [-float("inf")] * self.kd_online_topk
                            )
                            current_target_token_ids.append([0] * self.kd_online_topk)
                            current_target_mask.append([0] * self.kd_online_topk)
                            continue

                        # pos_top_logprobs: list of logprobs, pos_token_ids: list of token_ids
                        pos_logprobs_raw, pos_token_ids, _ = [
                            list(row)
                            for row in zip(*pos_top_logprobs_data, strict=False)
                        ]

                        # Ensure correct length (top_k)
                        if len(pos_logprobs_raw) < self.kd_online_topk:
                            pad_len = self.kd_online_topk - len(pos_logprobs_raw)
                            pos_logprobs_raw.extend([-float("inf")] * pad_len)
                            pos_token_ids.extend([0] * pad_len)  # Pad with 0 token_id

                        # truncate to top_k in case the response was longer
                        current_target_token_ids.append(
                            pos_token_ids[: self.kd_online_topk]
                        )

                        if self.kd_normalize_topk:
                            normalized_logprobs_for_position = self._normalize_logprobs(
                                pos_logprobs_raw[: self.kd_online_topk]
                            )
                            current_target_logprobs.append(
                                normalized_logprobs_for_position
                            )
                        else:
                            current_target_logprobs.append(
                                pos_logprobs_raw[: self.kd_online_topk]
                            )

                        # Mask depends on the corresponding label for the student
                        if label == self.DEFAULT_LABEL_PAD_TOKEN_ID:
                            current_target_mask.append([0] * self.kd_online_topk)
                        else:
                            current_target_mask.append([1] * self.kd_online_topk)
                    else:
                        # Pad if no logprobs for this position (either due to length mismatch or None entry)
                        current_target_logprobs.append(
                            [-float("inf")] * self.kd_online_topk
                        )
                        current_target_token_ids.append([0] * self.kd_online_topk)
                        current_target_mask.append([0] * self.kd_online_topk)

                ret_data_target_token_ids.append(current_target_token_ids)
                ret_data_target_logprobs.append(current_target_logprobs)
                ret_data_target_mask.append(current_target_mask)

        except requests.exceptions.RequestException as e:
            LOG.error(f"Error fetching logprobs from online teacher: {e}")
            raise e
            # ret_logprobs_data will be returned with empty lists, handled by the caller.
        except Exception as e:  # Catch other potential errors during processing
            LOG.error(
                f"Unexpected error processing API response in fetch_online_logprobs: {e}",
                exc_info=True,
            )
            raise e

        return {
            "target_token_ids": ret_data_target_token_ids,
            "target_logprobs": ret_data_target_logprobs,
            "target_mask": ret_data_target_mask,
        }

    @retry_on_request_exceptions(max_retries=10, delay=5)
    def fetch_online_logprobs_vllm(
        self, batch_input_ids: List[List[int]], labels: List[List[int]]
    ):
        """
        Fetches logprobs from an online teacher served by vllm for a batch of input_ids.
        Assumes API returns token IDs as strings in logprob dictionary keys.
        """
        api_endpoint = f"{self.kd_online_server_base_url}/v1/completions"

        payload = {
            "prompt": batch_input_ids,
            "echo": True,
            "logprobs": True,
            "prompt_logprobs": self.kd_online_topk,
            "top_logprobs": self.kd_online_topk,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
            "temperature": self.kd_temperature,
            "sampling_params": {
                "max_tokens": 0,
            },
        }

        # Initialize with empty lists, so if API call fails, these are returned.
        ret_data_target_token_ids: List[List[List[int]]] = []
        ret_data_target_logprobs: List[List[List[float]]] = []
        ret_data_target_mask: List[List[List[int]]] = []

        try:
            headers = {"Accept-Encoding": "deflate, gzip, br, zstd"}
            response = self.http_session.post(
                api_endpoint,
                json=payload,
                headers=headers,
                timeout=self.kd_online_timeout,
            )
            response.raise_for_status()
            api_data: dict = orjson.loads(response.content)
            choices: list[dict] = api_data["choices"]

            # Ensure api_data is a list, and its length matches batch_input_ids
            if not isinstance(choices, list) or len(choices) != len(batch_input_ids):
                LOG.error(
                    f"API response format error. Expected a list of {len(batch_input_ids)} "
                    f"items, got {type(api_data)} with length {len(api_data) if isinstance(api_data, list) else 'N/A'}."
                )
                # Return empty data; items processed later will get default empty KD fields
                return {
                    "target_token_ids": ret_data_target_token_ids,
                    "target_logprobs": ret_data_target_logprobs,
                    "target_mask": ret_data_target_mask,
                }

            for sequence_data, seq_input_ids, seq_labels in zip(
                choices, batch_input_ids, labels, strict=False
            ):
                # seq_input_ids: List[int]
                # seq_labels: List[int]

                current_target_logprobs = []
                current_target_token_ids = []
                current_target_mask = []

                # Ensure input_top_logprobs is a list
                input_top_logprobs: Optional[list[None | dict[str, dict]]] = (
                    sequence_data.pop("prompt_logprobs", [])
                )

                if not isinstance(input_top_logprobs, list):
                    LOG.warning(
                        f"Received non-list input_top_logprobs: {input_top_logprobs}. Skipping sequence."
                    )
                    input_top_logprobs = []  # Treat as empty

                # basic check that the logprob data len matches the input len, so no need to handle padding
                assert len(seq_input_ids) == len(input_top_logprobs)

                seq_len = len(seq_input_ids)

                for i, _, label in zip(
                    range(seq_len), seq_input_ids, seq_labels, strict=False
                ):
                    if i < len(input_top_logprobs) and input_top_logprobs[i] is None:
                        # this is always the case for the first token.
                        # there is never logprob data for the first token since that's a true input
                        continue
                    if (
                        i < len(input_top_logprobs)
                        and input_top_logprobs[i] is not None
                    ):
                        pos_top_logprobs_data: dict[str, dict] = input_top_logprobs[i]  # type: ignore[assignment]
                        # Ensure pos_top_logprobs_data is a list of lists as expected
                        if not (
                            isinstance(pos_top_logprobs_data, dict)
                            and all(
                                isinstance(item, dict)
                                for item in pos_top_logprobs_data.values()
                            )
                            and len(pos_top_logprobs_data.keys()) > 0
                        ):  # [logprob, token_id, token_str]
                            LOG.warning(
                                f"Malformed pos_top_logprobs_data: {pos_top_logprobs_data}. Padding this position."
                            )
                            current_target_logprobs.append(
                                [-float("inf")] * self.kd_online_topk
                            )
                            current_target_token_ids.append(
                                list(range(self.kd_online_topk))
                            )
                            current_target_mask.append([0] * self.kd_online_topk)
                            continue

                        # pos_top_logprobs: list of logprobs, pos_token_ids: list of token_ids
                        pos_token_ids_str = list(pos_top_logprobs_data.keys())
                        pos_logprobs_dict = pos_top_logprobs_data.values()
                        pos_token_ids = [
                            int(token_id) for token_id in pos_token_ids_str
                        ]
                        pos_logprobs_raw = [
                            float(logprob.get("logprob", -float("inf")))
                            for logprob in pos_logprobs_dict
                        ]

                        # Ensure correct length (top_k)
                        if len(pos_logprobs_raw) < self.kd_online_topk:
                            pad_len = self.kd_online_topk - len(pos_logprobs_raw)
                            LOG.warning(
                                f"Padding position {i} with {pad_len} top-k tokens and logprobs."
                            )
                            pos_logprobs_raw.extend([-float("inf")] * pad_len)
                            pos_token_ids.extend([0] * pad_len)  # Pad with 0 token_id

                        # truncate to top_k in case the response was longer
                        current_target_token_ids.append(
                            pos_token_ids[: self.kd_online_topk]
                        )

                        if self.kd_normalize_topk:
                            normalized_logprobs_for_position = self._normalize_logprobs(
                                pos_logprobs_raw[: self.kd_online_topk]
                            )
                            current_target_logprobs.append(
                                normalized_logprobs_for_position
                            )
                        else:
                            current_target_logprobs.append(
                                pos_logprobs_raw[: self.kd_online_topk]
                            )

                        # Mask depends on the corresponding label for the student
                        if label == self.DEFAULT_LABEL_PAD_TOKEN_ID:
                            current_target_mask.append([0] * self.kd_online_topk)
                        else:
                            current_target_mask.append([1] * self.kd_online_topk)
                    else:
                        # Pad if no logprobs for this position (either due to length mismatch or None entry)
                        current_target_logprobs.append(
                            [-float("inf")] * self.kd_online_topk
                        )
                        current_target_token_ids.append(
                            list(range(self.kd_online_topk))
                        )
                        current_target_mask.append([0] * self.kd_online_topk)
                for _ in range(max(0, seq_len - len(current_target_logprobs))):
                    current_target_logprobs.append(
                        [-float("inf")] * self.kd_online_topk
                    )
                    current_target_token_ids.append(list(range(self.kd_online_topk)))
                    current_target_mask.append([0] * self.kd_online_topk)

                ret_data_target_token_ids.append(current_target_token_ids)
                ret_data_target_logprobs.append(current_target_logprobs)
                ret_data_target_mask.append(current_target_mask)

                # TODO save and load targets to disk for caching for next epoch
                # generate a hmac SHA256 hash over the list seq_input_ids and convert it to an int
                # if self.kd_cache_dir:
                #     hash_input_ids = hmac_sha_from_int_list(
                #         seq_input_ids, f"{self.kd_online_server_base_url}:{self.kd_online_topk}"
                #     )
                #     with open(f"{self.kd_cache_dir}/{hash_input_ids}.parquet", "wb") as f:
                #         pd.DataFrame(ret_logprobs_data).to_parquet(f, index=False)

        except requests.exceptions.RequestException as e:
            LOG.error(f"Error fetching logprobs from online teacher: {e}")
            raise e
            # ret_logprobs_data will be returned with empty lists, handled by the caller.
        except Exception as e:  # Catch other potential errors during processing
            LOG.error(
                f"Unexpected error processing API response in fetch_online_logprobs: {e}",
                exc_info=True,
            )
            raise e

        return {
            "target_token_ids": ret_data_target_token_ids,
            "target_logprobs": ret_data_target_logprobs,
            "target_mask": ret_data_target_mask,
        }

    def __call__(
        self, features: List[List[Dict[str, Any]]], return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        if not features:
            return super().__call__(features, return_tensors=return_tensors)

        for (
            sub_batch_features
        ) in features:  # sub_batch_features is List[Dict[str, Any]]
            if not sub_batch_features:
                continue

            input_ids_for_api_call: List[List[int]] = []
            labels_for_api_call: List[List[int]] = []
            # Store references to the original item dictionaries to update them in-place
            items_for_api_call: List[Dict[str, Any]] = []

            for item_dict in sub_batch_features:
                if not isinstance(item_dict, dict):
                    LOG.warning(
                        f"Skipping non-dict item in sub_batch_features: {item_dict}"
                    )
                    continue

                current_input_ids = item_dict.get("input_ids")
                current_labels = item_dict.get("labels")

                if current_input_ids is not None and current_labels is not None:
                    # Ensure input_ids and labels are lists of ints for JSON serialization
                    input_ids_list = (
                        current_input_ids.tolist()
                        if hasattr(current_input_ids, "tolist")
                        else list(current_input_ids)
                    )
                    labels_list = (
                        current_labels.tolist()
                        if hasattr(current_labels, "tolist")
                        else list(current_labels)
                    )

                    input_ids_for_api_call.append(input_ids_list)
                    labels_for_api_call.append(labels_list)
                    items_for_api_call.append(item_dict)
                else:
                    # This item will not get teacher logprobs from the API.
                    # Initialize KD fields to empty lists so downstream collators handle them uniformly.
                    item_dict.setdefault("target_token_ids", [])
                    item_dict.setdefault("target_logprobs", [])
                    item_dict.setdefault("target_mask", [])

            # print(items_for_api_call)
            if items_for_api_call:  # Only call API if there's something to process
                if self.kd_online_server == "sglang":
                    api_responses_for_sub_batch = self.fetch_online_logprobs_sglang(
                        input_ids_for_api_call, labels_for_api_call
                    )
                else:
                    api_responses_for_sub_batch = self.fetch_online_logprobs_vllm(
                        input_ids_for_api_call, labels_for_api_call
                    )

                # api_responses_for_sub_batch has keys: "target_token_ids", "target_logprobs", "target_mask"
                # Each value is a list, corresponding to items_for_api_call
                for i, item_to_update in enumerate(items_for_api_call):
                    # TODO make sure to figure out which input in sub_batch_features to update the batch in the original `features` object so the super class can handle it properly.
                    if api_responses_for_sub_batch and i < len(
                        api_responses_for_sub_batch["target_token_ids"]
                    ):  # Check bounds
                        assert len(
                            api_responses_for_sub_batch["target_token_ids"][i]
                        ) == len(item_to_update["input_ids"])
                        assert len(
                            api_responses_for_sub_batch["target_logprobs"][i]
                        ) == len(item_to_update["input_ids"])
                        assert len(
                            api_responses_for_sub_batch["target_mask"][i]
                        ) == len(item_to_update["labels"])
                        item_to_update["target_token_ids"] = (
                            api_responses_for_sub_batch["target_token_ids"][i]
                        )
                        item_to_update["target_logprobs"] = api_responses_for_sub_batch[
                            "target_logprobs"
                        ][i]
                        item_to_update["target_mask"] = api_responses_for_sub_batch[
                            "target_mask"
                        ][i]
                    else:
                        # API call failed for this item, or response was shorter than expected.
                        # Ensure KD fields are initialized as empty lists.
                        LOG.warning(
                            f" (index {i}), or API response was too short. "
                            f"API response keys: {list(api_responses_for_sub_batch.keys()) if api_responses_for_sub_batch else 'None'}"
                        )
                        item_to_update.setdefault("target_token_ids", [])
                        item_to_update.setdefault("target_logprobs", [])
                        item_to_update.setdefault("target_mask", [])

        return super().__call__(features, return_tensors=return_tensors)
