"""Decision manifest compatibility and runtime-cost coverage."""

import json
from types import SimpleNamespace

from axolotl.integrations.decision.manifest import (
    DecisionManifest,
    build_decision_manifest,
)
from axolotl.utils.dict import DictDefault


class _ModelWithoutParameterTraversal:
    config = SimpleNamespace(mask_token_id=100, _commit_hash="resolved-revision")

    def named_parameters(self):
        raise AssertionError("manifest generation must not traverse model parameters")


def _manifest_config():
    return DictDefault(
        {
            "base_model": "nvidia/Nemotron-Labs-Diffusion-3B",
            "revision_of_model": "requested-revision",
            "model_config_type": "nemotron_labs_diffusion",
            "diffusion": {"canvas_width": 128, "mask_token_id": 100},
            "decision": {
                "layout": "thought_block",
                "reader": "hf",
                "latent": {"mode": "none"},
            },
        }
    )


def test_manifest_build_does_not_traverse_model_parameters():
    manifest = build_decision_manifest(
        _manifest_config(), model=_ModelWithoutParameterTraversal()
    )

    assert manifest.initial_common_adapter_sha256 is None
    assert manifest.initial_common_adapter_fingerprint_reason is None
    assert manifest.initial_common_adapter_devices == []
    assert manifest.initial_common_adapter_dtypes == []
    assert manifest.initial_common_adapter_tensor_count == 0


def test_manifest_parses_legacy_initial_adapter_fingerprint_fields():
    manifest = build_decision_manifest(_manifest_config())
    legacy = manifest.model_dump(mode="json") | {
        "initial_common_adapter_sha256": "a" * 64,
        "initial_common_adapter_fingerprint_reason": None,
        "initial_common_adapter_devices": ["cuda:0"],
        "initial_common_adapter_dtypes": ["torch.bfloat16"],
        "initial_common_adapter_tensor_count": 2,
    }

    restored = DecisionManifest.model_validate_json(json.dumps(legacy))

    assert restored.initial_common_adapter_sha256 == "a" * 64
    assert restored.initial_common_adapter_devices == ["cuda:0"]
    assert restored.initial_common_adapter_dtypes == ["torch.bfloat16"]
    assert restored.initial_common_adapter_tensor_count == 2
