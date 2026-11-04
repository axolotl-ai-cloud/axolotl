"""Removed checkpointing keys and the rules local to the activation_checkpointing block."""

import re

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.activation_checkpointing import (
    ActivationCheckpointingConfig,
    removed_checkpointing_keys_message,
)

SEE = (
    " See docs/activation_checkpointing.qmd#sec-migrate-activation-checkpointing for "
    "the migration guide."
)
DISK_NOTE = (
    " Disk offload is disabled pending a fix; `activation_offloading: disk` never "
    "wrote to disk (it ran the same CPU offloader as `true`), so the replacement "
    "offloads to CPU."
)

REMOVED_KEYS = [
    pytest.param(
        {"gradient_checkpointing": True},
        "`gradient_checkpointing` was removed. Use `activation_checkpointing: true` "
        "instead.",
        id="enabled",
    ),
    pytest.param(
        {"gradient_checkpointing": False},
        "`gradient_checkpointing` was removed. Use `activation_checkpointing: false` "
        "instead.",
        id="disabled",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        },
        "`gradient_checkpointing` and `gradient_checkpointing_kwargs` were removed. "
        "Use `activation_checkpointing: {enabled: true, reentrant: false}` instead.",
        id="kwargs",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {
                "use_reentrant": True,
                "every_n_layers": 2,
                "preserve_rng_state": False,
            },
        },
        "`gradient_checkpointing` and `gradient_checkpointing_kwargs` were removed. "
        "Use `activation_checkpointing: {enabled: true, reentrant: true, kwargs: "
        "{every_n_layers: 2, preserve_rng_state: false}}` instead.",
        id="passthrough_kwargs",
    ),
    pytest.param(
        {"gradient_checkpointing": "offload"},
        "`gradient_checkpointing` was removed. Use `activation_checkpointing: "
        "{enabled: true, offload: {all: cpu}}` instead.",
        id="offload_string",
    ),
    pytest.param(
        {"gradient_checkpointing": "offload_disk"},
        "`gradient_checkpointing` was removed. Use `activation_checkpointing: "
        "{enabled: true, offload: {all: cpu}}` instead." + DISK_NOTE,
        id="offload_disk_string",
    ),
    pytest.param(
        {"gradient_checkpointing_kwargs": {"use_reentrant": True, "every_n_layers": 2}},
        "`gradient_checkpointing_kwargs` was removed. Delete it: checkpointing is "
        "off, so it had no effect.",
        id="kwargs_without_checkpointing",
    ),
    pytest.param(
        {
            "gradient_checkpointing": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        },
        "`gradient_checkpointing` and `gradient_checkpointing_kwargs` were removed. "
        "Use `activation_checkpointing: false` instead. "
        "`gradient_checkpointing_kwargs` had no effect with checkpointing off.",
        id="kwargs_with_checkpointing_off",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"offload": True},
        },
        "`gradient_checkpointing` and `gradient_checkpointing_kwargs` were removed. "
        "Use `activation_checkpointing: {enabled: true, offload: {hidden_states: "
        "cpu}}` instead. `offload` in the checkpoint kwargs copied each checkpointed "
        "layer's input to CPU synchronously; `offload.hidden_states: cpu` offloads "
        "the same tensors with stream-overlapped copies.",
        id="kwargs_offload",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {
                "use_reentrant": False,
                "respect_saved_tensors_hooks": True,
            },
        },
        "`gradient_checkpointing` and `gradient_checkpointing_kwargs` were removed. "
        "Use `activation_checkpointing: {enabled: true, reentrant: false}` instead. "
        "`respect_saved_tensors_hooks` has no replacement: axolotl sets it.",
        id="kwargs_owned",
    ),
    pytest.param(
        {"gradient_checkpointing": True, "selective_checkpointing": True},
        "`gradient_checkpointing` and `selective_checkpointing` were removed. Use "
        "`activation_checkpointing: {enabled: true, selective: true}` instead.",
        id="selective",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "selective_checkpointing": {"save": ["attention"], "offload": True},
        },
        "`gradient_checkpointing` and `selective_checkpointing` were removed. Use "
        "`activation_checkpointing: {enabled: true, selective: {save: [attention]}, "
        "offload: {saves: cpu}}` instead.",
        id="selective_offload",
    ),
    pytest.param(
        {"selective_checkpointing": True},
        "`selective_checkpointing` was removed. Use `activation_checkpointing: "
        "{enabled: true, selective: true}` instead. `selective_checkpointing` "
        "requires checkpointing, so the replacement turns it on.",
        id="selective_without_checkpointing",
    ),
    pytest.param(
        {"gradient_checkpointing": True, "activation_offloading": "hidden_states"},
        "`gradient_checkpointing` and `activation_offloading` were removed. Use "
        "`activation_checkpointing: {enabled: true, offload: {hidden_states: cpu}}` "
        "instead.",
        id="hidden_states",
    ),
    pytest.param(
        {"activation_offloading": True},
        "`activation_offloading` was removed. Use `activation_checkpointing: "
        "{enabled: false, offload: {all: cpu}}` instead.",
        id="offload_without_recompute",
    ),
    pytest.param(
        {"gradient_checkpointing": True, "activation_offloading": "legacy"},
        "`gradient_checkpointing` and `activation_offloading` were removed. Use "
        "`activation_checkpointing: {enabled: true, offload: {all: cpu}}` instead.",
        id="offload_legacy",
    ),
    pytest.param(
        {"gradient_checkpointing": True, "activation_offloading": "disk"},
        "`gradient_checkpointing` and `activation_offloading` were removed. Use "
        "`activation_checkpointing: {enabled: true, offload: {all: cpu}}` instead."
        + DISK_NOTE,
        id="offload_disk",
    ),
    pytest.param(
        {"activation_offloading": False},
        "`activation_offloading` was removed. Delete it: activation offloading is "
        "off by default.",
        id="offload_off",
    ),
    pytest.param(
        {"fsdp_config": {"activation_checkpointing": True}},
        "`fsdp_config.activation_checkpointing` was removed. Use "
        "`activation_checkpointing: {enabled: true, implementation: fsdp}` instead.",
        id="fsdp",
    ),
    pytest.param(
        {"fsdp_config": {"fsdp_activation_checkpointing": True}},
        "`fsdp_config.fsdp_activation_checkpointing` was removed. Use "
        "`activation_checkpointing: {enabled: true, implementation: fsdp}` instead.",
        id="fsdp_prefixed",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "fsdp_config": {"activation_checkpointing": True},
        },
        "`gradient_checkpointing` and `fsdp_config.activation_checkpointing` were "
        "removed. Use `activation_checkpointing: true` instead. transformers' "
        "Trainer rejects both at once, so this config failed at startup. "
        "`{enabled: true, implementation: fsdp}` uses FSDP's checkpoint wrapper "
        "instead.",
        id="hf_and_fsdp",
    ),
    pytest.param(
        {
            "gradient_checkpointing_kwargs": {"use_reentrant": True},
            "fsdp_config": {"activation_checkpointing": True},
        },
        "`gradient_checkpointing_kwargs` and `fsdp_config.activation_checkpointing` "
        "were removed. Use `activation_checkpointing: {enabled: true, "
        "implementation: fsdp}` instead. FSDP's checkpoint wrapper ignored "
        "`gradient_checkpointing_kwargs`, so `use_reentrant` had no effect.",
        id="fsdp_ignored_kwargs",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "activation_checkpointing": {"reentrant": False},
        },
        "`gradient_checkpointing` was removed. Merge `activation_checkpointing: "
        "true` into your `activation_checkpointing` block and delete it.",
        id="alongside_the_block",
    ),
]


@pytest.mark.parametrize(("legacy", "message"), REMOVED_KEYS)
def test_removed_key_message(legacy, message):
    assert removed_checkpointing_keys_message(legacy) == message + SEE


@pytest.mark.parametrize(
    "legacy",
    [
        {"gradient_checkpointing": True},
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        {"selective_checkpointing": True},
        {"activation_offloading": True},
        {"fsdp_version": 2, "fsdp_config": {"activation_checkpointing": True}},
        {"fsdp_version": 2, "fsdp_config": {"fsdp_activation_checkpointing": True}},
    ],
    ids=lambda legacy: "-".join(legacy),
)
def test_validation_rejects_removed_key(min_base_cfg, legacy):
    with pytest.raises(ValueError, match="was removed"):
        validate_config(min_base_cfg | DictDefault(legacy))


def test_no_legacy_keys_no_message():
    assert (
        removed_checkpointing_keys_message({"activation_checkpointing": True}) is None
    )


def _rejects(block, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        ActivationCheckpointingConfig.model_validate(block)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        (
            "context_fn",
            "`activation_checkpointing.kwargs.context_fn` is set by axolotl: "
            "selective checkpointing injects its own. Use "
            "`activation_checkpointing.selective` to choose what is saved.",
        ),
        (
            "respect_saved_tensors_hooks",
            "`activation_checkpointing.kwargs.respect_saved_tensors_hooks` is set by "
            "axolotl so selective checkpointing composes with activation offloading. "
            "Remove it.",
        ),
        (
            "offload",
            "`activation_checkpointing.kwargs.offload` is not supported: it copies "
            "each checkpointed layer's input to CPU synchronously. Use "
            "`activation_checkpointing.offload.hidden_states: cpu`, which offloads "
            "the same tensors with stream-overlapped copies.",
        ),
    ],
)
def test_rejects_owned_kwargs(key, message):
    _rejects({"kwargs": {key: True}}, message)


def test_rejects_unknown_kwargs():
    _rejects(
        {"kwargs": {"preserve_rng": False}},
        "`activation_checkpointing.kwargs.preserve_rng` is not a "
        "torch.utils.checkpoint.checkpoint argument or `every_n_layers` (accepted: "
        "preserve_rng_state, determinism_check, debug, early_stop, every_n_layers). "
        "torch would pass it to each checkpointed layer's forward, where it is "
        "ignored or overrides the layer's own argument.",
    )


def test_kwargs_use_reentrant_conflict():
    _rejects(
        {"reentrant": False, "kwargs": {"use_reentrant": True}},
        "`activation_checkpointing.kwargs.use_reentrant` (true) contradicts "
        "`activation_checkpointing.reentrant` (false). Set only `reentrant`.",
    )


def test_kwargs_use_reentrant_folds_into_reentrant():
    block = ActivationCheckpointingConfig.model_validate(
        {"kwargs": {"use_reentrant": True, "preserve_rng_state": False}}
    )
    assert block.reentrant is True
    assert block.kwargs == {"preserve_rng_state": False}


@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_every_n_layers_must_be_positive_int(value):
    _rejects(
        {"kwargs": {"every_n_layers": value}},
        "`activation_checkpointing.kwargs.every_n_layers` must be an integer >= 1",
    )


def test_disk_rejected_for_every_target():
    _rejects(
        {"selective": True, "offload": {"saves": "disk", "hidden_states": "disk"}},
        "activation_checkpointing.offload.hidden_states, saves: disk offload is "
        "disabled pending a fix; use `cpu`.",
    )


def test_disk_settings_need_a_disk_target():
    _rejects(
        {"offload": {"all": "cpu", "disk": {"path": "/scratch"}}},
        "`activation_checkpointing.offload.disk` configures disk offload, but no "
        "offload target is set to `disk`.",
    )


def test_offload_bool_means_cpu():
    block = ActivationCheckpointingConfig.model_validate(
        {"selective": True, "offload": {"hidden_states": True, "saves": False}}
    )
    assert block.offload.hidden_states == "cpu"
    assert block.offload.saves == "none"


def test_disabled_rejects_recompute_settings():
    _rejects(
        {
            "enabled": False,
            "selective": True,
            "reentrant": False,
            "kwargs": {"every_n_layers": 2},
            "implementation": "fsdp",
        },
        "`activation_checkpointing.selective`, `activation_checkpointing.reentrant`, "
        "`activation_checkpointing.kwargs`, `activation_checkpointing.implementation` "
        "require checkpointing; set `activation_checkpointing.enabled: true` or "
        "remove them.",
    )


def test_disabled_allows_offload_all():
    block = ActivationCheckpointingConfig.model_validate(
        {"enabled": False, "offload": {"all": "cpu"}}
    )
    assert block.offload.all == "cpu"


def test_offload_all_covers_hidden_states():
    _rejects(
        {"offload": {"all": "cpu", "hidden_states": "cpu"}},
        "`activation_checkpointing.offload.all` already offloads each checkpointed "
        "layer's input; set `offload.hidden_states: none`.",
    )


def test_fsdp_implementation_rejects_reentrant():
    _rejects(
        {"implementation": "fsdp", "reentrant": True},
        "`activation_checkpointing.reentrant: true` requires `implementation: hf`: "
        "FSDP's checkpoint wrapper is non-reentrant.",
    )


def test_fsdp_implementation_takes_every_other_setting():
    block = ActivationCheckpointingConfig.model_validate(
        {
            "implementation": "fsdp",
            "reentrant": False,
            "kwargs": {"every_n_layers": 2, "preserve_rng_state": False},
            "selective": {"save": ["attention"]},
            "offload": {"hidden_states": "cpu", "saves": "cpu"},
        }
    )
    assert block.selective.save == ["attention"]
    assert block.kwargs == {"every_n_layers": 2, "preserve_rng_state": False}
    assert (
        ActivationCheckpointingConfig.model_validate(
            {"implementation": "fsdp", "offload": {"all": "cpu"}}
        ).offload.all
        == "cpu"
    )


def test_fsdp_implementation_requires_fsdp_config(min_base_cfg):
    with pytest.raises(ValueError, match="requires `fsdp_config`"):
        validate_config(
            min_base_cfg
            | DictDefault(activation_checkpointing={"implementation": "fsdp"})
        )


@pytest.mark.parametrize(
    ("value", "enabled"), [(None, False), (False, False), (True, True), ({}, True)]
)
def test_shorthand(min_base_cfg, value, enabled):
    cfg = validate_config(min_base_cfg | DictDefault(activation_checkpointing=value))
    assert cfg.activation_checkpointing.enabled is enabled
    assert cfg.activation_checkpointing.implementation == "hf"


def test_unknown_block_field_rejected(min_base_cfg):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        validate_config(
            min_base_cfg
            | DictDefault(activation_checkpointing={"use_reentrant": False})
        )
