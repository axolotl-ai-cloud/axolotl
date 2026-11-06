"""Deprecated checkpointing keys and the rules local to the activation_checkpointing
block."""

import logging
import re

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.activation_checkpointing import (
    ActivationCheckpointingConfig,
    migrate_legacy_checkpointing_keys,
)

SEE = " See docs/activation_checkpointing.qmd#sec-migrate-activation-checkpointing."
SEE_GUIDE = (
    " See docs/activation_checkpointing.qmd#sec-migrate-activation-checkpointing for "
    "the migration guide."
)
UNSET = object()


def _deprecated(names, rest):
    verb = "are" if " and " in names else "is"
    return f"{names} {verb} deprecated and will be removed in a future release. {rest}"


DEPRECATED_KEYS = [
    pytest.param(
        {"gradient_checkpointing": True},
        _deprecated(
            "`gradient_checkpointing`", "Use `activation_checkpointing: true` instead."
        ),
        True,
        id="enabled",
    ),
    pytest.param(
        {"gradient_checkpointing": False},
        _deprecated(
            "`gradient_checkpointing`", "Use `activation_checkpointing: false` instead."
        ),
        False,
        id="disabled",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        },
        _deprecated(
            "`gradient_checkpointing` and `gradient_checkpointing_kwargs`",
            "Use `activation_checkpointing: {enabled: true, reentrant: false}` "
            "instead.",
        ),
        {"enabled": True, "reentrant": False},
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
        _deprecated(
            "`gradient_checkpointing` and `gradient_checkpointing_kwargs`",
            "Use `activation_checkpointing: {enabled: true, reentrant: true, kwargs: "
            "{every_n_layers: 2, preserve_rng_state: false}}` instead.",
        ),
        {
            "enabled": True,
            "reentrant": True,
            "kwargs": {"every_n_layers": 2, "preserve_rng_state": False},
        },
        id="passthrough_kwargs",
    ),
    pytest.param(
        {"gradient_checkpointing": "offload"},
        _deprecated(
            "`gradient_checkpointing`",
            "Use `activation_checkpointing: {enabled: true, offload: {all: cpu}}` "
            "instead.",
        ),
        {"enabled": True, "offload": {"all": "cpu"}},
        id="offload_string",
    ),
    pytest.param(
        {"gradient_checkpointing_kwargs": {"use_reentrant": True, "every_n_layers": 2}},
        _deprecated(
            "`gradient_checkpointing_kwargs`",
            "Delete it: checkpointing is off, so it has no effect.",
        ),
        UNSET,
        id="kwargs_without_checkpointing",
    ),
    pytest.param(
        {
            "gradient_checkpointing": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        },
        _deprecated(
            "`gradient_checkpointing` and `gradient_checkpointing_kwargs`",
            "Use `activation_checkpointing: false` instead. "
            "`gradient_checkpointing_kwargs` had no effect with checkpointing off.",
        ),
        False,
        id="kwargs_with_checkpointing_off",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"offload": True},
        },
        _deprecated(
            "`gradient_checkpointing` and `gradient_checkpointing_kwargs`",
            "Use `activation_checkpointing: {enabled: true, offload: {hidden_states: "
            "cpu}}` instead. `offload` in the checkpoint kwargs copied each "
            "checkpointed layer's input to CPU synchronously; "
            "`offload.hidden_states: cpu` offloads the same tensors with "
            "stream-overlapped copies.",
        ),
        {"enabled": True, "offload": {"hidden_states": "cpu"}},
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
        _deprecated(
            "`gradient_checkpointing` and `gradient_checkpointing_kwargs`",
            "Use `activation_checkpointing: {enabled: true, reentrant: false}` "
            "instead. `respect_saved_tensors_hooks` has no replacement: axolotl sets "
            "it.",
        ),
        {"enabled": True, "reentrant": False},
        id="kwargs_owned",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False, "preserve": 1},
        },
        _deprecated(
            "`gradient_checkpointing` and `gradient_checkpointing_kwargs`",
            "Use `activation_checkpointing: {enabled: true, reentrant: false}` "
            "instead. `preserve` is not a torch.utils.checkpoint.checkpoint argument "
            "and is dropped; torch passed it to each checkpointed layer's forward.",
        ),
        {"enabled": True, "reentrant": False},
        id="kwargs_unknown",
    ),
    pytest.param(
        {"gradient_checkpointing": True, "selective_checkpointing": True},
        _deprecated(
            "`gradient_checkpointing` and `selective_checkpointing`",
            "Use `activation_checkpointing: {enabled: true, selective: true}` instead.",
        ),
        {"enabled": True, "selective": True},
        id="selective",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "selective_checkpointing": {"save": ["attention"], "offload": True},
        },
        _deprecated(
            "`gradient_checkpointing` and `selective_checkpointing`",
            "Use `activation_checkpointing: {enabled: true, selective: {save: "
            "[attention]}, offload: {selective: cpu}}` instead.",
        ),
        {
            "enabled": True,
            "selective": {"save": ["attention"]},
            "offload": {"selective": "cpu"},
        },
        id="selective_offload",
    ),
    pytest.param(
        {"gradient_checkpointing": True, "activation_offloading": "hidden_states"},
        _deprecated(
            "`gradient_checkpointing` and `activation_offloading`",
            "Use `activation_checkpointing: {enabled: true, offload: {hidden_states: "
            "cpu}}` instead.",
        ),
        {"enabled": True, "offload": {"hidden_states": "cpu"}},
        id="hidden_states",
    ),
    pytest.param(
        {"activation_offloading": True},
        _deprecated(
            "`activation_offloading`",
            "Use `activation_checkpointing: {enabled: false, offload: {all: cpu}}` "
            "instead.",
        ),
        {"enabled": False, "offload": {"all": "cpu"}},
        id="offload_without_recompute",
    ),
    pytest.param(
        {"gradient_checkpointing": True, "activation_offloading": "legacy"},
        _deprecated(
            "`gradient_checkpointing` and `activation_offloading`",
            "Use `activation_checkpointing: {enabled: true, offload: {all: cpu}}` "
            "instead.",
        ),
        {"enabled": True, "offload": {"all": "cpu"}},
        id="offload_legacy",
    ),
    pytest.param(
        {"activation_offloading": False},
        _deprecated(
            "`activation_offloading`",
            "Delete it: activation offloading is off by default.",
        ),
        UNSET,
        id="offload_off",
    ),
    pytest.param(
        {"fsdp_config": {"activation_checkpointing": True}},
        _deprecated(
            "`fsdp_config.activation_checkpointing`",
            "Use `activation_checkpointing: {enabled: true, implementation: fsdp}` "
            "instead.",
        ),
        {"enabled": True, "implementation": "fsdp"},
        id="fsdp",
    ),
    pytest.param(
        {"fsdp_config": {"fsdp_activation_checkpointing": True}},
        _deprecated(
            "`fsdp_config.fsdp_activation_checkpointing`",
            "Use `activation_checkpointing: {enabled: true, implementation: fsdp}` "
            "instead.",
        ),
        {"enabled": True, "implementation": "fsdp"},
        id="fsdp_prefixed",
    ),
    pytest.param(
        {"fsdp_config": {"activation_checkpointing": False}},
        _deprecated(
            "`fsdp_config.activation_checkpointing`",
            "Delete it: activation checkpointing and offloading are off by default.",
        ),
        UNSET,
        id="fsdp_off",
    ),
    pytest.param(
        {
            "gradient_checkpointing_kwargs": {"use_reentrant": True},
            "fsdp_config": {"activation_checkpointing": True},
        },
        _deprecated(
            "`gradient_checkpointing_kwargs` and `fsdp_config.activation_checkpointing`",
            "Use `activation_checkpointing: {enabled: true, implementation: fsdp}` "
            "instead. FSDP's checkpoint wrapper ignored "
            "`gradient_checkpointing_kwargs`, so `use_reentrant` had no effect.",
        ),
        {"enabled": True, "implementation": "fsdp"},
        id="fsdp_ignored_kwargs",
    ),
]


@pytest.mark.parametrize(("legacy", "message", "block"), DEPRECATED_KEYS)
def test_deprecated_keys_migrate(legacy, message, block):
    data = {**legacy, "base_model": "m"}
    assert migrate_legacy_checkpointing_keys(data) == message + SEE

    migrated = {"base_model": "m"}
    if "fsdp_config" in legacy:
        migrated["fsdp_config"] = {"fsdp_version": 2}
    if block is not UNSET:
        migrated["activation_checkpointing"] = block
    assert data == migrated


REJECTED_LEGACY_KEYS = [
    pytest.param(
        {"gradient_checkpointing": "offload_disk"},
        "`gradient_checkpointing: offload_disk` is disabled pending a fix: it never "
        "wrote to disk (it ran the same CPU offloader as `activation_offloading: "
        "true`). For CPU offload use `activation_checkpointing: {enabled: true, "
        "offload: {all: cpu}}`.",
        id="offload_disk_string",
    ),
    pytest.param(
        {"gradient_checkpointing": True, "activation_offloading": "disk"},
        "`activation_offloading: disk` is disabled pending a fix: it never wrote to "
        "disk (it ran the same CPU offloader as `activation_offloading: true`). For "
        "CPU offload use `activation_checkpointing: {enabled: true, offload: {all: "
        "cpu}}`.",
        id="offload_disk",
    ),
    pytest.param(
        {"selective_checkpointing": {"save": ["attention"]}},
        "`selective_checkpointing` requires gradient checkpointing. Use "
        "`activation_checkpointing: {enabled: true, selective: {save: [attention]}}`.",
        id="selective_without_checkpointing",
    ),
    pytest.param(
        {"activation_offloading": "hidden_states"},
        "`activation_offloading: hidden_states` requires gradient checkpointing. Use "
        "`activation_checkpointing: {enabled: true, offload: {hidden_states: cpu}}`.",
        id="hidden_states_without_checkpointing",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "fsdp_config": {"activation_checkpointing": True},
        },
        "`gradient_checkpointing` and `fsdp_config.activation_checkpointing` can't "
        "both be set (transformers' Trainer rejects them together). Use "
        "`activation_checkpointing: true` for transformers' checkpointing or "
        "`activation_checkpointing: {enabled: true, implementation: fsdp}` for FSDP's "
        "checkpoint wrapper.",
        id="hf_and_fsdp",
    ),
    pytest.param(
        {
            "gradient_checkpointing": True,
            "activation_checkpointing": {"reentrant": False},
        },
        "`gradient_checkpointing` is deprecated and can't be combined with "
        "`activation_checkpointing`. Merge `activation_checkpointing: true` into your "
        "`activation_checkpointing` block and delete it.",
        id="alongside_the_block",
    ),
]


@pytest.mark.parametrize(("legacy", "message"), REJECTED_LEGACY_KEYS)
def test_rejected_legacy_keys(legacy, message):
    data = dict(legacy)
    with pytest.raises(ValueError) as excinfo:
        migrate_legacy_checkpointing_keys(data)
    assert str(excinfo.value) == message + SEE_GUIDE
    assert data == legacy


def test_no_legacy_keys_no_message():
    data = {"activation_checkpointing": True, "gradient_checkpointing": None}
    assert migrate_legacy_checkpointing_keys(data) is None
    assert data == {"activation_checkpointing": True, "gradient_checkpointing": None}


def _validate_logging(cfg, caplog):
    # configure_logging() turns off propagation on the `axolotl` logger
    ax_logger = logging.getLogger("axolotl")
    old_propagate = ax_logger.propagate
    ax_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="axolotl"):
            return validate_config(cfg)
    finally:
        ax_logger.propagate = old_propagate


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        pytest.param(
            {
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            },
            {"enabled": True, "reentrant": False},
            id="kwargs",
        ),
        pytest.param(
            {
                "gradient_checkpointing": True,
                "selective_checkpointing": {"save": ["attention"], "offload": True},
            },
            {
                "enabled": True,
                "selective": {"save": ["attention"]},
                "offload_selective": "cpu",
            },
            id="selective_offload",
        ),
        pytest.param(
            {"activation_offloading": True},
            {"enabled": False, "all": "cpu"},
            id="offload_without_recompute",
        ),
        pytest.param(
            {
                "fsdp_version": 2,
                "fsdp_config": {
                    "activation_checkpointing": True,
                    "reshard_after_forward": True,
                },
            },
            {"enabled": True, "implementation": "fsdp"},
            id="fsdp",
        ),
    ],
)
def test_validation_migrates_deprecated_keys(min_base_cfg, caplog, legacy, expected):
    cfg = _validate_logging(min_base_cfg | DictDefault(legacy), caplog)

    block = cfg.activation_checkpointing
    assert block.enabled is expected["enabled"]
    assert block.implementation == expected.get("implementation", "hf")
    if "reentrant" in expected:
        assert block.reentrant is expected["reentrant"]
    if "selective" in expected:
        assert block.selective.save == expected["selective"]["save"]
    assert block.offload.selective == expected.get("offload_selective", "none")
    assert block.offload.all == expected.get("all", "none")
    for key in ("gradient_checkpointing", "selective_checkpointing"):
        assert cfg.get(key) is None
    assert "activation_checkpointing" not in (cfg.fsdp_config or {})
    assert "is deprecated" in caplog.text or "are deprecated" in caplog.text


@pytest.mark.parametrize("enabled", [True, False])
def test_fsdp_stays_on_when_its_only_key_is_deprecated(min_base_cfg, enabled):
    cfg = validate_config(
        min_base_cfg
        | DictDefault(fsdp_version=2, fsdp_config={"activation_checkpointing": enabled})
    )

    assert cfg.fsdp_config == {"fsdp_version": 2}
    assert cfg.activation_checkpointing.enabled is enabled
    assert cfg.activation_checkpointing.implementation == ("fsdp" if enabled else "hf")


def test_model_validator_migrates_without_editing_the_input(min_base_cfg):
    from axolotl.utils.schemas.config import AxolotlInputConfig

    data = (min_base_cfg | DictDefault(gradient_checkpointing=True)).to_dict()
    config = AxolotlInputConfig.model_validate(data)

    assert config.activation_checkpointing.enabled is True
    assert config.gradient_checkpointing is None
    assert data["gradient_checkpointing"] is True


@pytest.mark.parametrize(
    "legacy",
    [
        {"gradient_checkpointing": True, "activation_checkpointing": True},
        {"selective_checkpointing": True},
        {"activation_offloading": "disk"},
    ],
    ids=lambda legacy: "-".join(legacy),
)
def test_validation_rejects_invalid_legacy_keys(min_base_cfg, legacy):
    with pytest.raises(ValueError, match="for the migration guide"):
        validate_config(min_base_cfg | DictDefault(legacy))


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
        {"selective": True, "offload": {"selective": "disk", "hidden_states": "disk"}},
        "activation_checkpointing.offload.hidden_states, selective: disk offload is "
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
        {"selective": True, "offload": {"hidden_states": True, "selective": False}}
    )
    assert block.offload.hidden_states == "cpu"
    assert block.offload.selective == "none"


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
            "offload": {"hidden_states": "cpu", "selective": "cpu"},
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
