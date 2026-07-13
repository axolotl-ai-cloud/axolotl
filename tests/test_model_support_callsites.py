"""Offline contracts for model-support lifecycle and component callsites."""

from types import SimpleNamespace

import pytest

from axolotl.model_support import (
    ModelFamilyTemplate,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
    ModelStrategies,
    ModelSupport,
)
from axolotl.utils.dict import DictDefault


class _DefaultAutoModel:
    pass


class _ProfileAutoModel:
    pass


class _MappedMultimodalAutoModel:
    pass


class _LegacyAutoModel:
    pass


def _bare_model_loader(model_loader_cls, model_type: str, *, is_multimodal: bool):
    loader = object.__new__(model_loader_cls)
    loader.cfg = DictDefault(is_multimodal=is_multimodal)
    loader.model_config = SimpleNamespace(model_type=model_type)
    loader.auto_model_loader = _DefaultAutoModel
    return loader


@pytest.mark.parametrize("is_multimodal", [False, True])
def test_model_loader_uses_profile_auto_model_strategy(monkeypatch, is_multimodal):
    from axolotl.loaders import model as model_loader_module

    class ProfileSupport(ModelSupport):
        model_types = ("profile_loader_test",)
        profile = ModelProfile(
            family=ModelFamilyTemplate(
                name="profile_loader_family",
                strategies=ModelStrategies(
                    auto_model_cls=lambda: _ProfileAutoModel,
                ),
            )
        )

    support = ProfileSupport()
    loader = _bare_model_loader(
        model_loader_module.ModelLoader,
        "profile_loader_test",
        is_multimodal=is_multimodal,
    )
    monkeypatch.setattr(
        model_loader_module,
        "get_model_support",
        lambda model_type: support if model_type == "profile_loader_test" else None,
    )
    monkeypatch.setattr(
        model_loader_module,
        "MULTIMODAL_AUTO_MODEL_MAPPING",
        {"profile_loader_test": _MappedMultimodalAutoModel},
    )

    loader._set_auto_model_loader()

    assert loader.auto_model_loader is _ProfileAutoModel


def test_model_loader_preserves_multimodal_fallback_for_nullable_strategy(
    monkeypatch,
):
    from axolotl.loaders import model as model_loader_module

    class NullableProfileSupport(ModelSupport):
        model_types = ("nullable_loader_test",)
        profile = ModelProfile(
            family=ModelFamilyTemplate(
                name="nullable_loader_family",
                strategies=ModelStrategies(auto_model_cls=lambda: None),
            )
        )

    support = NullableProfileSupport()
    loader = _bare_model_loader(
        model_loader_module.ModelLoader,
        "nullable_loader_test",
        is_multimodal=True,
    )
    monkeypatch.setattr(model_loader_module, "get_model_support", lambda _type: support)
    monkeypatch.setattr(
        model_loader_module,
        "MULTIMODAL_AUTO_MODEL_MAPPING",
        {"nullable_loader_test": _MappedMultimodalAutoModel},
    )

    loader._set_auto_model_loader()

    assert loader.auto_model_loader is _MappedMultimodalAutoModel


def test_model_loader_preserves_default_for_unregistered_causal_model(monkeypatch):
    from axolotl.loaders import model as model_loader_module

    loader = _bare_model_loader(
        model_loader_module.ModelLoader,
        "unregistered_loader_test",
        is_multimodal=False,
    )
    monkeypatch.setattr(model_loader_module, "get_model_support", lambda _type: None)

    loader._set_auto_model_loader()

    assert loader.auto_model_loader is _DefaultAutoModel


def test_model_loader_uses_legacy_auto_model_provider(monkeypatch):
    from axolotl.loaders import model as model_loader_module

    class LegacySupport(ModelSupport):
        model_types = ("legacy_loader_test",)

        def get_auto_model_cls(self):
            return _LegacyAutoModel

    support = LegacySupport()
    loader = _bare_model_loader(
        model_loader_module.ModelLoader,
        "legacy_loader_test",
        is_multimodal=False,
    )
    monkeypatch.setattr(model_loader_module, "get_model_support", lambda _type: support)

    loader._set_auto_model_loader()

    assert loader.auto_model_loader is _LegacyAutoModel


def test_model_loader_forwards_lifecycle_context_to_patch_manager(monkeypatch):
    from axolotl.loaders import model as model_loader_module

    cfg = DictDefault(
        base_model="context_loader_test",
        type_of_model=None,
        overrides_of_model_kwargs=None,
    )
    model_config = object()
    tokenizer = object()
    processor = object()
    calls = []

    class RecordingPatchManager:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        model_loader_module,
        "load_model_config",
        lambda candidate: model_config if candidate is cfg else None,
    )
    monkeypatch.setattr(model_loader_module, "PatchManager", RecordingPatchManager)

    loader = model_loader_module.ModelLoader(
        cfg,
        tokenizer,
        processor=processor,
        inference=True,
        reference_model=True,
    )

    assert loader.processor is processor
    assert calls == [
        {
            "cfg": cfg,
            "model_config": model_config,
            "inference": True,
            "tokenizer": tokenizer,
            "processor": processor,
            "reference_model": True,
        }
    ]


def test_patch_manager_dispatches_lifecycle_hooks_with_complete_context(monkeypatch):
    from axolotl.loaders import patch_manager as patch_manager_module

    observed = []
    placement = []

    def capture(phase):
        def hook(context):
            observed.append((phase, context))
            placement.append(("hook", phase))

        return hook

    phases = (
        ModelHookPhase.BEFORE_CONFIG_LOAD,
        ModelHookPhase.BEFORE_TOKENIZER_LOAD,
        ModelHookPhase.BEFORE_MODEL_BUILD,
        ModelHookPhase.AFTER_BASE_MODEL_BUILD,
        ModelHookPhase.AFTER_ADAPTER_LOAD,
    )
    callsite_profile = ModelProfile(
        family=ModelFamilyTemplate(
            name="patch_manager_callsite_family",
            hooks=ModelHooks({phase: (capture(phase),) for phase in phases}),
        )
    )

    class CallsiteSupport(ModelSupport):
        model_types = ("patch_manager_callsite_test",)
        profile = callsite_profile

    support = CallsiteSupport()
    cfg = DictDefault(
        model_config_type="patch_manager_callsite_test",
        inference=True,
    )
    model_config = object()
    tokenizer = object()
    processor = object()
    base_model = object()
    adapter_model = object()

    monkeypatch.setattr(
        patch_manager_module,
        "get_model_support_for_cfg",
        lambda candidate: support if candidate is cfg else None,
    )
    monkeypatch.setattr(
        patch_manager_module,
        "get_model_support",
        lambda model_type: (
            support if model_type == "patch_manager_callsite_test" else None
        ),
    )

    patch_manager_module.PatchManager.apply_pre_config_load_patches(cfg)
    patch_manager_module.PatchManager.apply_pre_tokenizer_load_patches(cfg)

    manager = patch_manager_module.PatchManager(
        cfg,
        model_config,
        inference=True,
        tokenizer=tokenizer,
        processor=processor,
        reference_model=True,
    )
    manager._apply_model_support_pre_load_hook()

    placement.clear()
    monkeypatch.setattr(
        manager,
        "_apply_gemma_hybrid_attention",
        lambda model: placement.append(("generic", model)),
    )
    monkeypatch.setattr(manager, "_apply_gemma4_loss_kwargs", lambda: None)
    monkeypatch.setattr(
        manager, "_finalize_moe_expert_quantization", lambda _model: None
    )
    before_post_build = len(observed)
    manager.apply_post_model_build_patches(base_model)
    assert observed[before_post_build][0] is ModelHookPhase.AFTER_BASE_MODEL_BUILD
    assert placement == [
        ("hook", ModelHookPhase.AFTER_BASE_MODEL_BUILD),
        ("generic", base_model),
    ]

    placement.clear()
    for method_name in (
        "_apply_llama_flash_attn_patches",
        "_apply_lora_kernel_patch",
        "_apply_scaling_softmax_patch",
        "_apply_fp8_attention_patches",
        "_apply_tiled_mlp_post_load",
    ):
        monkeypatch.setattr(
            manager,
            method_name,
            lambda model, name=method_name: placement.append((name, model)),
        )
    before_post_adapter = len(observed)
    manager.apply_post_model_load_patches(adapter_model)
    assert observed[before_post_adapter][0] is ModelHookPhase.AFTER_ADAPTER_LOAD
    assert placement[0] == ("hook", ModelHookPhase.AFTER_ADAPTER_LOAD)
    assert len(placement) == 6
    assert all(event[1] is adapter_model for event in placement[1:])

    assert [phase for phase, _context in observed] == list(phases)
    pre_config_context = observed[0][1]
    pre_tokenizer_context = observed[1][1]
    for context in (pre_config_context, pre_tokenizer_context):
        assert context.cfg is cfg
        assert context.model_config is None
        assert context.tokenizer is None
        assert context.processor is None
        assert context.model is None
        assert context.inference is True
        assert context.reference_model is None

    before_model_context = observed[2][1]
    assert before_model_context.cfg is cfg
    assert before_model_context.model_config is model_config
    assert before_model_context.tokenizer is tokenizer
    assert before_model_context.processor is processor
    assert before_model_context.model is None
    assert before_model_context.inference is True
    assert before_model_context.reference_model is True

    post_build_context = observed[3][1]
    post_adapter_context = observed[4][1]
    assert post_build_context.model is base_model
    assert post_adapter_context.model is adapter_model
    for context in (post_build_context, post_adapter_context):
        assert context.cfg is cfg
        assert context.model_config is model_config
        assert context.tokenizer is tokenizer
        assert context.processor is processor
        assert context.inference is True
        assert context.reference_model is True


def test_model_loader_places_post_build_and_post_adapter_phases(monkeypatch):
    from axolotl.loaders import model as model_loader_module

    events = []
    base_model = object()
    adapter_model = object()

    class RecordingPatchManager:
        def apply_pre_model_load_patches(self):
            events.append("before_model_build")

        def apply_post_plugin_pre_model_load_patches(self):
            events.append("post_plugin_pre_model_load")

        def apply_post_model_build_patches(self, model):
            assert model is base_model
            events.append("after_base_model_build")

        def apply_post_model_load_patches(self, model):
            assert model is adapter_model
            events.append("after_adapter_load")

    class RecordingPluginManager:
        def pre_model_load(self, _cfg):
            events.append("plugin_pre_model_load")

        def post_model_build(self, _cfg, model):
            assert model is base_model
            events.append("plugin_post_model_build")

        def pre_lora_load(self, _cfg, model):
            assert model is base_model
            events.append("plugin_pre_adapter_load")

        def post_lora_load(self, _cfg, model):
            assert model is adapter_model
            events.append("plugin_post_adapter_load")

        def post_model_load(self, _cfg, model):
            assert model is adapter_model
            events.append("plugin_post_model_load")

    loader = object.__new__(model_loader_module.ModelLoader)
    loader.cfg = DictDefault(fp32_norms=False)
    loader.model = base_model
    loader.patch_manager = RecordingPatchManager()

    loader._apply_pre_model_load_setup = lambda: events.append("pre_model_setup")

    def build_model():
        events.append("build_model")
        loader.model = base_model
        return False

    loader._build_model = build_model
    loader._apply_post_model_load_setup = lambda: events.append("post_model_setup")

    def load_adapters():
        events.append("load_adapters")
        loader.model = adapter_model
        return None

    loader._load_adapters = load_adapters
    loader._apply_post_lora_load_setup = lambda _skip: events.append(
        "post_adapter_setup"
    )
    monkeypatch.setattr(
        model_loader_module,
        "PLUGIN_MANAGER",
        RecordingPluginManager(),
    )

    model, lora_config = model_loader_module.ModelLoader.load.__wrapped__(loader)

    assert model is adapter_model
    assert lora_config is None
    assert events == [
        "before_model_build",
        "pre_model_setup",
        "plugin_pre_model_load",
        "post_plugin_pre_model_load",
        "build_model",
        "after_base_model_build",
        "plugin_post_model_build",
        "post_model_setup",
        "plugin_pre_adapter_load",
        "load_adapters",
        "plugin_post_adapter_load",
        "post_adapter_setup",
        "after_adapter_load",
        "plugin_post_model_load",
    ]


def test_cli_loads_processor_before_model_and_forwards_context(monkeypatch):
    from axolotl.cli.utils import load as load_module

    events = []
    tokenizer = object()
    processor = object()
    expected_processor = processor
    model = object()
    cfg = DictDefault(
        tokenizer_config="processor_callsite_test",
        base_model_config="processor_callsite_test",
        is_multimodal=True,
    )

    def load_tokenizer(candidate):
        assert candidate is cfg
        events.append("tokenizer")
        return tokenizer

    def load_processor(candidate, received_tokenizer):
        assert candidate is cfg
        assert received_tokenizer is tokenizer
        events.append("processor")
        return processor

    class RecordingModelLoader:
        def __init__(
            self,
            candidate,
            received_tokenizer,
            *,
            processor: object | None = None,
            inference: bool = False,
        ):
            assert candidate is cfg
            assert received_tokenizer is tokenizer
            assert processor is expected_processor
            assert inference is True
            events.append("model_loader")

        def load(self):
            events.append("model")
            return model, None

    monkeypatch.setattr(load_module, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(load_module, "load_processor", load_processor)
    monkeypatch.setattr(load_module, "ModelLoader", RecordingModelLoader)

    result = load_module.load_model_and_tokenizer(cfg=cfg, inference=True)

    assert result == (model, tokenizer, processor)
    assert events == ["tokenizer", "processor", "model_loader", "model"]


def test_reference_model_forwards_processor_to_model_loader(monkeypatch):
    import axolotl.train as train_module

    tokenizer = object()
    processor = object()
    reference_model = object()
    cfg = DictDefault(
        rl=train_module.RLType.DPO,
        adapter=None,
        rl_adapter_ref_model=False,
        trl=None,
    )
    calls = []

    class RecordingModelLoader:
        def __init__(
            self,
            candidate,
            received_tokenizer,
            *,
            processor: object | None = None,
            reference_model: bool = False,
        ):
            calls.append(
                (
                    candidate,
                    received_tokenizer,
                    processor,
                    reference_model,
                )
            )

        def load(self):
            return reference_model, None

    monkeypatch.setattr(train_module, "ModelLoader", RecordingModelLoader)

    result = train_module.setup_reference_model(cfg, tokenizer, processor)

    assert result is reference_model
    assert calls == [(cfg, tokenizer, processor, True)]
