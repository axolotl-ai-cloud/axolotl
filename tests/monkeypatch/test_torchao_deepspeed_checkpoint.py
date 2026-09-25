import pytest
import torch

from axolotl.monkeypatch.torchao_deepspeed import load_native_nvfp4_adapter_state


def _model():
    model = torch.nn.Module()
    native = type("NVFP4Tensor", (torch.nn.Parameter,), {})
    model.base = native(torch.ones(2), requires_grad=False)
    model.adapter = torch.nn.Parameter(torch.zeros(2))
    return model


def test_adapter_load_allows_frozen_native_omission():
    model = _model()
    load_native_nvfp4_adapter_state(model, {"adapter": torch.ones(2)})
    assert torch.equal(model.adapter, torch.ones(2))


def test_adapter_load_rejects_missing_trainable_key():
    with pytest.raises(ValueError, match="missing"):
        load_native_nvfp4_adapter_state(_model(), {})


def test_adapter_load_rejects_unknown_key():
    with pytest.raises(ValueError, match="unexpected"):
        load_native_nvfp4_adapter_state(
            _model(), {"adapter": torch.ones(2), "bad": torch.ones(1)}
        )


@pytest.mark.parametrize("marked", [False, True])
def test_engine_load_preserves_strict_default_and_validates_adapters(marked):
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    class Engine:
        def __init__(self, model):
            self.module = model

        def _broadcast_model(self):
            pass

        def load_module_state_dict(
            self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False
        ):
            if custom_load_fn:
                custom_load_fn(src=checkpoint["module"], dst=self.module)
            else:
                self.module.load_state_dict(checkpoint["module"], strict=strict)

    model = _model()
    model.register_buffer("counter", torch.zeros(1))
    if marked:
        model._axolotl_native_nvfp4_deepspeed_names = {"base"}
    _install_native_nvfp4_broadcast_filter(Engine)
    engine = Engine(model)
    state = {"adapter": torch.ones(2), "counter": torch.ones(1)}
    if not marked:
        with pytest.raises(RuntimeError, match="Missing key"):
            engine.load_module_state_dict({"module": state})
        return
    engine.load_module_state_dict({"module": state})
    assert torch.equal(model.adapter, torch.ones(2))
    assert torch.equal(model.counter, torch.ones(1))
    with pytest.raises(ValueError, match="missing"):
        engine.load_module_state_dict({"module": {"counter": torch.zeros(1)}})
    assert torch.equal(model.counter, torch.ones(1))


def test_marked_engine_rejects_frozen_native_checkpoint_key():
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    class Engine:
        def __init__(self, model):
            self.module = model

        def _broadcast_model(self):
            pass

        def load_module_state_dict(
            self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False
        ):
            if custom_load_fn:
                return custom_load_fn(src=checkpoint["module"], dst=self.module)
            return self.module.load_state_dict(checkpoint["module"], strict=strict)

    model = _model()
    model._axolotl_native_nvfp4_deepspeed_names = {"base"}
    _install_native_nvfp4_broadcast_filter(Engine)
    with pytest.raises(ValueError, match="frozen native"):
        Engine(model).load_module_state_dict(
            {"module": {"adapter": torch.ones(2), "base": torch.ones(2)}}
        )


def test_marked_engine_rejects_missing_persistent_buffer_but_allows_nonpersistent():
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    class Engine:
        def __init__(self, model):
            self.module = model

        def _broadcast_model(self):
            pass

        def load_module_state_dict(
            self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False
        ):
            if custom_load_fn:
                return custom_load_fn(src=checkpoint["module"], dst=self.module)
            return self.module.load_state_dict(checkpoint["module"], strict=strict)

    model = _model()
    model.register_buffer("persistent", torch.ones(1))
    model.register_buffer("scratch", torch.ones(1), persistent=False)
    model._axolotl_native_nvfp4_deepspeed_names = {"base"}
    _install_native_nvfp4_broadcast_filter(Engine)
    with pytest.raises(ValueError, match="missing"):
        Engine(model).load_module_state_dict({"module": {"adapter": torch.ones(2)}})

    Engine(model).load_module_state_dict(
        {"module": {"adapter": torch.ones(2), "persistent": torch.zeros(1)}}
    )
    assert torch.equal(model.persistent, torch.zeros(1))
    assert torch.equal(model.scratch, torch.ones(1))


@pytest.mark.parametrize("marked", [False, True])
def test_native_save_excludes_every_frozen_alias(marked):
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    class Engine:
        def _broadcast_model(self):
            pass

        def module_state_dict(
            self,
            destination=None,
            prefix="",
            keep_vars=False,
            exclude_frozen_parameters=False,
        ):
            state = self.module.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
            if exclude_frozen_parameters:
                for name, parameter in self.module.named_parameters():
                    if not parameter.requires_grad:
                        state.pop(prefix + name, None)
            return state

    model = _model()
    model.tied = model.base
    if marked:
        model._axolotl_native_nvfp4_deepspeed_names = {"base"}
    engine = Engine()
    engine.module = model
    _install_native_nvfp4_broadcast_filter(Engine)
    saved = engine.module_state_dict(prefix="prefix.", exclude_frozen_parameters=True)
    assert "prefix.adapter" in saved
    assert "prefix.base" not in saved
    assert ("prefix.tied" in saved) == (not marked)
    assert "prefix.tied" in engine.module_state_dict(prefix="prefix.")


def test_marked_engine_rejects_frozen_native_alias_before_load():
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    class Engine:
        def __init__(self, model):
            self.module = model

        def _broadcast_model(self):
            pass

        def load_module_state_dict(
            self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False
        ):
            if custom_load_fn:
                return custom_load_fn(src=checkpoint["module"], dst=self.module)
            return self.module.load_state_dict(checkpoint["module"], strict=strict)

    model = _model()
    model.register_parameter("base_alias", model.base)
    model._axolotl_native_nvfp4_deepspeed_names = {"base"}
    _install_native_nvfp4_broadcast_filter(Engine)
    before = model.adapter.detach().clone()
    with pytest.raises(ValueError, match="frozen native"):
        Engine(model).load_module_state_dict(
            {"module": {"adapter": torch.ones(2), "base_alias": torch.zeros(2)}}
        )
    assert torch.equal(model.adapter, before)


def test_marked_engine_rejects_ordinary_frozen_parameter_replacement():
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    class Engine:
        def __init__(self, model):
            self.module = model

        def _broadcast_model(self):
            pass

    model = torch.nn.Module()
    model.base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model._axolotl_native_nvfp4_deepspeed_names = {"base"}
    _install_native_nvfp4_broadcast_filter(Engine)
    with pytest.raises(ValueError, match="must remain frozen"):
        Engine(model)._broadcast_model()


def test_unconverted_native_engine_rejects_zero3_fetch_on_restore():
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    class Engine:
        def __init__(self, model):
            self.module = model

        def _broadcast_model(self):
            pass

        def load_module_state_dict(
            self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False
        ):
            return None

    model = _model()
    model._axolotl_native_nvfp4_deepspeed_names = {"base"}
    _install_native_nvfp4_broadcast_filter(Engine)
    with pytest.raises(ValueError, match="does not support ZeRO-3"):
        Engine(model).load_module_state_dict(
            {"module": {"adapter": torch.ones(2)}}, fetch_z3_params=True
        )


def test_marked_engine_forwards_zero3_load_keywords():
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    calls = []

    class Engine:
        def __init__(self, model):
            self.module = model

        def _broadcast_model(self):
            pass

        def load_module_state_dict(
            self,
            checkpoint,
            strict=True,
            custom_load_fn=None,
            fetch_z3_params=False,
            **kwargs,
        ):
            calls.append((strict, fetch_z3_params, kwargs))
            return custom_load_fn(src=checkpoint["module"], dst=self.module)

    model = _model()
    model.base = torch.nn.Parameter(torch.ones(2, dtype=torch.uint8), False)
    model._axolotl_native_nvfp4_deepspeed_names = {"base"}
    model._axolotl_native_nvfp4_zero3_components = {"base"}
    _install_native_nvfp4_broadcast_filter(Engine)

    Engine(model).load_module_state_dict(
        {"module": {"adapter": torch.ones(2)}},
        strict=False,
        fetch_z3_params=True,
        z3_params_to_fetch=[model.adapter],
    )

    assert calls == [(False, True, {"z3_params_to_fetch": [model.adapter]})]
    assert torch.equal(model.adapter, torch.ones(2))
