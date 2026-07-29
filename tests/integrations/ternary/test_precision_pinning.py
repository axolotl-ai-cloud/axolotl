"""The gates and the calibration solve must be immune to ambient TF32 state."""

import pytest
import torch

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.export import parity
from axolotl.integrations.ternary.ptq import calibrate


def test_the_context_pins_and_restores():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    with quant.pinned_fp32_precision():
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "high"
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_the_context_restores_on_exception():
    torch.set_float32_matmul_precision("high")
    with pytest.raises(RuntimeError):
        with quant.pinned_fp32_precision():
            raise RuntimeError("boom")
    assert torch.get_float32_matmul_precision() == "high"


class _RecordingExtractors(dict):
    def __init__(self):
        super().__init__()
        self.seen = None

    def get(self, key, default=None):
        self.seen = torch.get_float32_matmul_precision()
        return None


def test_run_smoke_eval_is_immune_to_ambient_tf32(monkeypatch, tmp_path):
    recorder = _RecordingExtractors()
    monkeypatch.setattr(parity, "EXTRACTORS", recorder)
    torch.set_float32_matmul_precision("high")
    assert parity.run_smoke_eval(tmp_path, tmp_path, "hf_bitnet") is None
    assert recorder.seen == "highest"
    assert torch.get_float32_matmul_precision() == "high"


def test_run_parity_gate_is_immune_to_ambient_tf32(monkeypatch, tmp_path):
    recorder = _RecordingExtractors()
    monkeypatch.setattr(parity, "EXTRACTORS", recorder)
    torch.set_float32_matmul_precision("high")
    with pytest.raises(ValueError, match="unpacker registered"):
        parity.run_parity_gate(tmp_path, tmp_path, "hf_bitnet", manifest=None)
    assert recorder.seen == "highest"
    assert torch.get_float32_matmul_precision() == "high"


def test_calibration_is_immune_to_ambient_tf32(monkeypatch):
    seen = {}

    def record_and_stop(model):
        seen["precision"] = torch.get_float32_matmul_precision()
        raise RuntimeError("stop here")

    monkeypatch.setattr(calibrate, "decoder_layers", record_and_stop)
    monkeypatch.setattr(
        calibrate,
        "resolve_ternary_config",
        lambda cfg: type("C", (), {"init_calibration": None})(),
    )
    torch.set_float32_matmul_precision("high")
    with pytest.raises(RuntimeError, match="stop here"):
        calibrate.calibrate_model_latents(model=None, manifest=None, cfg={})
    assert seen["precision"] == "highest"
    assert torch.get_float32_matmul_precision() == "high"
