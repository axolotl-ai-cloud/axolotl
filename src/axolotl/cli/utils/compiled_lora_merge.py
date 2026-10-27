"""Opt-in compilation of merge-aware expert LoRA arithmetic."""

from contextlib import contextmanager, nullcontext

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def merge_expert_chunk(base, lora_a, lora_b, scaling):
    experts = base.shape[0]
    a = lora_a.float()
    b = lora_b.float()
    if experts == 1:
        delta = (b @ a) * scaling
    else:
        a = a.reshape(experts, -1, a.shape[-1])
        b = b.reshape(b.shape[0], -1, experts)
        delta = torch.einsum("o r e, e r i -> e o i", b, a) * scaling
    return (base.float() + delta).to(torch.bfloat16)


@contextmanager
def _ieee_matmul():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class CompiledExpertMerge:
    """Validate each specialization against PEFT before using compiled results."""

    def __init__(self, precision="strict"):
        if precision not in ("strict", "relaxed"):
            raise ValueError(f"Unsupported merge compile precision: {precision}")
        self.precision = precision
        self.compiled = None
        self.disabled = False
        self.validated = set()
        if precision == "relaxed":
            LOG.warning(
                "Relaxed compiled expert merging uses compiler-default numerics and "
                "skips PEFT equivalence checks; merged NVFP4 bytes may differ from "
                "the merge-aware training recipe"
            )

    def _precision_context(self):
        return _ieee_matmul() if self.precision == "strict" else nullcontext()

    def __call__(self, base, lora_a, lora_b, scaling, reference):
        if self.disabled:
            with self._precision_context():
                return reference()
        signature = tuple(
            (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
            for t in (base, lora_a, lora_b)
        ) + (scaling,)
        try:
            with self._precision_context():
                if self.compiled is None:
                    self.compiled = torch.compile(
                        merge_expert_chunk,
                        fullgraph=True,
                        dynamic=False,
                        options={
                            "emulate_precision_casts": True,
                            "use_fast_math": False,
                            "cuda.use_fast_math": False,
                            "cpp.enable_floating_point_contract_flag": "off",
                            "epilogue_fusion": False,
                        }
                        if self.precision == "strict"
                        else {},
                    )
                merged = self.compiled(base, lora_a, lora_b, scaling)
                if self.precision == "strict" and signature not in self.validated:
                    expected = reference()
                    if not torch.equal(
                        merged.contiguous().view(torch.uint8),
                        expected.contiguous().view(torch.uint8),
                    ):
                        self.disabled = True
                        LOG.warning(
                            "Compiled expert merge differs from PEFT; using eager merging"
                        )
                        return expected
                    self.validated.add(signature)
                    LOG.info(
                        "Validated strict compiled expert merge for shape %s",
                        tuple(base.shape),
                    )
                return merged
        except (torch.cuda.OutOfMemoryError, MemoryError):
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.disabled = True
            LOG.warning(
                "Compiled expert merge unavailable (%s); using eager merging", exc
            )
            with self._precision_context():
                return reference()
