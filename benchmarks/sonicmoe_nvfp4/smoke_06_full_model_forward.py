"""Smoke 6: FULL-MODEL forward of nvidia/Qwen3-30B-A3B-NVFP4 through the qwen3_moe adapter.

Drives the real integration path end to end: ``KernelsPlugin.pre_model_load`` (sonicmoe
experts registration + the adapter's layout inspection and WeightConverter registration),
``from_pretrained`` (converters fuse the per-expert NVFP4 tensors into 3D ``NVFP4Tensor``
params during load), then runs real text through all 48 layers twice, forcing the base-GEMM
backend via ``AXOLOTL_SONICMOE_NVFP4_BACKEND``:

- ``dequant``  (W4A16: dequantized base matmul, the reference)
- ``fp4_cute`` (W4A4: in-kernel SM100 grouped GEMM, activations quantized)

and compares logits: NLL/perplexity on the text (the decisive OQ1 measure with REAL
activations, unlike smoke 5's synthetic hiddens), top-1 agreement, KL(dequant||fp4), and
forward wall time. Gates: the model loads coherently (dequant NLL sane) and fp4_cute does
not blow it up (NLL within 0.5 nats); the rest is reported as info.

Env: AXOLOTL_SMOKE06_REPO (default nvidia/Qwen3-30B-A3B-NVFP4), AXOLOTL_SMOKE06_MAXTOK
(default 512 tokens per sample). Needs ~40 GB GPU (bf16 non-experts + NVFP4 experts).
"""

import os
import time

from _common import check, finish, require_sm100

REPO = os.environ.get("AXOLOTL_SMOKE06_REPO", "nvidia/Qwen3-30B-A3B-NVFP4")
MAXTOK = int(os.environ.get("AXOLOTL_SMOKE06_MAXTOK", "512"))

TEXTS = [
    # encyclopedic
    "The African bush elephant is the largest living terrestrial animal. Adult males "
    "can reach a shoulder height of four metres and a body mass of over ten tonnes. "
    "Elephants are herbivores, spending up to eighteen hours a day feeding on grasses, "
    "bark, roots, and leaves. Their trunks, containing tens of thousands of muscle "
    "fascicles, serve for breathing, smelling, touching, grasping, and producing sound. "
    "Elephant social structure is matriarchal: related females and their offspring live "
    "in cohesive family groups led by the oldest cow, while adult males range alone or "
    "form loose bachelor herds. Populations have declined sharply over the past century "
    "due to habitat loss and ivory poaching, and the species is listed as endangered.",
    # code
    "def quicksort(arr):\n"
    "    if len(arr) <= 1:\n"
    "        return arr\n"
    "    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n"
    "    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)\n\n"
    "# In-place variants avoid the list allocations above. The Lomuto partition scheme\n"
    "# swaps elements below the pivot toward the front while scanning once, then places\n"
    "# the pivot after them; Hoare's original scheme converges two indices from both\n"
    "# ends and typically performs three times fewer swaps on average.\n",
    # instructions / chat-flavored
    "To prepare a basic sourdough loaf, begin by feeding your starter eight to twelve "
    "hours before mixing, until it doubles and passes the float test. Combine flour and "
    "water and let the mixture rest for an hour; this autolyse stage hydrates the flour "
    "and begins gluten development without effort. Add the ripe starter and salt, then "
    "perform a series of stretch-and-folds at half-hour intervals over the next three "
    "hours. Once the dough has risen by roughly half and shows bubbles at the edges, "
    "shape it gently, place it seam-side up in a floured banneton, and retard it in the "
    "refrigerator overnight. Bake in a preheated Dutch oven, covered for twenty minutes "
    "and uncovered for twenty more, until deeply browned.",
    # narrative
    "The lighthouse keeper climbed the spiral stairs for the last time that October "
    "evening, counting each of the hundred and twelve steps out of habit rather than "
    "need. The lamp had been automated for a month already; his job now amounted to "
    "signing forms and handing over keys. From the gallery he watched the ferry cross "
    "the strait, its wake a pale seam on the darkening water, and he thought about the "
    "winters when no boat could make that crossing for weeks, when the light he tended "
    "was the only proof to the mainland that the island had not slipped beneath the "
    "waves entirely.",
]


def main():
    import torch

    require_sm100()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import is_nvfp4_param
    from axolotl.integrations.kernels.plugin import KernelsPlugin
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault({"base_model": REPO, "use_sonicmoe": True})
    plugin = KernelsPlugin()
    plugin.pre_model_load(cfg)

    adapters = [a.name for a in plugin._adapters(cfg)]
    check("qwen3_moe adapter matched", "qwen3_moe" in adapters)
    check("experts_implementation set", cfg.experts_implementation == "sonicmoe")

    tok = AutoTokenizer.from_pretrained(REPO)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        REPO, dtype=torch.bfloat16, device_map={"": 0}
    )
    print(f"[info] from_pretrained (converter path): {time.time() - t0:.0f}s")
    model.set_experts_implementation("sonicmoe")
    plugin.post_model_load(cfg, model)
    model.eval()

    experts0 = model.model.layers[0].mlp.experts
    check(
        "experts are NVFP4 after load",
        is_nvfp4_param(experts0.gate_up_proj) and is_nvfp4_param(experts0.down_proj),
    )

    batches = [
        tok(t, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(
            "cuda"
        )
        for t in TEXTS
    ]
    print(f"[info] {len(batches)} samples, {[b.numel() for b in batches]} tokens")

    def run(backend):
        os.environ["AXOLOTL_SONICMOE_NVFP4_BACKEND"] = backend
        logits, nlls, times = [], [], []
        with torch.no_grad():
            for ids in batches:
                torch.cuda.synchronize()
                t0 = time.time()
                lg = model(ids).logits.float()
                torch.cuda.synchronize()
                times.append(time.time() - t0)
                logits.append(lg[0])
                nll = torch.nn.functional.cross_entropy(lg[0, :-1], ids[0, 1:])
                nlls.append(float(nll))
        return logits, nlls, times

    lg_dq, nll_dq, t_dq = run("dequant")
    lg_dq, nll_dq, t_dq2 = run("dequant")  # rerun: first pass pays warmup/compile
    lg_fp4, nll_fp4, _ = run("fp4_cute")
    lg_fp4, nll_fp4, t_fp4 = run("fp4_cute")
    os.environ.pop("AXOLOTL_SONICMOE_NVFP4_BACKEND", None)

    finite = all(torch.isfinite(lg).all() for lg in lg_dq + lg_fp4)
    check("logits finite (both backends)", finite)

    mean_nll_dq = sum(nll_dq) / len(nll_dq)
    mean_nll_fp4 = sum(nll_fp4) / len(nll_fp4)
    check(f"dequant NLL sane ({mean_nll_dq:.3f} < 4.0)", mean_nll_dq < 4.0)
    check(
        f"fp4_cute NLL within 0.5 nats of dequant "
        f"({mean_nll_fp4:.3f} vs {mean_nll_dq:.3f})",
        mean_nll_fp4 < mean_nll_dq + 0.5,
    )

    import math

    for i, (a, b) in enumerate(zip(lg_dq, lg_fp4, strict=True)):
        diff = (a - b).abs()
        rel_fro = float((a - b).norm() / a.norm())
        top1 = float((a.argmax(-1) == b.argmax(-1)).float().mean())
        kl = float(
            torch.nn.functional.kl_div(
                b.log_softmax(-1), a.log_softmax(-1), log_target=True, reduction="none"
            )
            .sum(-1)
            .mean()
        )
        print(
            f"[info] sample {i}: NLL dq={nll_dq[i]:.4f} fp4={nll_fp4[i]:.4f} "
            f"(ppl {math.exp(nll_dq[i]):.2f} -> {math.exp(nll_fp4[i]):.2f}); "
            f"logits max_abs={diff.max():.3f} rel_fro={rel_fro:.4f}; "
            f"top1_agree={top1:.1%}; KL(dq||fp4)={kl:.5f}"
        )
    print(
        f"[info] mean NLL: dequant={mean_nll_dq:.4f} (ppl {math.exp(mean_nll_dq):.2f}) "
        f"fp4_cute={mean_nll_fp4:.4f} (ppl {math.exp(mean_nll_fp4):.2f}) "
        f"delta={mean_nll_fp4 - mean_nll_dq:+.4f} nats"
    )
    print(
        f"[info] fwd wall time per sample (s): dequant={[f'{t:.2f}' for t in t_dq2]} "
        f"fp4_cute={[f'{t:.2f}' for t in t_fp4]}"
    )

    finish()


if __name__ == "__main__":
    main()
