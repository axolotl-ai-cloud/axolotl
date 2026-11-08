"""Sweep SonicMoE's quack GEMM configs on synthetic MoE shape families and analyse the winners.

``run`` builds a 2-layer Qwen3-MoE-layout model at one family's real expert dims (tiny vocab and
attention, random weights), routes it through the kernels plugin's SonicMoE experts path, and
runs one forward+backward with full quack sweeps (``quack_autotune_key_buckets: shape`` by
default: one sweep per kernel and weight shape). The winners and every candidate's time go to
``quack_cache_path(--autotune-cache-path)``, the file ``autotune_cache_path`` training runs write.
Expert parallelism is emulated on one GPU: ``--ep 8`` keeps E/8 local experts and the same top-k
over the same tokens, which gives a rank's routed-pair count (T*k) and rows per expert (8*T*k/E).

    python scripts/sonicmoe_quack_sweep.py run --family F5 --mode lora_split --ep 8 \
        --tokens 16384 --autotune-cache-path out/F5-lora_split-ep8-t16384.json

``analyze`` reads the ``*.quack.json`` files of many runs (repeat passes of a run are averaged)
and reports, per kernel and GEMM role, the distinct winners, how close the top configs are, the
smallest config sets that stay within a tolerance of every key's best, and what the
``quack_autotune: false`` rule loses.

    python scripts/sonicmoe_quack_sweep.py analyze out/ --tolerance 2
"""

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time
from collections import defaultdict

# H, I, E, top-k, activation; dims from the configs collected in Nov 2026.
FAMILIES = {
    "F1": dict(hidden=2048, inter=768, experts=128, top_k=8, act="silu"),
    "F1-qwen35": dict(hidden=2048, inter=512, experts=256, top_k=8, act="silu"),
    "F2": dict(hidden=2816, inter=704, experts=128, top_k=8, act="gelu_pytorch_tanh"),
    "F5": dict(hidden=4096, inter=2048, experts=256, top_k=8, act="silu"),
    "F5-qwen35": dict(hidden=4096, inter=1024, experts=512, top_k=10, act="silu"),
    # GLM-5.3-Flash and DeepSeek-V4-Pro shapes; their clamp-10 SwiGLU only changes the
    # gated epilogue, so these run plain SwiGLU
    "F5-glm53flash": dict(hidden=4096, inter=2048, experts=288, top_k=8, act="silu"),
    "F6": dict(hidden=7168, inter=2048, experts=256, top_k=8, act="silu"),
    "F7": dict(hidden=7168, inter=4096, experts=128, top_k=4, act="silu"),
    "F7-dsv4pro": dict(hidden=7168, inter=3072, experts=384, top_k=6, act="silu"),
    "F8": dict(hidden=4096, inter=14336, experts=8, top_k=2, act="silu"),
}
MODES = ("full", "lora", "lora_split")
VOCAB = 1024


def build_model(fam, ep, layers, seq):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    experts = fam["experts"] // ep
    if fam["top_k"] > experts:
        raise ValueError(f"top_k {fam['top_k']} > {experts} local experts at EP={ep}")
    config = AutoConfig.for_model("qwen3_moe")
    config.hidden_size = fam["hidden"]
    config.moe_intermediate_size = fam["inter"]
    config.intermediate_size = fam["inter"]
    config.num_experts = experts
    config.num_experts_per_tok = fam["top_k"]
    config.norm_topk_prob = True
    config.decoder_sparse_step = 1
    config.mlp_only_layers = []
    config.num_hidden_layers = layers
    config.num_attention_heads = 8
    config.num_key_value_heads = 2
    config.head_dim = 128
    config.vocab_size = VOCAB
    config.max_position_embeddings = seq
    config.hidden_act = fam["act"]
    config.tie_word_embeddings = False
    config._experts_implementation = "sonicmoe"
    config._attn_implementation = "sdpa"
    with torch.device("cuda"):
        return AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)


def apply_lora(model, rank):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters=["gate_up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
        ),
    )


def _persist(path, meta):
    from axolotl.integrations.kernels.autotune_cache import write_autotune_cache
    from axolotl.integrations.kernels.sonicmoe_autotune import (
        collect_sonicmoe_records,
        quack_file_metadata,
    )

    records = collect_sonicmoe_records()
    packages, extra = quack_file_metadata()
    write_autotune_cache(path, records, packages, {**extra, "sweep": meta})
    return len(records)


def repeat_sweeps(repeats, log):
    """Benchmark every candidate ``repeats`` times per sweep and keep each one's median.

    One quack sweep moves a config by a few percent between processes, as much as the gap
    between the leading configs, so a single sweep cannot rank them. Wraps the outermost
    ``check_disk_cache``, so ``key`` is the raw key ``Autotuner.__call__`` stores under.
    """
    from axolotl.integrations.kernels.sonicmoe_autotune import sonicmoe_autotuners

    for _name, _module, tuner, is_quack in sonicmoe_autotuners():
        if is_quack and not getattr(tuner, "_sweep_repeats", False):
            tuner._sweep_repeats = True
            tuner.check_disk_cache = _repeating(
                tuner, tuner.check_disk_cache, repeats, log
            )


def _repeating(tuner, orig, repeats, log):
    from axolotl.integrations.kernels.sonicmoe_autotune import _quack_config

    def check_disk_cache(key, configs, bench_fn):
        def bench():
            tables = []
            for _ in range(repeats):
                bench_fn()
                tables.append(dict(tuner.configs_timings))
            ms = {c: [float(t[c][0]) for t in tables] for c in tables[0]}
            log.append(
                {
                    "kernel": _kernel(tuner),
                    "key": [str(k) for k in key],
                    "ms": {
                        short(_quack_config(c)["config"]): [round(v, 5) for v in vals]
                        for c, vals in ms.items()
                    },
                }
            )
            merged = {c: [statistics.median(v), min(v), max(v)] for c, v in ms.items()}
            tuner.cache[key] = min(merged, key=lambda c: merged[c][0])
            tuner.configs_timings = merged

        return orig(key, configs, bench)

    return check_disk_cache


def _kernel(tuner):
    from axolotl.integrations.kernels.autotune_cache import _kernel_name

    return _kernel_name(tuner)


def run(args):
    import torch

    from axolotl.integrations.kernels.autotune_cache import (
        install_timing_capture,
        quack_cache_path,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import (
        configure_lora_split,
        sonic_ops,
    )
    from axolotl.integrations.kernels.plugin import _check_sonicmoe_gpu_compat
    from axolotl.integrations.kernels.quack_autotune_policy import (
        install_quack_key_policy,
    )

    fam = FAMILIES[args.family]
    seq = args.tokens // args.micro_batch
    meta = {
        "family": args.family,
        "mode": args.mode,
        "ep": args.ep,
        "tokens": args.tokens,
        "micro_batch": args.micro_batch,
        "layers": args.layers,
        "rank": None if args.mode == "full" else args.rank,
        "buckets": args.buckets,
        "local_experts": fam["experts"] // args.ep,
        **fam,
        "gpu": torch.cuda.get_device_name(),
    }
    path = quack_cache_path(args.autotune_cache_path)
    print(f"run {json.dumps(meta)} -> {path}", flush=True)

    _check_sonicmoe_gpu_compat()
    register_sonicmoe_experts()
    configure_lora_split(args.mode == "lora_split")
    torch.manual_seed(args.seed)
    model = build_model(fam, args.ep, args.layers, seq)
    if args.mode != "full":
        model = apply_lora(model, args.rank)
    model.train()
    sonic_ops()  # loads the hub kernel and imports the vendored quack's gemm_interface
    install_timing_capture()
    profiles = None
    if args.profiles:
        import yaml

        with open(args.profiles, encoding="utf-8") as handle:
            profiles = yaml.safe_load(handle)
        profiles = profiles.get("autotune_profiles", profiles)
        meta["profiles"] = args.profiles
    if args.plugin:
        from axolotl.integrations.kernels.plugin import KernelsPlugin
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault(
            use_sonicmoe=True,
            autotune_profiles=profiles,
            quack_autotune_key_buckets=args.buckets,
            quack_autotune=not args.no_sweep,
            quack_autotune_default_profiles=not args.no_default_profiles,
        )
        meta["plugin"] = {k: v for k, v in cfg.items() if k != "autotune_profiles"}
        KernelsPlugin().post_model_load(cfg, model)
    else:
        if profiles:
            from axolotl.integrations.kernels.autotune_profiles import (
                apply_autotune_profiles,
            )

            apply_autotune_profiles(profiles)
        install_quack_key_policy(args.buckets, tune=not args.no_sweep)
    if args.repeats > 1:
        meta["repeats"] = args.repeats
        meta["repeat_ms"] = []
        repeat_sweeps(args.repeats, meta["repeat_ms"])

    ids = torch.randint(0, VOCAB, (args.micro_batch, seq), device="cuda")
    start = time.time()
    out = model(input_ids=ids, labels=ids)
    torch.cuda.synchronize()
    meta["forward_seconds"] = round(time.time() - start, 1)
    print(f"forward keys: {_persist(path, meta)}", flush=True)
    start = time.time()
    out.loss.backward()
    torch.cuda.synchronize()
    meta["backward_seconds"] = round(time.time() - start, 1)
    count = _persist(path, meta)
    print(f"forward+backward keys: {count}", flush=True)

    step_ms = []
    for _ in range(args.timed_steps):
        model.zero_grad(set_to_none=True)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        model(input_ids=ids, labels=ids).loss.backward()
        end.record()
        torch.cuda.synchronize()
        step_ms.append(round(begin.elapsed_time(end), 3))
    meta["step_ms"] = step_ms
    meta["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    after = _persist(path, meta)
    meta["keys_after_timed_steps"] = after
    _persist(path, meta)
    print(f"step ms {step_ms}, peak {meta['peak_gib']} GiB, keys {count}->{after}")


# ---------------------------------------------------------------- analysis


def _fields(config_json):
    data = json.loads(config_json)
    return data.get("config", data)


PROFILE_FIELDS = (
    "tile_m",
    "tile_n",
    "cluster_m",
    "cluster_n",
    "is_dynamic_persistent",
    "use_tma_gather",
    "swap_ab",
    "pingpong",
)


def short(fields):
    name = f"{fields['tile_m']}x{fields['tile_n']} c{fields['cluster_m']}x{fields['cluster_n']}"
    name += " P" if fields.get("is_dynamic_persistent") else " np"
    if fields.get("use_tma_gather"):
        name += " tg"
    if fields.get("swap_ab"):
        name += " swap"
    if fields.get("pingpong"):
        name += " pp"
    return name


def _dims(call, name):
    return call.get(f"{name}.shape")


def gemm_role(kernel, call, meta):
    """A name for what a quack call computes in the MoE step, from its argument shapes."""
    hidden, inter = meta["hidden"], meta["inter"]
    gather = "A_idx.shape" in call
    addend = "C.shape" in call
    if kernel == "gemm_gated_tuned":
        return "up+gate fwd" + (" +C" if addend else "")
    if kernel == "gemm_dgated_tuned":
        return "down dgated bwd"
    if kernel != "gemm_tuned":
        return kernel
    out = _dims(call, "out") or []
    b = _dims(call, "B") or []
    if "cu_seqlens_k.shape" in call:
        m, n = out[-2], out[-1]
        if min(m, n) > 128:
            return f"dW{'1' if n == 2 * inter else '2'} varlen-K"
        return f"LoRA grad varlen-K [{m}x{n}]" + (" gather" if gather else "")
    k, n = b[-2], b[-1]
    if n <= 128:
        return f"rank-r out K={k}" + (" gather" if gather else "")
    if k <= 128:
        return f"rank-r in N={n}"
    if (k, n) == (inter, hidden):
        return "down fwd" + (" +C" if addend else "")
    if (k, n) == (2 * inter, hidden):
        return "dX up" + (" +C" if addend else "")
    return f"gemm K={k} N={n}"


# Calls one profile ``match`` can single out, in the order profiles must be listed (the first
# matching profile decides). A ``dimN`` condition only holds when the argument is a tensor, so the
# varlen ones keep every profile to grouped (MoE) calls.
_VARLEN_M = {"cu_seqlens_m.dim0": {"ge": 2}}
_VARLEN_K = {"cu_seqlens_k.dim0": {"ge": 2}}
PROFILE_GROUPS = {
    ("gemm_gated_tuned", "gated +C"): {**_VARLEN_M, "C.dim0": {"ge": 1}},
    ("gemm_gated_tuned", "gated"): _VARLEN_M,
    ("gemm_dgated_tuned", "dgated"): _VARLEN_M,
    ("gemm_tuned", "dW varlen-K"): {
        **_VARLEN_K,
        "out.dim-1": {"gt": 128},
        "out.dim-2": {"gt": 128},
    },
    ("gemm_tuned", "LoRA grad varlen-K"): _VARLEN_K,
    ("gemm_tuned", "rank-r out"): {**_VARLEN_M, "out.dim-1": {"le": 128}},
    ("gemm_tuned", "rank-r in"): {**_VARLEN_M, "B.dim-2": {"le": 128}},
    ("gemm_tuned", "varlen-M"): _VARLEN_M,
}


def profile_group(kernel, role):
    if kernel == "gemm_gated_tuned":
        return kernel, "gated +C" if role.endswith("+C") else "gated"
    if kernel == "gemm_dgated_tuned":
        return kernel, "dgated"
    if role.startswith("dW"):
        return kernel, "dW varlen-K"
    for prefix in ("LoRA grad varlen-K", "rank-r out", "rank-r in"):
        if role.startswith(prefix):
            return kernel, prefix
    return kernel, "varlen-M"


def _heuristic_name(kernel, call):
    import torch

    from axolotl.integrations.kernels.quack_autotune_policy import heuristic_fields

    args = {}
    for key, shape in call.items():
        if key.endswith(".shape"):
            args[key[: -len(".shape")]] = torch.empty(shape, device="meta")
    try:
        return short(heuristic_fields(kernel, args))
    except Exception:  # noqa: BLE001
        return None


def load_runs(paths):
    """{(run, kernel, key json): {"meta", "kernel", "call", "passes": [{config: ms}]}}."""
    keys = {}
    for path in paths:
        payload = json.loads(pathlib.Path(path).read_text())
        meta = payload.get("sweep") or {}
        name = pathlib.Path(path).name.split(".pass")[0].removesuffix(".quack.json")
        for record in payload.get("records", []):
            timings = record.get("timings_ms")
            if record.get("backend") != "quack":
                continue
            if record.get("source") not in ("measured", "profile"):
                continue
            if not timings:
                continue
            ident = (name, record["kernel"], json.dumps(record["key"], sort_keys=True))
            entry = keys.setdefault(
                ident,
                {
                    "run": name,
                    "meta": meta,
                    "kernel": record["kernel"],
                    "call": record.get("call_args") or {},
                    "passes": [],
                    "failed": record.get("failed_configs", 0),
                    "sweep_seconds": record.get("sweep_seconds"),
                },
            )
            fields = {short(_fields(c)): _fields(c) for c in timings}
            entry.setdefault("fields", {}).update(fields)
            entry["passes"].append(
                {short(_fields(c)): ms for c, ms in timings.items() if ms > 0}
            )
    return keys


def summarize(entry):
    """Per config, its regret over each pass's best, averaged over passes (``regret``) and the
    worst pass (``worst``). Normalising within a pass removes the few-percent shift a whole
    process can show."""
    passes = entry["passes"]
    names = set.intersection(*(set(p) for p in passes))
    per_pass = []
    for p in passes:
        best = min(p[c] for c in names)
        per_pass.append({c: (p[c] / best - 1) * 100 for c in names})
    regret = {c: statistics.fmean(r[c] for r in per_pass) for c in names}
    worst = {c: max(r[c] for r in per_pass) for c in names}
    ranked = sorted(names, key=regret.get)
    noise = None
    if len(passes) > 1:
        noise = statistics.median(
            max(r[c] for r in per_pass) - min(r[c] for r in per_pass)
            for c in ranked[:10]
        )
    return {
        "ranked": ranked,
        "regret": {c: regret[c] for c in ranked},
        "worst": worst,
        "best_ms": statistics.fmean(p[ranked[0]] for p in passes),
        "noise_pct": noise,
        "pass_winners": {min(p, key=p.get) for p in passes},
    }


def greedy_cover(entries, tolerance):
    """Smallest-first config set leaving every key within ``tolerance`` % of its best."""
    uncovered = set(range(len(entries)))
    chosen = []
    while uncovered:
        gain = defaultdict(set)
        for i in uncovered:
            for config, pct in entries[i]["summary"]["regret"].items():
                if pct <= tolerance:
                    gain[config].add(i)
        if not gain:
            break
        config = max(gain, key=lambda c: (len(gain[c]), -_mean_regret(entries, c)))
        chosen.append(config)
        uncovered -= gain[config]
    worst = max(
        min(e["summary"]["regret"].get(c, float("inf")) for c in chosen)
        for e in entries
    )
    worst_pass = max(
        min(e["summary"]["worst"].get(c, float("inf")) for c in chosen) for e in entries
    )
    return chosen, worst, worst_pass


def _mean_regret(entries, config):
    vals = [e["summary"]["regret"].get(config, 100.0) for e in entries]
    return statistics.fmean(vals)


def _match(call):
    """A profile ``match`` for one run's call: its weight shape plus which optional tensors."""
    match = {}
    for name in ("A", "B"):
        if f"{name}.shape" in call:
            match[f"{name}.shape"] = call[f"{name}.shape"]
    for name in ("C", "A_idx", "cu_seqlens_k"):
        if f"{name}.shape" in call:
            match[f"{name}.dim0"] = {"ge": 1}
    return match


def write_pins(entries, outdir):
    """Per run, profiles pinning every key to its measured best config (no sweep)."""
    import yaml

    per_run = defaultdict(list)
    for e in entries:
        best = e["summary"]["ranked"][0]
        per_run[e["run"]].append(
            {
                "kernel": e["kernel"],
                "backend": "quack",
                "match": _match(e["call"]),
                "configs": [{f: e["fields"][best][f] for f in PROFILE_FIELDS}],
            }
        )
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for run_name, profiles in per_run.items():
        profiles.sort(key=lambda p: -len(p["match"]))
        (outdir / f"{run_name}.yaml").write_text(
            yaml.safe_dump(
                {"autotune_profiles": profiles},
                sort_keys=False,
                width=200,
                default_flow_style=None,
            )
        )


def write_shortlists(entries, groups, outdir, top=5, within=4.0, pool_top=2, cap=10):
    """Per run, an ``autotune_profiles`` YAML pinning each key's sweep to its contenders.

    A key's contenders: its ``top`` fastest, anything within ``within`` % of its best, the
    ``quack_autotune: false`` rule's pick, and the ``pool_top`` fastest of every key in the
    same profile group (so covers across runs compare the same configs), capped at ``cap``.
    """
    import yaml

    pool = defaultdict(set)
    for (kernel, role), members in groups.items():
        for e in members:
            pool[profile_group(kernel, role)].update(e["summary"]["ranked"][:pool_top])
    per_run = defaultdict(dict)
    for e in entries:
        s = e["summary"]
        group = pool[profile_group(e["kernel"], e["role"])]
        keep = set(s["ranked"][:top]) | {
            c for c in s["ranked"] if s["regret"][c] <= within
        }
        keep |= {c for c in group | {e["heuristic"]} if c in s["regret"]}
        keep = sorted(keep, key=lambda c: s["regret"][c])[:cap]
        match = _match(e["call"])
        ident = (e["kernel"], json.dumps(match, sort_keys=True))
        have = per_run[e["run"]].setdefault(
            ident, {"kernel": e["kernel"], "match": match, "configs": {}}
        )
        for c in keep:
            have["configs"][c] = {f: e["fields"][c][f] for f in PROFILE_FIELDS}
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for run_name, profiles in per_run.items():
        listed = [
            {
                "kernel": p["kernel"],
                "backend": "quack",
                "match": p["match"],
                "configs": list(p["configs"].values()),
            }
            # the first matching profile decides, so the most specific match goes first
            for p in sorted(profiles.values(), key=lambda p: -len(p["match"]))
        ]
        (outdir / f"{run_name}.yaml").write_text(
            yaml.safe_dump(
                {"autotune_profiles": listed},
                sort_keys=False,
                width=200,
                default_flow_style=None,
            )
        )


ROLE_COLUMNS = (
    ("gemm_gated_tuned", "up+gate fwd"),
    ("gemm_tuned", "down fwd"),
    ("gemm_dgated_tuned", "down dgated bwd"),
    ("gemm_tuned", "dX up"),
    ("gemm_tuned", "dW1 varlen-K"),
    ("gemm_tuned", "dW2 varlen-K"),
    ("gemm_tuned", "rank-r out"),
    ("gemm_tuned", "rank-r in"),
    ("gemm_tuned", "LoRA grad varlen-K"),
)


def matrix(entries, tol):
    """Runs x GEMM roles: winner, configs within ``tol`` %, and the rule's regret.

    Roles a run has several keys for (rank-r, LoRA grads) show the worst of them."""
    cells = defaultdict(list)
    for e in entries:
        role = e["role"].removesuffix(" +C")
        for kernel, column in ROLE_COLUMNS:
            if kernel == e["kernel"] and role.startswith(column):
                cells[(e["run"], column)].append(e)
    runs = sorted({run for run, _ in cells})
    head = [c for _, c in ROLE_COLUMNS if any((r, c) in cells for r in runs)]
    lines = [
        f"## Winners per run (best; configs within {tol:g}%; rule regret)",
        "",
        "| run | " + " | ".join(head) + " |",
        "|---" * (len(head) + 1) + "|",
    ]
    for run in runs:
        row = []
        for column in head:
            es = cells.get((run, column))
            if not es:
                row.append("")
                continue
            e = max(es, key=lambda e: e["summary"]["regret"].get(e["heuristic"], 99))
            s = e["summary"]
            within = sum(1 for v in s["regret"].values() if v <= tol)
            rule = s["regret"].get(e["heuristic"])
            rule_txt = "n/a" if rule is None else f"{rule:+.0f}%"
            more = f" (x{len(es)})" if len(es) > 1 else ""
            row.append(f"{s['ranked'][0]}; {within}; {rule_txt}{more}")
        lines.append(f"| {run} | " + " | ".join(row) + " |")
    return lines


def evaluate(entries, profiles, tol):
    """Per profile: the best regret its configs leave each measured key it would match.

    Matching goes through Axolotl's own ``_quack_matches`` on meta tensors of the recorded call
    shapes, in profile order; the first matching profile takes the key. Only configs measured for
    a key count, so a key whose profile pins only unmeasured configs gets infinite regret. Training
    also tries the next profile when none of a profile's configs is valid for the call; the
    timings cannot tell invalid from unmeasured, so such a key also shows up here as infinite.
    """
    import torch

    from axolotl.integrations.kernels.autotune_profiles import _quack_matches

    rows = defaultdict(list)
    for e in entries:
        args = {
            k.removesuffix(".shape"): torch.empty(v, device="meta")
            if k.endswith(".shape")
            else v
            for k, v in e["call"].items()
        }
        hit = None
        for i, p in enumerate(profiles):
            if p["kernel"] != e["kernel"] or not _quack_matches(p.get("match"), args):
                continue
            chosen = [
                c
                for c in e["summary"]["regret"]
                if any(
                    all(e["fields"][c].get(f) == v for f, v in want.items())
                    for want in p["configs"]
                )
            ]
            regret = (
                min(e["summary"]["regret"][c] for c in chosen) if chosen else math.inf
            )
            hit = (i, regret, len(chosen))
            break
        rows[hit[0] if hit else None].append((e, hit))
    lines = [
        "| profile | kernel | match | keys | distinct winners | swept per key | median regret | "
        f"max regret | rule median / max | keys over {tol:g}% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i in sorted(rows, key=lambda i: (i is None, i)):
        hits = rows[i]
        if i is None:
            names = sorted({f"{e['run']}:{e['role']}" for e, _ in hits})
            lines.append(f"| none | | | {len(hits)} | full search | | | {names} |")
            continue
        p = profiles[i]
        regrets = [h[1] for _, h in hits]
        over = [f"{e['run']}:{e['role']} {h[1]:+.1f}%" for e, h in hits if h[1] > tol]
        swept = sorted({h[2] for _, h in hits})
        winners = len({e["summary"]["ranked"][0] for e, _ in hits})
        rule = [
            e["summary"]["regret"][e["heuristic"]]
            for e, _ in hits
            if e["heuristic"] in e["summary"]["regret"]
        ]
        rule_txt = (
            f"{statistics.median(rule):+.1f}% / {max(rule):+.1f}%" if rule else "n/a"
        )
        lines.append(
            f"| {i} | {p['kernel']} | {json.dumps(p.get('match'))} | {len(hits)} | "
            f"{winners} | {swept} | {statistics.median(regrets):+.1f}% | "
            f"{max(regrets):+.1f}% | {rule_txt} | {'; '.join(over)} |"
        )
    return lines


def _gpu_label(entries):
    """``H100`` for runs on an "NVIDIA H100 80GB HBM3", the profiles' ``gpu`` substring."""
    names = {e["meta"].get("gpu") or "" for e in entries}
    words = {n.removeprefix("NVIDIA ").split(" ")[0] for n in names}
    return words.pop() if len(words) == 1 else None


def analyze(args):
    paths = []
    for target in args.paths:
        target = pathlib.Path(target)
        paths += sorted(target.rglob("*.quack.json")) if target.is_dir() else [target]
    keys = load_runs(paths)
    groups = defaultdict(list)
    for entry in keys.values():
        entry["summary"] = summarize(entry)
        entry["role"] = gemm_role(entry["kernel"], entry["call"], entry["meta"])
        entry["heuristic"] = _heuristic_name(entry["kernel"], entry["call"])
        groups[(entry["kernel"], entry["role"])].append(entry)

    if args.shortlist:
        write_shortlists(keys.values(), groups, args.shortlist)
    if args.oracle:
        write_pins(keys.values(), args.oracle)
    tol = args.tolerance
    out = []
    out.append(
        "| kernel | role | run | B | best | ms | 2nd | 3rd | 5th | within "
        f"{tol:g}% | noise | rule (regret) |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (kernel, role), entries in sorted(groups.items()):
        for e in sorted(entries, key=lambda e: e["run"]):
            s = e["summary"]
            r = [s["regret"][c] for c in s["ranked"]]
            within = [c for c in s["ranked"] if s["regret"][c] <= tol]
            rule = e["heuristic"]
            rule_txt = (
                f"{rule} ({s['regret'][rule]:+.1f}%)"
                if rule in s["regret"]
                else f"{rule} (n/a)"
            )
            b = _dims(e["call"], "B") or _dims(e["call"], "out")
            noise = "" if s["noise_pct"] is None else f"{s['noise_pct']:.1f}%"
            nth = [f"{r[i]:.1f}" if len(r) > i else "" for i in (1, 2, 4)]
            out.append(
                f"| {kernel} | {role} | {e['run']} | {b} | {s['ranked'][0]} | "
                f"{s['best_ms']:.3f} | {' | '.join(nth)} | {len(within)} | {noise} | "
                f"{rule_txt} |"
            )
    out.append("")
    out.append(f"## Covers (every key within {tol:g}% of its best)")
    out.append("")
    out.append(
        "| kernel | role | keys | distinct winners | cover | worst regret | worst pass |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for (kernel, role), entries in sorted(groups.items()):
        winners = {e["summary"]["ranked"][0] for e in entries}
        cover, worst, worst_pass = greedy_cover(entries, tol)
        out.append(
            f"| {kernel} | {role} | {len(entries)} | {len(winners)} | "
            f"{'; '.join(cover)} | {worst:.1f}% | {worst_pass:.1f}% |"
        )
    by_group = defaultdict(list)
    for (kernel, role), entries in groups.items():
        by_group[profile_group(kernel, role)] += entries
    out.append("")
    out.append(f"## Profile groups (one `match` each, every key within {tol:g}%)")
    out.append("")
    out.append(
        "| kernel | group | keys | distinct winners | cover | worst regret | worst pass |"
    )
    out.append("|---|---|---|---|---|---|---|")
    gpu = args.gpu or _gpu_label(keys.values())
    profiles = []
    for group in PROFILE_GROUPS:
        entries = by_group.get(group)
        if not entries:
            continue
        cover, worst, worst_pass = greedy_cover(entries, tol)
        winners = {e["summary"]["ranked"][0] for e in entries}
        out.append(
            f"| {group[0]} | {group[1]} | {len(entries)} | {len(winners)} | "
            f"{'; '.join(cover)} | {worst:.1f}% | {worst_pass:.1f}% |"
        )
        fields = {}
        for e in entries:
            fields.update(e["fields"])
        profiles.append(
            {
                "kernel": group[0],
                "backend": "quack",
                "gpu": gpu,
                "match": json.loads(json.dumps(PROFILE_GROUPS[group])),
                "configs": [{f: fields[c][f] for f in PROFILE_FIELDS} for c in cover],
            }
        )
    out += ["", *matrix(keys.values(), tol)]
    if args.evaluate:
        import yaml

        with open(args.evaluate, encoding="utf-8") as handle:
            proposal = yaml.safe_load(handle)
        proposal = proposal.get("autotune_profiles", proposal)
        out += ["", f"## {args.evaluate}", "", *evaluate(keys.values(), proposal, tol)]
    text = "\n".join(out)
    print(text)
    if args.yaml:
        import yaml

        pathlib.Path(args.yaml).write_text(
            yaml.safe_dump(
                {"autotune_profiles": profiles},
                sort_keys=False,
                width=200,
                default_flow_style=None,
            )
        )
    if args.json:
        dump = [
            {
                "run": e["run"],
                "meta": e["meta"],
                "kernel": e["kernel"],
                "role": e["role"],
                "call": e["call"],
                "best": e["summary"]["ranked"][0],
                "best_ms": e["summary"]["best_ms"],
                "noise_pct": e["summary"]["noise_pct"],
                "regret_pct": {
                    c: round(v, 2) for c, v in e["summary"]["regret"].items()
                },
                "rule": e["heuristic"],
            }
            for e in keys.values()
        ]
        pathlib.Path(args.json).write_text(json.dumps(dump, indent=1))


def steps(args):
    """Per run and pass: the first forward+backward's seconds (sweeps included), the median
    step time after it, and the step time relative to the first pass."""
    rows = defaultdict(dict)
    for target in args.paths:
        target = pathlib.Path(target)
        for path in (
            sorted(target.rglob("*.quack.json")) if target.is_dir() else [target]
        ):
            payload = json.loads(path.read_text())
            meta = payload.get("sweep") or {}
            if "step_ms" not in meta:
                continue
            run_name, has_pass, rest = path.name.partition(".pass")
            run_name = run_name.removesuffix(".quack.json")
            sources = {r.get("source") for r in payload.get("records", [])}
            label = meta.get("profiles") or (
                "rule" if "heuristic" in sources else "sweep"
            )
            if meta.get("plugin"):
                label = f"plugin, {label}"
            rows[run_name][int(rest.split(".")[0]) if has_pass else 1] = (
                pathlib.Path(label).name,
                meta["forward_seconds"] + meta["backward_seconds"],
                statistics.median(meta["step_ms"]),
            )
    print("| run | pass | variant | first step s | median step ms | vs pass 1 |")
    print("|---|---|---|---|---|---|")
    for run_name, passes in sorted(rows.items()):
        base = passes[min(passes)][2]
        for n, (label, first, ms) in sorted(passes.items()):
            print(
                f"| {run_name} | {n} | {label} | {first:.0f} | {ms:.2f} | "
                f"{(ms / base - 1) * 100:+.1f}% |"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--family", choices=sorted(FAMILIES), required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--buckets", choices=("shape", "pow2", "none"), default="shape")
    p.add_argument("--no-sweep", action="store_true", help="quack_autotune: false")
    p.add_argument("--profiles", help="YAML with autotune_profiles to apply first")
    p.add_argument(
        "--plugin",
        action="store_true",
        help="set up profiles and the key policy through KernelsPlugin.post_model_load "
        "(built-in profiles included)",
    )
    p.add_argument(
        "--no-default-profiles",
        action="store_true",
        help="with --plugin: quack_autotune_default_profiles: false",
    )
    p.add_argument("--timed-steps", type=int, default=3)
    p.add_argument(
        "--repeats", type=int, default=1, help="benchmarks per candidate (median kept)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--autotune-cache-path", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("paths", nargs="+")
    a.add_argument("--tolerance", type=float, default=2.0)
    a.add_argument("--json", help="also write per-key summaries here")
    a.add_argument("--yaml", help="also write the covers as autotune_profiles here")
    a.add_argument("--evaluate", help="score an autotune_profiles YAML on the runs")
    a.add_argument("--gpu", help="the --yaml profiles' gpu (default: from the runs)")
    a.add_argument(
        "--oracle", help="write per-run profiles pinning each key to its best"
    )
    a.add_argument(
        "--shortlist", help="write per-run profiles pinning each key to its contenders"
    )
    st = sub.add_parser("steps")
    st.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    if args.cmd == "steps":
        steps(args)
    elif args.cmd == "run":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        run(args)
    else:
        analyze(args)


if __name__ == "__main__":
    sys.exit(main())
