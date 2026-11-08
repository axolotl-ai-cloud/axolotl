"""Run ``scripts/sonicmoe_quack_sweep.py`` on Modal B200s. Manual; no CI workflow calls this.

Each run spec ``family:mode:ep:tokens`` gets its own container and a fresh ``QUACK_CACHE_DIR``,
and runs twice, each pass in a fresh process: the first compiles and sweeps every candidate,
the second reuses the compiled kernels and re-times them all (``QUACK_FORCE_CACHE_UPDATE=1``),
which gives the benchmark noise. Cache files and logs land on the ``axolotl-quack-sweeps``
volume under ``<tag>/``. ``--extra`` is passed to every run, with ``{name}`` replaced by the
run's name (``<family>-<mode>-ep<ep>-t<tokens>``), e.g. a per-run ``--profiles`` file; ``||``
separates per-pass variants (pass N takes the Nth, the last one repeats).

    GPU_TYPE=B200 GITHUB_REF=refs/heads/<branch> GITHUB_SHA=<sha> <CI image env> \
      modal run --detach -m cicd.quack_profile_sweep --tag <tag> \
        --runs F1:full:1:16384,F5:lora_split:8:16384
    modal volume get axolotl-quack-sweeps <tag> ./sweeps
"""

import os
import pathlib
import shlex
import subprocess  # nosec
import tempfile
import time

import modal

from .single_gpu import GPU_CONFIG, VOLUME_CONFIG, app, cicd_image, refresh_source

ROOT = "/workspace/axolotl"
RESULTS = "/workspace/sweeps"
results_volume = modal.Volume.from_name("axolotl-quack-sweeps", create_if_missing=True)


def _last_candidate(log: pathlib.Path) -> str | None:
    last = None
    with open(log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Autotuning kernel"):
                last = line.strip()
    return last


@app.function(
    image=cicd_image,
    gpu=GPU_CONFIG,
    timeout=4 * 60 * 60,
    cpu=8.0,
    memory=131072,
    volumes={**VOLUME_CONFIG, RESULTS: results_volume},
)
def sweep(spec: str, tag: str, passes: int = 2, extra: str = ""):
    refresh_source(ROOT)
    subprocess.run(["bash", "./cicd/prepare_hf_cache.sh"], cwd=ROOT, check=False)  # nosec
    family, mode, ep, tokens = spec.split(":")
    name = f"{family}-{mode}-ep{ep}-t{tokens}"
    outdir = pathlib.Path(RESULTS, tag)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = tempfile.mkdtemp(prefix="quack-")
    variants = [v.strip() for v in extra.split("||")]
    status: dict[int, dict[str, object]] = {}
    for n in range(1, passes + 1):
        env = {**os.environ, "QUACK_CACHE_DIR": cache, "QUACK_PRINT_AUTOTUNING": "1"}
        if n > 1:
            env["QUACK_FORCE_CACHE_UPDATE"] = "1"
        log = outdir / f"{name}.pass{n}.log"
        cmd = [
            "python",
            "scripts/sonicmoe_quack_sweep.py",
            "run",
            *("--family", family, "--mode", mode, "--ep", ep, "--tokens", tokens),
            *("--autotune-cache-path", str(outdir / f"{name}.pass{n}.json")),
            *shlex.split(variants[min(n, len(variants)) - 1].replace("{name}", name)),
        ]
        start = time.time()
        with open(log, "w", encoding="utf-8") as handle:
            rc = subprocess.call(  # nosec
                cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
        status[n] = {"rc": rc, "seconds": round(time.time() - start)}
        if rc:
            status[n]["last_candidate"] = _last_candidate(log)
        results_volume.commit()
        print(f"{name} pass {n}: {status[n]}", flush=True)
    return name, status


@app.local_entrypoint()
def main(runs: str, tag: str, passes: int = 2, extra: str = ""):
    specs = [spec for spec in runs.split(",") if spec]
    calls = [(spec, tag, passes, extra) for spec in specs]
    for result in sweep.starmap(calls, return_exceptions=True):
        print(result, flush=True)
