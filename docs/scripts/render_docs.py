"""Render the documentation in isolated workers, preserving project navigation."""

import argparse
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def partition_inputs(inputs: list[str], jobs: int) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for name in inputs:
        parts = Path(name).parts
        if len(parts) > 2 and parts[0] == "docs":
            target = str(Path(*parts[:2]))
        elif len(parts) > 1 and parts[0] != "docs":
            target = parts[0]
        else:
            target = name
        groups.setdefault(target, []).append(name)

    assignments: list[list[str]] = [[] for _ in range(jobs)]
    loads = [0] * jobs
    for target, members in sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        worker = loads.index(min(loads))
        assignments[worker].append(target)
        # Individual renders repeat project indexing and asset handling.
        loads[worker] += len(members) + 5
    return [targets for targets in assignments if targets]


def merge_sites(sites: list[Path], destination: Path, home_worker: int) -> None:
    search: dict[str, dict] = {}
    destination.mkdir()
    for worker, site in enumerate(sites):
        for source in sorted(site.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(site)
            if relative == Path("index.html") and worker != home_worker:
                continue
            if relative == Path("search.json"):
                for entry in json.loads(source.read_text()):
                    href = entry["href"]
                    if href in search and search[href] != entry:
                        raise ValueError(f"Conflicting search entry: {href}")
                    search[href] = entry
                continue
            target = destination / relative
            if target.exists():
                if target.read_bytes() != source.read_bytes():
                    raise ValueError(f"Conflicting output: {relative}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    if not (destination / "index.html").is_file() or not search:
        raise ValueError("Missing home page or search index")
    (destination / "search.json").write_text(
        json.dumps(list(search.values()), ensure_ascii=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    project = Path(__file__).resolve().parents[2]
    quarto = shutil.which("quarto")
    if quarto is None:
        raise RuntimeError("quarto is not on PATH")
    if args.jobs == 1:
        subprocess.run([quarto, "render"], cwd=project, check=True)  # nosec B603
        return
    config = yaml.safe_load((project / "_quarto.yml").read_text())
    if config.get("website", {}).get("site-url"):
        raise ValueError("Parallel rendering does not yet support sitemap merging")
    if config["project"].get("post-render"):
        raise ValueError("Parallel rendering does not support post-render hooks")
    for script in config["project"].pop("pre-render", []):
        if not script.endswith(".py") or not (project / script).is_file():
            raise ValueError(f"Expected a Python pre-render script: {script}")
        subprocess.run([sys.executable, script], cwd=project, check=True)  # nosec B603

    # Generators can update the project configuration as well as its pages.
    config = yaml.safe_load((project / "_quarto.yml").read_text())
    config["project"].pop("pre-render", None)
    inventory = json.loads(
        subprocess.check_output([quarto, "inspect"], cwd=project, text=True)  # nosec B603
    )
    inputs = sorted(
        str(Path(name).relative_to(project)) for name in inventory["files"]["input"]
    )
    if "index.qmd" not in inputs:
        raise ValueError("Expected index.qmd as the site home page")
    assignments = partition_inputs(inputs, args.jobs)
    home_worker = next(
        i for i, targets in enumerate(assignments) if "index.qmd" in targets
    )
    with tempfile.TemporaryDirectory(prefix="axolotl-docs-") as temporary:
        temporary_path = Path(temporary)
        workers = []
        for i in range(len(assignments)):
            worker = temporary_path / f"worker-{i}"
            shutil.copytree(
                project,
                worker,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".quarto",
                    "_site",
                    ".venv",
                    "__pycache__",
                    ".mypy_cache",
                    ".pytest_cache",
                    ".ruff_cache",
                ),
            )
            (worker / "_quarto.yml").write_text(yaml.safe_dump(config, sort_keys=False))
            workers.append(worker)

        def render(i: int) -> None:
            worker = workers[i]
            log = temporary_path / f"worker-{i}.log"
            env = dict(os.environ, PYTHONPATH=str(worker / "src"))
            with log.open("w") as stream:
                result = subprocess.run(  # nosec B603
                    [quarto, "render", *assignments[i], "--no-clean"],
                    cwd=worker,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                print(log.read_text(), file=sys.stderr)
                result.check_returncode()
            print(f"Worker {i + 1}/{len(workers)} finished", flush=True)

        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            list(pool.map(render, range(len(workers))))
        merged = temporary_path / "merged"
        merge_sites([worker / "_site" for worker in workers], merged, home_worker)
        output = project / "_site"
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(merged), output)
    print(f"Rendered {len(inputs)} pages to {output}")


if __name__ == "__main__":
    main()
