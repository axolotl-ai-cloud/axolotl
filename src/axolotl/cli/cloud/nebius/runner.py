"""Bootstrap copied into the small Nebius job context."""

import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path, PurePosixPath

import yaml


def validate_config(config, mounts):
    """Reject ephemeral output and implicit recovery before allocating a GPU."""
    output = config.get("output_dir", "training")
    if not isinstance(output, str) or not output or ".." in PurePosixPath(output).parts:
        raise ValueError("output_dir must be a relative path or a path under /outputs")
    path = PurePosixPath(output)
    if path.is_absolute() and not path.is_relative_to("/outputs"):
        raise ValueError("Nebius output_dir must be relative or under /outputs")
    if config.get("auto_resume_from_checkpoints"):
        raise ValueError(
            "Use an explicit resume_from_checkpoint on a mounted volume; automatic resume is unsupported"
        )
    resume = config.get("resume_from_checkpoint")
    if resume:
        if (
            not isinstance(resume, str)
            or ".." in PurePosixPath(resume).parts
            or not any(PurePosixPath(resume).is_relative_to(mount) for mount in mounts)
        ):
            raise ValueError(
                "resume_from_checkpoint must be an explicit path on an attached volume"
            )
        if config.get("save_only_model"):
            raise ValueError("Resuming requires save_only_model: false")
    plugins = config.get("plugins")
    if plugins is not None and (
        not isinstance(plugins, list)
        or not all(isinstance(plugin, str) for plugin in plugins)
    ):
        raise ValueError("plugins must be a list of module/class names")
    if config.get("use_ray"):
        raise ValueError("Nebius supports single-node jobs, not Ray clusters")


def prepare_config(root, output_root, mounts):
    """Resolve mounted output paths and check the selected resume checkpoint."""
    config = yaml.safe_load((root / "train.yaml").read_text(encoding="utf-8"))
    validate_config(config, mounts)
    output = PurePosixPath(config.get("output_dir", "training"))
    relative = output.relative_to("/outputs") if output.is_absolute() else output
    output_path = output_root / str(relative)
    output_path.mkdir(parents=True, exist_ok=True)
    if any(output_path.iterdir()):
        raise ValueError("Output directory is not empty; use a new job/output prefix")
    config["output_dir"] = str(output_path)
    resume = config.get("resume_from_checkpoint")
    if resume:
        checkpoint = Path(resume)
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        if any(
            not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
            for name in required
        ):
            raise ValueError(
                "Selected checkpoint lacks Trainer/optimizer/scheduler state; verify a complete single-node checkpoint"
            )
        state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(state, dict)
            or type(state.get("global_step")) is not int
            or state["global_step"] < 0
        ):
            raise ValueError("Selected checkpoint has invalid Trainer state")
        if not list(checkpoint.glob("rng_state*.pth")):
            raise ValueError("Selected checkpoint lacks RNG state")
        if not list(checkpoint.glob("*safetensors")) and not list(
            checkpoint.glob("*model*.bin")
        ):
            raise ValueError("Selected checkpoint has no model or adapter weights")
    config.setdefault("plugins", [])
    if config["plugins"] is None:
        config["plugins"] = []
    config["plugins"].append("nebius_completion.NebiusCompletionPlugin")
    resolved = root / "resolved.yaml"
    resolved.write_text(yaml.safe_dump(config), encoding="utf-8")
    return resolved, output_path


def main():
    """Execute Axolotl in the provided runtime and preserve training failures."""
    root = Path(__file__).resolve().parent
    os.chdir(root)
    launch = json.loads((root / "launch.json").read_text(encoding="utf-8"))
    output = os.environ.get("NEBIUS_OUTPUT_DIR")
    if not output or not Path(output).is_absolute() or not Path(output).is_dir():
        raise RuntimeError("Nebius did not provide a mounted NEBIUS_OUTPUT_DIR")
    resolved, output_path = prepare_config(root, Path(output), launch["mounts"])
    completion = root / "training-complete.json"
    completion.unlink(missing_ok=True)
    env = os.environ.copy()
    env["AXOLOTL_NEBIUS_COMPLETION_FILE"] = str(completion)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    axolotl = shutil.which("axolotl")
    if not axolotl:
        raise RuntimeError("The selected image must contain Axolotl on PATH")
    command = [axolotl, "train", str(resolved), "--launcher", launch["launcher"]]
    if launch["launcher_args"]:
        command.extend(["--", *launch["launcher_args"]])
    cuda_env = Path("/workspace/axolotl/scripts/cuda13_env.sh")
    if cuda_env.is_file():
        # job run may override the image entrypoint that normally sources this.
        command = [
            "/bin/bash",
            "-c",
            'source /workspace/axolotl/scripts/cuda13_env.sh && exec "$@"',
            "axolotl-nebius",
            *command,
        ]
    subprocess.run(command, cwd=root, env=env, check=True)  # nosec B603
    if not completion.is_file():
        raise RuntimeError(
            "Training exited without normal completion; saved weights may be partial"
        )
    result = json.loads(completion.read_text(encoding="utf-8"))
    (output_path / "nebius-result.json").write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )
    print(f"Training finished. Outputs: {output_path}", flush=True)


if __name__ == "__main__":
    main()
