"""
Runpod train entrypoint
"""

import asyncio


async def train(config_path: str, gpu_id: str = "0", preprocess: bool = True):
    """
    Run preprocessing (if enabled) and training with the given config file
    :param config_path: Path to the YAML config file
    :param gpu_id: GPU ID to use (default: "0")
    :param preprocess: Whether to run preprocessing (default: True)

    """
    # First check if preprocessing is needed
    if preprocess:
        # Preprocess command
        preprocess_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} axolotl preprocess {config_path}"
        )
        process = await asyncio.create_subprocess_shell(
            preprocess_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        if process.stdout is not None:
            async for line in process.stdout:
                yield f"Preprocessing: {line.decode().strip()}"
        await process.wait()
        yield "Preprocessing completed."
    else:
        yield "Skipping preprocessing step."

    # Training command
    train_cmd = f"axolotl train {config_path}"
    process = await asyncio.create_subprocess_shell(
        train_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )

    if process.stdout is not None:
        async for line in process.stdout:
            yield f"Training: {line.decode().strip()}"
    await process.wait()
