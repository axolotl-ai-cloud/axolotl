"""Pin a base checkpoint by identity and layout without copying its weights.

The merge reads the base on every publish, so it must not change under a chain.
``pin_base`` records each tensor's name, dtype and shape from the safetensors
headers, derives an identity from those headers, the file stats and the config,
and hard-links the files into the publisher's workdir so a later delete of the
source cannot break the chain. Where links are impossible (another filesystem)
the source is referenced in place and re-checked before every merge.
"""

import errno
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "pinned-base.json"

SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}
FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
SUPPORTED_DTYPES = FLOAT_DTYPES | {"uint8", "float8_e4m3fn"}
NVFP4_LEAVES = ("weight_scale", "weight_scale_2")
_BLOB_RE = re.compile(r"^[0-9a-f]{64}$")


def torch_dtype_name(dtype: str) -> str:
    """Map a safetensors header dtype to the torch name the chain records."""
    try:
        return SAFETENSORS_DTYPES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported safetensors dtype {dtype}") from exc


def read_header(path: Path) -> tuple[bytes, dict]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        raw = handle.read(length)
    return raw, json.loads(raw)


def safetensors_layout(path: Path) -> dict[str, tuple[str, list[int]]]:
    _, header = read_header(path)
    return {
        key: (torch_dtype_name(entry["dtype"]), list(entry["shape"]))
        for key, entry in header.items()
        if key != "__metadata__"
    }


def base_format(config: dict) -> str:
    """``unquantized`` or ``modelopt``; any other quantization is refused."""
    quantization = config.get("quantization_config")
    if not quantization:
        return "unquantized"
    method = quantization.get("quant_method")
    if method == "modelopt":
        return "modelopt"
    raise ValueError(
        "Checkpoint publishing supports unquantized or modelopt NVFP4 bases, "
        f"not quant_method={method!r}"
    )


def check_layout(layout: dict[str, tuple[str, list[int]]], fmt: str):
    """Refuse dtypes and quantized tensors the format-preserving merge cannot rebuild."""
    for name, (dtype, _) in layout.items():
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported base tensor dtype {dtype} for {name}")
        if dtype in FLOAT_DTYPES:
            continue
        if fmt != "modelopt":
            raise ValueError(
                f"Base tensor {name} is {dtype} but the base has no modelopt "
                "quantization_config"
            )
        if dtype == "uint8":
            stem = name.removesuffix(".weight")
            if stem == name or any(
                f"{stem}.{leaf}" not in layout for leaf in NVFP4_LEAVES
            ):
                raise ValueError(
                    f"Packed tensor {name} lacks the NVFP4 weight_scale/weight_scale_2"
                )


def _stat(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _content_id(path: Path) -> str | None:
    """Hugging Face cache files are symlinks to blobs named by their sha256."""
    if path.is_symlink():
        name = path.resolve().name
        if _BLOB_RE.match(name):
            return name
    return None


@dataclass
class PinnedBase:
    """The base checkpoint a chain merges against, and how to check it is unchanged."""

    source: Path
    path: Path
    identity: str
    format: str
    layout: dict[str, tuple[str, list[int]]]
    files: dict[str, str]
    config: dict
    linked: bool
    stats: dict[str, list[int]] = field(default_factory=dict)

    def verify(self):
        for name, expected in self.stats.items():
            path = self.path / name
            if not path.is_file() or _stat(path) != expected:
                raise RuntimeError(
                    f"Pinned base file {path} changed or disappeared; publishing it "
                    "would break the chain"
                )

    @property
    def num_experts(self) -> int | None:
        for config in (self.config, self.config.get("text_config") or {}):
            for key in ("num_experts", "num_local_experts", "n_routed_experts"):
                if isinstance(config.get(key), int):
                    return config[key]
        return None


def _scan(source: Path) -> tuple[dict, dict, dict, list[str]]:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    index = source / INDEX_NAME
    if index.is_file():
        expected = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(expected.values()))
    else:
        expected = None
        names = ["model.safetensors"]
    layout: dict[str, tuple[str, list[int]]] = {}
    files: dict[str, str] = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError("Base shards must be in the model directory")
        for key, entry in safetensors_layout(source / name).items():
            if key in layout or (expected is not None and expected.get(key) != name):
                raise ValueError("Invalid base checkpoint index")
            layout[key] = entry
            files[key] = name
    if expected is not None and layout.keys() != expected.keys():
        raise ValueError("Base checkpoint index is incomplete")
    if not layout:
        raise ValueError("Base checkpoint has no tensors")
    pinned = names + ["config.json"] + ([INDEX_NAME] if expected is not None else [])
    return config, layout, files, pinned


def _identity(source: Path, names: list[str], revision: str | None) -> str:
    digest = hashlib.sha256(json.dumps({"revision": revision}).encode())
    for name in sorted(names):
        path = source / name
        digest.update(name.encode())
        if (blob := _content_id(path)) is not None:
            digest.update(blob.encode())
            continue
        digest.update(json.dumps(_stat(path)).encode())
        if name.endswith(".safetensors"):
            digest.update(read_header(path)[0])
        else:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _link_all(source: Path, target: Path, names: list[str]) -> bool:
    try:
        for name in names:
            os.link((source / name).resolve(), target / name)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP}:
            raise
        return False
    return True


def pin_base(
    source: Path | str, workdir: Path | str, *, revision: str | None = None
) -> PinnedBase:
    """Validate the base's format and layout and pin it under ``workdir``."""
    source = Path(source)
    workdir = Path(workdir)
    config, layout, files, names = _scan(source)
    fmt = base_format(config)
    check_layout(layout, fmt)
    identity = _identity(source, names, revision)
    pinned = workdir / f".base-{identity[:32]}"
    linked = pinned.is_dir()
    if linked:
        manifest = json.loads((pinned / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise RuntimeError(f"Pinned base {pinned} does not match {source}")
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".base-staging-", dir=workdir))
        try:
            if _link_all(source, staging, names):
                (staging / MANIFEST_NAME).write_text(
                    json.dumps({"identity": identity}), encoding="utf-8"
                )
                staging.rename(pinned)
                linked = True
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    path = pinned if linked else source
    base = PinnedBase(
        source=source,
        path=path,
        identity=identity,
        format=fmt,
        layout=layout,
        files=files,
        config=config,
        linked=linked,
        stats={name: _stat(path / name) for name in names},
    )
    return base
