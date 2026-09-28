"""Cache preparation preserves existing assets and never extracts failed downloads."""

import io
import os
import subprocess
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("failures", [0, 1, 3])
def test_hf_cache_download_recovery(tmp_path, failures):
    cache = tmp_path / "cache" / "hub"
    cache.mkdir(parents=True)
    (cache / "existing").write_text("keep")
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        entry = tarfile.TarInfo("hub/new")
        entry.size = 3
        tar.addfile(entry, io.BytesIO(b"new"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "assert args[args.index('--continue-at') + 1] == '-'\n"
        "assert 0 < int(args[args.index('--max-time') + 1]) <= 900\n"
        "dest = pathlib.Path(args[args.index('--output') + 1])\n"
        "counter = pathlib.Path(os.environ['TEST_ATTEMPTS'])\n"
        "attempt = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(attempt))\n"
        "source = pathlib.Path(os.environ['TEST_ARCHIVE']).read_bytes()\n"
        "if attempt > 1:\n"
        "    assert dest.read_bytes() == source[:100]\n"
        "if attempt <= int(os.environ['TEST_FAILURES']):\n"
        "    dest.write_bytes(source[:100])\n"
        "    sys.exit(56)\n"
        "with dest.open('ab') as stream:\n"
        "    stream.write(source[dest.stat().st_size:])\n"
    )
    curl.chmod(0o755)
    unzstd = bin_dir / "unzstd"
    unzstd.write_text("#!/bin/sh\ncat\n")
    unzstd.chmod(0o755)
    counter = tmp_path / "attempts"
    result = subprocess.run(
        ["bash", str(Path(__file__).resolve().parents[1] / "cicd/prepare_hf_cache.sh")],
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "HF_HOME": str(cache.parent),
            "TMPDIR": str(tmp_path),
            "TEST_ARCHIVE": str(archive),
            "TEST_ATTEMPTS": str(counter),
            "TEST_FAILURES": str(failures),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert int(counter.read_text()) == min(failures + 1, 3)
    assert (cache / "existing").read_text() == "keep"
    assert not list(tmp_path.glob("tmp.*"))
    if failures < 3:
        assert result.returncode == 0, result.stderr
        assert (cache / "new").read_text() == "new"
    else:
        assert result.returncode != 0
        assert not (cache / "new").exists()
