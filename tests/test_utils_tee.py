import os
import tempfile


def _dummy_cfg(output_dir: str, append: bool = False):
    # Minimal object with attributes used by prepare_debug_log
    class Cfg:
        def __init__(self, out, append):
            self.output_dir = out
            self._append = append

        def get(self, key, default=None):
            if key in {"resume_from_checkpoint", "auto_resume_from_checkpoints"}:
                return self._append
            return default

    return Cfg(output_dir, append)


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_file_only_stream_writes_after_prepare(monkeypatch):
    from axolotl.utils import tee

    with tempfile.TemporaryDirectory() as td:
        # Avoid stdout tee in this test
        monkeypatch.setenv("AXOLOTL_TEE_STDOUT", "0")
        cfg = _dummy_cfg(td, append=False)

        # before prepare: writing to file_only_stream creates no file
        tee.file_only_stream.write("before\n")
        tee.file_only_stream.flush()
        assert not os.path.exists(os.path.join(td, "debug.log"))

        # prepare and write
        path = tee.prepare_debug_log(cfg)
        assert os.path.basename(path) == "debug.log"
        tee.file_only_stream.write("hello\n")
        tee.file_only_stream.flush()

        content = read(path)
        assert "hello" in content

        tee.close_debug_log()


def test_stdout_is_mirrored_after_prepare(capsys, monkeypatch):
    from axolotl.utils import tee

    with tempfile.TemporaryDirectory() as td:
        cfg = _dummy_cfg(td, append=False)
        try:
            # Install tee while capture is disabled so stdout tee wraps real stdout.
            with capsys.disabled():
                monkeypatch.setenv("AXOLOTL_TEE_STDOUT", "1")
                path = tee.prepare_debug_log(cfg)
                import sys

                print("printed-line")
                sys.stdout.flush()

            # Now verify file contains the line
            content = read(path)
            assert "printed-line" in content
        finally:
            tee.close_debug_log()


def test_truncate_vs_append_behavior(monkeypatch):
    from axolotl.utils import tee

    with tempfile.TemporaryDirectory() as td:
        # Avoid stdout tee in this test
        monkeypatch.setenv("AXOLOTL_TEE_STDOUT", "0")
        # First run creates file with A
        cfg = _dummy_cfg(td, append=False)
        _ = tee.prepare_debug_log(cfg)
        try:
            tee.file_only_stream.write("A\n")
            tee.file_only_stream.flush()
        finally:
            tee.close_debug_log()

        # Second run with append=False truncates
        cfg2 = _dummy_cfg(td, append=False)
        path2 = tee.prepare_debug_log(cfg2)
        try:
            tee.file_only_stream.write("B\n")
            tee.file_only_stream.flush()
            content = read(path2)
            assert "A\n" not in content and "B\n" in content
        finally:
            tee.close_debug_log()

        # Third run with append=True preserves existing
        cfg3 = _dummy_cfg(td, append=True)
        path3 = tee.prepare_debug_log(cfg3)
        try:
            tee.file_only_stream.write("C\n")
            tee.file_only_stream.flush()
            content = read(path3)
            assert "B\n" in content and "C\n" in content
        finally:
            tee.close_debug_log()
