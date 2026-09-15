import logging
import tempfile

import pytest


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(autouse=True)
def _reset_logging_state():
    # Ensure a clean slate for logging between tests
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.shutdown()
    # Note: dictConfig in configure_logging will set up handlers again
    yield
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.shutdown()


def test_axolotl_logs_captured_at_all_levels(monkeypatch):
    from axolotl.logging_config import configure_logging
    from axolotl.utils import tee
    from axolotl.utils.logging import get_logger

    with tempfile.TemporaryDirectory() as td:
        # Avoid stdout tee in this test to simplify interaction with pytest capture
        monkeypatch.setenv("AXOLOTL_TEE_STDOUT", "0")
        configure_logging()
        path = tee.prepare_debug_log(
            type("Cfg", (), {"output_dir": td, "get": lambda *_: False})
        )

        log = get_logger("axolotl.test")
        log.info("AX-INFO")
        log.debug("AX-DEBUG")
        tee.file_only_stream.flush()

        data = read(path)
        assert "AX-INFO" in data
        assert "AX-DEBUG" in data
        tee.close_debug_log()


def test_third_party_logs_filtered_and_warning_captured(monkeypatch):
    from axolotl.logging_config import configure_logging
    from axolotl.utils import tee

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("AXOLOTL_TEE_STDOUT", "0")
        configure_logging()
        path = tee.prepare_debug_log(
            type("Cfg", (), {"output_dir": td, "get": lambda *_: False})
        )

        # Third-party logger (non-axolotl)
        other = logging.getLogger("thirdparty.lib")
        other.info("TP-INFO")
        other.warning("TP-WARN")

        # Simulate Python warnings routed through logging
        logging.getLogger("py.warnings").warning("PY-WARN")

        # Push through buffers
        tee.file_only_stream.flush()

        data = read(path)
        # INFO from non-axolotl should be filtered out (not present)
        assert "TP-INFO" not in data
        # WARNING+ should be present
        assert "TP-WARN" in data
        # Python warnings captured (via py.warnings logger)
        assert "PY-WARN" in data
        tee.close_debug_log()
        tee.close_debug_log()


def test_prepare_debug_log_idempotent_and_no_duplicate(monkeypatch):
    from axolotl.logging_config import configure_logging
    from axolotl.utils import tee
    from axolotl.utils.logging import get_logger

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("AXOLOTL_TEE_STDOUT", "0")
        configure_logging()
        cfg = type("Cfg", (), {"output_dir": td, "get": lambda *_: False})
        p1 = tee.prepare_debug_log(cfg)
        p2 = tee.prepare_debug_log(cfg)
        assert p1 == p2

        log = get_logger("axolotl.test")
        marker = "UNIQUE-MARKER-12345"
        log.info(marker)
        tee.file_only_stream.flush()

        data = read(p1)
        # Ensure the marker appears once (not duplicated via propagation)
        assert data.count(marker) == 1
        tee.close_debug_log()


def test_hub_unauthenticated_nag_suppressed(monkeypatch):
    from axolotl.logging_config import configure_logging
    from axolotl.utils import tee

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("AXOLOTL_TEE_STDOUT", "0")
        configure_logging()
        path = tee.prepare_debug_log(
            type("Cfg", (), {"output_dir": td, "get": lambda *_: False})
        )

        hub_http = logging.getLogger("huggingface_hub.utils._http")
        hub_http.warning(
            "Warning: You are sending unauthenticated requests to the HF Hub."
            " Please set a HF_TOKEN to enable higher rate limits and faster downloads."
        )
        hub_http.warning("Retrying in 2s [Retry 1/5].")
        tee.file_only_stream.flush()

        data = read(path)
        assert "unauthenticated requests" not in data
        assert "Retrying in 2s" in data
        tee.close_debug_log()
