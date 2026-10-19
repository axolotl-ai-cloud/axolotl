"""Fixtures shared by the e2e tests."""

import pytest

from axolotl.cli import preprocess as preprocess_cli
from axolotl.utils.data import lock, shared


@pytest.fixture(autouse=True)
def _isolated_default_prepared_path(monkeypatch, tmp_path):
    """Give each test its own fallback ``dataset_prepared_path``.

    Most e2e configs leave the key unset, so under xdist every worker would otherwise
    share ``./last_run_prepared`` and serialise on its single prepare lock.
    """
    path = str(tmp_path / "last_run_prepared")
    for module in (lock, shared, preprocess_cli):
        monkeypatch.setattr(module, "DEFAULT_DATASET_PREPARED_PATH", path)
