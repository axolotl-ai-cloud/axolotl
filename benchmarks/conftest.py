"""print results of selected benchmarks"""

from pytest_harvest import get_session_results_df
from tabulate import tabulate  # type: ignore


def pytest_sessionfinish(session):
    out = get_session_results_df(session)
    out.drop(
        [
            "status",
            "duration_ms",
            "cfg",
            "model_cfg",
            "attn_cfg",
            "dtype_cfg",
            "adapter_cfg",
            "pytest_obj",
            "vram_baseline",
            "vram_last",
            "train_result",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    print("")
    print(tabulate(out, headers="keys"))
