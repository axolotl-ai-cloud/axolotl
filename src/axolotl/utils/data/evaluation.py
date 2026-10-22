"""Named evaluation dataset preparation."""

from collections import defaultdict
from collections.abc import Callable
from typing import TypeAlias

from datasets import Dataset, DatasetDict

from axolotl.utils.dict import DictDefault

EvalDataset: TypeAlias = Dataset | dict[str, Dataset] | None


def load_eval_datasets(
    cfg: DictDefault, load: Callable[[DictDefault], Dataset]
) -> EvalDataset:
    """Load the aggregate and optionally independently prepared named groups."""
    aggregate = load(cfg)
    groups = defaultdict(list)
    for dataset in cfg.test_datasets or []:
        if name := dataset.get("eval_name"):
            groups[name].append(dataset)
    if not groups:
        return aggregate

    datasets = {"": aggregate}
    for name, configs in groups.items():
        group_cfg = DictDefault(dict(cfg))
        group_cfg.test_datasets = configs
        datasets[name] = load(group_cfg)
    return DatasetDict(datasets)
