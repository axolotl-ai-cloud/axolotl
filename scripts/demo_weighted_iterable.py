#!/usr/bin/env python
"""Demo for WeightedPromptedIterableDataset.
Print empirical vs expected per-task counts for a single epoch.
"""
from __future__ import annotations
import argparse
from collections import Counter
try:
    from axolotl.data.weighted_prompted_dataset import WeightedPromptedIterableDataset  # type: ignore
except ModuleNotFoundError:
    import sys, os, importlib.util, types
    root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # Manually load submodule if namespace package resolution failed
    mod_path = os.path.join(src_path, 'axolotl', 'data', 'weighted_prompted_dataset.py')
    spec = importlib.util.spec_from_file_location('axolotl.data.weighted_prompted_dataset', mod_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    # ensure parent packages in sys.modules
    if 'axolotl' not in sys.modules:
        pkg = types.ModuleType('axolotl')
        pkg.__path__ = [os.path.join(src_path, 'axolotl')]  # type: ignore
        sys.modules['axolotl'] = pkg
    if 'axolotl.data' not in sys.modules:
        data_pkg = types.ModuleType('axolotl.data')
        data_pkg.__path__ = [os.path.join(src_path, 'axolotl', 'data')]  # type: ignore
        sys.modules['axolotl.data'] = data_pkg
    sys.modules['axolotl.data.weighted_prompted_dataset'] = module
    from axolotl.data.weighted_prompted_dataset import WeightedPromptedIterableDataset  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--epoch-size', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--quota', action='store_true', help='Enable enforce_quota deterministic per-task counts')
    args = ap.parse_args()

    ds = WeightedPromptedIterableDataset(
        path=args.data,
        epoch_size=args.epoch_size,
        temperature=args.temperature,
        seed=args.seed,
        enforce_quota=args.quota,
    )
    counts = Counter()
    for rec in ds:
        counts[rec.get('task')] += 1
    expected = ds.expected_task_counts()
    print('Epoch empirical counts:')
    for t in sorted(counts.keys()):
        print(f"  {t}: empirical={counts[t]} expected={expected.get(t):.2f} diff={counts[t]-expected.get(t,0):.2f}")

if __name__ == '__main__':
    main()
