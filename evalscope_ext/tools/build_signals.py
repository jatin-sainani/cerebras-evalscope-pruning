"""Precompute and cache the offline pruning signals from the shipped reference data.

The raw predictions are multi-GB and the challenge data is not redistributed, so
we ship a compact per-benchmark signal cache (a few hundred KB of synthetic-data
statistics) under ``evalscope_ext/data/signals/``. The evalscope runtime loads
*that* to prune — it never needs the raw Evals tree.

Run:
    python -m evalscope_ext.tools.build_signals --evals "/path/to/Evals/Part 1"
"""
from __future__ import annotations

import argparse
import os
from typing import List

from evalscope_ext.pruning.signals import BENCHMARK_SPECS, SignalStore

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'signals')


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Build cached pruning signal files.')
    ap.add_argument('--evals', required=True, help='Path to the shipped "Evals/Part 1" directory.')
    ap.add_argument('--out', default=DEFAULT_OUT, help='Output directory for signal caches.')
    ap.add_argument('--benchmarks', nargs='*', default=list(BENCHMARK_SPECS))
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    for bench in args.benchmarks:
        store = SignalStore.from_evals(args.evals, bench)
        path = os.path.join(out_dir, f'{bench}.json')
        store.to_cache(path)
        diff = [store.difficulty(i) for i in store.item_ids]
        n_disc = sum(1 for i in store.item_ids if 0 < (store.difficulty(i) or 0) < 1)
        print(
            f'{bench}: {len(store.item_ids)} items, models={store.models}, '
            f'{n_disc} discriminating -> {path}'
        )


if __name__ == '__main__':
    main()
