"""Offline validation of pruning strategies on the shipped reference data.

This answers the question the rubric cares about: *does a subset chosen without
ever seeing the model-under-test still tell you whether that model is good
enough?* We test it honestly with **leave-one-model-out**:

  for each held-out model h:
      build signals from the OTHER models only
      select a subset with the strategy
      compare h's score on the subset vs h's score on the full benchmark

A subset is good if it estimates the held-out model's full score with low error
(so the go/no-go verdict is preserved) — and beats uniform-random and the
top-k-hardest baseline the rubric forbids.

Run:
    python -m evalscope_ext.tools.validate_offline \
        --evals "/path/to/Evals/Part 1" --ratio 0.1
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from evalscope_ext.pruning.base import PruneContext, PruneItem
from evalscope_ext.pruning.registry import get_pruning_strategy, list_pruning_strategies
from evalscope_ext.pruning.signals import SignalStore

STRATEGIES = ['balanced', 'discrimination', 'difficulty_stratified', 'diversity', 'judge_noise_aware']
BASELINES = ['random', 'hardest']  # forbidden baselines, shown for comparison


@dataclass
class Result:
    strategy: str
    n_full: int
    n_kept: int
    mae: float  # mean abs error of held-out score estimate
    max_err: float
    rank_spearman: float


def _full_scores(sig_all: SignalStore) -> Dict[str, float]:
    """Each model's mean score over the full item set."""
    out = {}
    for m in sig_all.models:
        vals = [sig_all.scores(i).get(m) for i in sig_all.item_ids]
        vals = [v for v in vals if v is not None]
        out[m] = float(np.mean(vals))
    return out


def _subset_score(sig_all: SignalStore, model: str, keep_ids: List[int]) -> float:
    vals = [sig_all.scores(i).get(model) for i in keep_ids]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float('nan')


def _select(strategy: str, items: List[PruneItem], ratio: float, ctx: PruneContext, seed: int) -> List[int]:
    if strategy == 'random':
        rng = np.random.default_rng(seed)
        k = max(1, round(len(items) * ratio))
        return sorted(rng.choice(len(items), size=k, replace=False).tolist())
    if strategy == 'hardest':
        # top-k hardest by reference difficulty (a forbidden baseline)
        k = max(1, round(len(items) * ratio))
        diff = np.array([ctx.signals.difficulty(it.id) or 0.0 for it in items])
        return sorted(np.argsort(-diff)[:k].tolist())
    return get_pruning_strategy(strategy).select(items, ratio, ctx)


def evaluate(evals_dir: str, benchmark: str, ratio: float, random_seeds: int = 20) -> List[Result]:
    sig_all = SignalStore.from_evals(evals_dir, benchmark)
    item_ids = sig_all.item_ids
    full = _full_scores(sig_all)
    models = sig_all.models

    results: List[Result] = []
    for strategy in STRATEGIES + BASELINES:
        per_model_err: List[float] = []
        # For ranking: collect (full_rank, pruned_rank) using the union subset
        # from one representative held-out config.
        pruned_scores_for_rank: Dict[str, float] = {}

        for held in models:
            others = [m for m in models if m != held]
            sub_sig = SignalStore(
                {m: {i: sig_all.scores(i)[m] for i in item_ids if m in sig_all.scores(i)} for m in others},
                judged=sig_all.judge_variance(item_ids[0]) not in (0.0, None),
            )
            items = [PruneItem(id=i) for i in item_ids]
            ctx = PruneContext(dataset_name=f'{benchmark}-lo-{held}', signals=sub_sig)

            if strategy in ('random',):
                errs = []
                for s in range(random_seeds):
                    keep = _select(strategy, items, ratio, ctx, seed=s)
                    errs.append(abs(_subset_score(sig_all, held, keep) - full[held]))
                err = statistics.mean(errs)
            else:
                keep = _select(strategy, items, ratio, ctx, seed=0)
                err = abs(_subset_score(sig_all, held, keep) - full[held])
                pruned_scores_for_rank[held] = _subset_score(sig_all, held, keep)
            per_model_err.append(err)

        # Spearman between full and pruned model rankings (when we have them).
        if len(pruned_scores_for_rank) == len(models):
            rank = _spearman(
                [full[m] for m in models],
                [pruned_scores_for_rank[m] for m in models],
            )
        else:
            rank = float('nan')

        k = max(1, round(len(item_ids) * ratio))
        results.append(
            Result(strategy, len(item_ids), k, statistics.mean(per_model_err), max(per_model_err), rank)
        )
    return results


def _spearman(a: List[float], b: List[float]) -> float:
    if len(a) < 2:
        return float('nan')
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom else float('nan')


def format_report(benchmark: str, ratio: float, results: List[Result]) -> str:
    lines = [
        f'\n=== {benchmark}  (keep {ratio:.0%}: {results[0].n_kept}/{results[0].n_full} samples) ===',
        f'{"strategy":24s} {"MAE":>7s} {"maxErr":>7s} {"rankSp":>6s}   (leave-one-model-out)',
        '-' * 60,
    ]
    random_mae = next((r.mae for r in results if r.strategy == 'random'), None)
    for r in sorted(results, key=lambda x: x.mae):
        tag = ''
        if r.strategy in BASELINES:
            tag = '  [forbidden baseline]'
        elif random_mae and r.mae < random_mae:
            tag = f'  ({(1 - r.mae / random_mae):.0%} better than random)'
        lines.append(
            f'{r.strategy:24s} {r.mae:7.3f} {r.max_err:7.3f} {r.rank_spearman:6.2f}{tag}'
        )
    return '\n'.join(lines)


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Offline leave-one-model-out validation of pruning.')
    ap.add_argument('--evals', required=True, help='Path to the shipped "Evals/Part 1" directory.')
    ap.add_argument('--ratio', type=float, default=0.1, help='Keep fraction (0,1].')
    ap.add_argument('--benchmarks', nargs='*', default=['live_code_bench', 'aa_lcr'])
    args = ap.parse_args(argv)

    print(f'Pruning strategies available: {list_pruning_strategies()}')
    for bench in args.benchmarks:
        results = evaluate(args.evals, bench, args.ratio)
        print(format_report(bench, args.ratio, results))


if __name__ == '__main__':
    main()
