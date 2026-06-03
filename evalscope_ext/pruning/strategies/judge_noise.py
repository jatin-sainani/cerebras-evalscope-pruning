"""Judge-noise-aware pruning, for LLM-judged benchmarks like AA-LCR.

AA-LCR is graded by a non-deterministic LLM judge, so a single recorded score is
a noisy draw. If we prune toward items whose verdict is fragile under judge
resampling, the pruned score becomes mostly judge noise. This strategy keeps a
difficulty-representative subset while **preferring verdict-stable items**
(low estimated judge variance) within each difficulty band.

When the grader is deterministic (e.g. the LiveCodeBench sandbox) every item's
judge variance is 0 and this degrades gracefully to difficulty-stratified
coverage.
"""
from __future__ import annotations

from typing import List

import numpy as np

from evalscope_ext.pruning.base import PruneContext, PruneItem, PruningStrategy, target_count
from evalscope_ext.pruning.registry import register_pruning_strategy
from evalscope_ext.pruning.strategies.diversity import build_features, kcenter_greedy
from evalscope_ext.pruning.strategies.stratified import (
    DEFAULT_BINS,
    _fixup,
    _largest_remainder_alloc,
    difficulty_vector,
)


def judge_variance_vector(items: List[PruneItem], ctx: PruneContext) -> np.ndarray:
    """Per-item judge-variance estimate; 0 when the grader is deterministic."""
    sig = ctx.signals
    out = np.zeros(len(items), dtype=float)
    if sig is None:
        return out
    for i, it in enumerate(items):
        v = sig.judge_variance(it.id)
        if v is not None:
            out[i] = v
    return out


@register_pruning_strategy('judge_noise_aware')
class JudgeNoiseAwarePruning(PruningStrategy):
    def select(self, items: List[PruneItem], ratio: float, ctx: PruneContext) -> List[int]:
        n = len(items)
        k = target_count(n, ratio)
        n_bins = int(ctx.strategy_args.get('bins', DEFAULT_BINS))

        diff = difficulty_vector(items, ctx)
        var = judge_variance_vector(items, ctx)
        feats = build_features(items, ctx)

        edges = np.unique(np.quantile(diff, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([diff.min(), diff.max() + 1e-9])
        bin_idx = np.clip(np.digitize(diff, edges[1:-1]), 0, len(edges) - 2)
        bins = [np.where(bin_idx == b)[0] for b in range(len(edges) - 1)]
        sizes = np.array([len(b) for b in bins], dtype=float)
        alloc = _largest_remainder_alloc(sizes, k)

        chosen: List[int] = []
        for b, take in zip(bins, alloc):
            if take <= 0 or len(b) == 0:
                continue
            take = min(take, len(b))
            # Prefer the lowest-variance items in this band, then keep them
            # coverage-diverse so we still span the band's content.
            order = b[np.argsort(var[b])]
            pool = order[: min(len(order), max(take, 2 * take))]
            local = kcenter_greedy(feats[pool], take, ctx.rng)
            chosen.extend(int(pool[i]) for i in local)

        chosen = _fixup(chosen, feats, k, ctx.rng)
        return sorted(chosen)
