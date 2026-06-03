"""The default ``balanced`` strategy: unbiased coverage + within-stratum discrimination.

A go/no-go needs two things that pull in opposite directions:

* an **unbiased estimate** of the model's true score (so "score >= bar" is right),
  which wants a subset whose difficulty distribution matches the full benchmark;
* **statistical power** to tell capable models from weak ones, which wants the
  *discriminating* items the reference panel splits.

Selecting purely by discrimination oversamples ~50/50 items and biases the score
estimate (the pruned mean drifts toward 0.5). So ``balanced`` uses a
**difficulty-stratified backbone** — allocate the budget across difficulty bins
in proportion to their mass, which keeps the estimate unbiased — and *within each
stratum* prefers the most discriminating items (then coverage-diversifies them).
You get an unbiased score and concentrated power at the same time.

For LLM-judged benchmarks it first drops the noisiest judge-variance quantile so
fragile verdicts can't dominate. With a deterministic grader that step is a no-op.
"""
from __future__ import annotations

from typing import List

import numpy as np

from evalscope_ext.pruning.base import PruneContext, PruneItem, PruningStrategy, target_count
from evalscope_ext.pruning.registry import register_pruning_strategy
from evalscope_ext.pruning.strategies.discrimination import discrimination_scores
from evalscope_ext.pruning.strategies.diversity import build_features, kcenter_greedy
from evalscope_ext.pruning.strategies.judge_noise import judge_variance_vector
from evalscope_ext.pruning.strategies.stratified import (
    DEFAULT_BINS,
    _fixup,
    _largest_remainder_alloc,
    difficulty_vector,
)

JUDGE_DROP_QUANTILE = 0.15  # drop noisiest 15% when judge variance is material
WITHIN_BIN_POOL = 2.0  # consider this many * take most-discriminating items per bin


@register_pruning_strategy('balanced')
class BalancedPruning(PruningStrategy):
    def select(self, items: List[PruneItem], ratio: float, ctx: PruneContext) -> List[int]:
        n = len(items)
        k = target_count(n, ratio)
        feats = build_features(items, ctx)
        mat = PruningStrategy._score_matrix(items, ctx)

        # No reference signal at all: fall back to pure coverage.
        if mat is None:
            return sorted(kcenter_greedy(feats, k, ctx.rng))

        info = discrimination_scores(mat)
        diff = difficulty_vector(items, ctx)
        alive = np.arange(n)

        # 1. Drop the noisiest items for non-deterministic graders.
        var = judge_variance_vector(items, ctx)
        if np.any(var > 0):
            thresh = np.quantile(var, 1.0 - JUDGE_DROP_QUANTILE)
            mask = var <= thresh
            if mask.sum() >= k:
                alive = alive[mask]

        # 2. Stratified backbone: allocate budget across difficulty bins by mass
        #    (this is what keeps the score estimate unbiased).
        n_bins = int(ctx.strategy_args.get('bins', DEFAULT_BINS))
        d = diff[alive]
        edges = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([d.min(), d.max() + 1e-9])
        bin_idx = np.clip(np.digitize(d, edges[1:-1]), 0, len(edges) - 2)
        bins = [alive[np.where(bin_idx == b)[0]] for b in range(len(edges) - 1)]
        sizes = np.array([len(b) for b in bins], dtype=float)
        alloc = _largest_remainder_alloc(sizes, min(k, alive.size))

        chosen: List[int] = []
        for b, take in zip(bins, alloc):
            if take <= 0 or len(b) == 0:
                continue
            take = min(take, len(b))
            # 3. Within the stratum, prefer discriminating items, then diversify.
            order = b[np.argsort(-info[b])]
            pool = order[: min(len(order), max(take, int(WITHIN_BIN_POOL * take)))]
            local = kcenter_greedy(feats[pool], take, ctx.rng, seed_order=np.argsort(-info[pool]))
            chosen.extend(int(pool[i]) for i in local)

        chosen = _fixup(chosen, feats, k, ctx.rng)
        return sorted(chosen)
