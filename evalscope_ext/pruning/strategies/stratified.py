"""Difficulty-stratified coverage pruning.

Bins items by *difficulty* (the fraction of reference models that got them
wrong), then allocates the budget across bins **proportionally to bin mass**, so
the pruned set preserves the full benchmark's difficulty distribution rather than
skewing easy or hard. Within each bin we pick by farthest-first coverage in the
behavioural-fingerprint space, so selection is principled, not random.

This is explicitly *not* a forbidden baseline: it is neither uniform random
(it is distribution-preserving + coverage-driven) nor top-k easiest/hardest (it
spans all bins).
"""
from __future__ import annotations

from typing import List

import numpy as np

from evalscope_ext.pruning.base import PruneContext, PruneItem, PruningStrategy, target_count
from evalscope_ext.pruning.registry import register_pruning_strategy
from evalscope_ext.pruning.strategies.diversity import build_features, kcenter_greedy

DEFAULT_BINS = 5


def difficulty_vector(items: List[PruneItem], ctx: PruneContext) -> np.ndarray:
    """Per-item difficulty in [0,1]; 0.5 when no signal is available."""
    sig = ctx.signals
    out = np.full(len(items), 0.5, dtype=float)
    if sig is None:
        return out
    for i, it in enumerate(items):
        d = sig.difficulty(it.id)
        if d is not None:
            out[i] = d
    return out


def _largest_remainder_alloc(weights: np.ndarray, k: int) -> np.ndarray:
    """Apportion ``k`` integer slots across bins by largest-remainder (Hamilton)."""
    raw = weights / weights.sum() * k
    base = np.floor(raw).astype(int)
    remainder = k - base.sum()
    if remainder > 0:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return base


@register_pruning_strategy('difficulty_stratified')
class DifficultyStratifiedPruning(PruningStrategy):
    def select(self, items: List[PruneItem], ratio: float, ctx: PruneContext) -> List[int]:
        n = len(items)
        k = target_count(n, ratio)
        n_bins = int(ctx.strategy_args.get('bins', DEFAULT_BINS))

        diff = difficulty_vector(items, ctx)
        feats = build_features(items, ctx)

        # Quantile bin edges keep bins balanced even with a skewed difficulty
        # distribution; fall back to fixed edges when there are too few uniques.
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
            local = kcenter_greedy(feats[b], take, ctx.rng)
            chosen.extend(int(b[i]) for i in local)

        # If rounding left us short/over, fix up by coverage over the whole set.
        chosen = _fixup(chosen, feats, k, ctx.rng)
        return sorted(chosen)


def _fixup(chosen: List[int], feats: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    chosen = list(dict.fromkeys(chosen))  # dedupe, keep order
    if len(chosen) == k:
        return chosen
    if len(chosen) > k:
        return chosen[:k]
    # Need more: add farthest points from the current selection.
    sel = set(chosen)
    if not chosen:
        return kcenter_greedy(feats, k, rng)
    dist = np.min(
        np.stack([np.linalg.norm(feats - feats[c], axis=1) for c in chosen], axis=0), axis=0
    )
    order = np.argsort(-dist)
    for idx in order:
        if len(chosen) >= k:
            break
        if int(idx) not in sel:
            chosen.append(int(idx))
            sel.add(int(idx))
    return chosen
