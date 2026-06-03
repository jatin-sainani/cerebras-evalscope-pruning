"""Embedding/feature-diversity pruning via farthest-first (k-center) greedy.

Selects a subset that maximally *covers* the item feature space, so no region of
the benchmark (a topic cluster, a difficulty band, a failure mode) is dropped
wholesale. This is representativeness-driven and never looks at the
model-under-test, so it generalises to unseen models.

Feature precedence per item: explicit ``embedding`` -> reference score vector
("which models fail here") -> scalar difficulty. Falling back to the score
vector means that even without text embeddings we cluster by *behavioural
fingerprint*, which is a strong, cheap signal.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from evalscope_ext.pruning.base import PruneContext, PruneItem, PruningStrategy, target_count
from evalscope_ext.pruning.registry import register_pruning_strategy


def build_features(items: List[PruneItem], ctx: PruneContext) -> np.ndarray:
    """Build an (n, d) feature matrix, standardised per column."""
    if items and items[0].embedding is not None and all(it.embedding is not None for it in items):
        feats = np.vstack([np.asarray(it.embedding, dtype=float) for it in items])
    else:
        mat = PruningStrategy._score_matrix(items, ctx)
        if mat is not None:
            feats = mat
        else:
            # Last resort: difficulty scalar (or zeros). Coverage degenerates to
            # spreading across the difficulty axis.
            sig = ctx.signals
            diff = np.array(
                [
                    (sig.difficulty(it.id) if sig and sig.difficulty(it.id) is not None else 0.5)
                    for it in items
                ],
                dtype=float,
            ).reshape(-1, 1)
            feats = diff
    # Standardise so no single dimension dominates the distance metric.
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (feats - mu) / sd


def kcenter_greedy(feats: np.ndarray, k: int, rng: np.random.Generator, seed_order: Optional[np.ndarray] = None) -> List[int]:
    """Farthest-first traversal: pick ``k`` points that maximise min-distance coverage.

    Args:
        feats: (n, d) feature matrix.
        k: number of points to select.
        rng: RNG used only to break ties / choose the first centre.
        seed_order: optional priority over the first centre and tie-breaks.
    """
    n = feats.shape[0]
    k = max(1, min(k, n))
    # Start from the medoid-ish point (closest to the centroid) for determinism,
    # nudged by seed_order if provided.
    centroid = feats.mean(axis=0, keepdims=True)
    d0 = np.linalg.norm(feats - centroid, axis=1)
    first = int(np.argmin(d0)) if seed_order is None else int(seed_order[0])
    selected = [first]
    dist = np.linalg.norm(feats - feats[first], axis=1)
    while len(selected) < k:
        # Next centre = farthest from the current selection.
        cand = int(np.argmax(dist))
        if dist[cand] == 0:  # all remaining points coincide with selected ones
            remaining = [i for i in range(n) if i not in set(selected)]
            if not remaining:
                break
            cand = int(rng.choice(remaining))
        selected.append(cand)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[cand], axis=1))
    return selected


@register_pruning_strategy('diversity')
class DiversityPruning(PruningStrategy):
    """k-center coverage over item features."""

    def select(self, items: List[PruneItem], ratio: float, ctx: PruneContext) -> List[int]:
        k = target_count(len(items), ratio)
        feats = build_features(items, ctx)
        return sorted(kcenter_greedy(feats, k, ctx.rng))
