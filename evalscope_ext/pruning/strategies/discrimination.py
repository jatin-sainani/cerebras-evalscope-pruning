"""Discrimination-based pruning (IRT-flavoured item selection).

Most of a saturated benchmark is dead weight for a go/no-go decision: items that
*every* reference model passes (or fails) tell you nothing about whether a new
model is good enough — they only inflate cost. This strategy keeps the items that
best **separate strong from weak models**, which is precisely the signal that
transfers to an unseen fourth model.

Per item we combine two classic psychometric quantities computed from the
reference score matrix S (n_items x n_models):

* **Fisher information** ``p(1-p)`` where ``p`` is the item's pass-rate across
  models — maximised for items the panel splits ~50/50.
* **Point-biserial discrimination** ``corr(S_i, ability)`` where ``ability`` is
  each model's mean score — positive when stronger models do better on the item,
  i.e. the item is aligned with overall capability rather than noise.

We rank by the product and then run farthest-first over the high-information pool
so the kept items are both discriminating and non-redundant.
"""
from __future__ import annotations

from typing import List

import numpy as np

from evalscope_ext.pruning.base import PruneContext, PruneItem, PruningStrategy, target_count
from evalscope_ext.pruning.registry import register_pruning_strategy
from evalscope_ext.pruning.strategies.diversity import build_features, kcenter_greedy


def discrimination_scores(mat: np.ndarray) -> np.ndarray:
    """Per-item discrimination in [0, ~1] from an (n_items, n_models) matrix."""
    p = mat.mean(axis=1)
    fisher = p * (1.0 - p)  # 0 for unanimous items, max at p=0.5

    ability = mat.mean(axis=0)  # per-model mean score
    ability_c = ability - ability.mean()
    if np.allclose(ability_c, 0):
        point_biserial = np.zeros(mat.shape[0])
    else:
        item_c = mat - mat.mean(axis=1, keepdims=True)
        num = item_c @ ability_c
        den = np.linalg.norm(item_c, axis=1) * np.linalg.norm(ability_c) + 1e-12
        point_biserial = num / den
    # Keep alignment as a non-negative multiplier; Fisher info dominates.
    return fisher * (0.5 + 0.5 * np.clip(point_biserial, -1, 1))


@register_pruning_strategy('discrimination')
class DiscriminationPruning(PruningStrategy):
    def select(self, items: List[PruneItem], ratio: float, ctx: PruneContext) -> List[int]:
        n = len(items)
        k = target_count(n, ratio)
        mat = PruningStrategy._score_matrix(items, ctx)
        if mat is None:
            # No reference signal -> fall back to coverage so we never crash.
            feats = build_features(items, ctx)
            return sorted(kcenter_greedy(feats, k, ctx.rng))

        info = discrimination_scores(mat)
        # Candidate pool: the most informative items (2x budget, capped at n),
        # then pick a coverage-diverse subset within it so we don't select many
        # near-identical "splitter" items.
        pool_size = min(n, max(k, int(2 * k)))
        pool = np.argsort(-info)[:pool_size]
        feats = build_features(items, ctx)[pool]
        # Bias the first centre toward the single most informative item.
        order = np.argsort(-info[pool])
        local = kcenter_greedy(feats, k, ctx.rng, seed_order=order)
        return sorted(int(pool[i]) for i in local)
