"""Core abstractions for benchmark pruning.

A :class:`PruningStrategy` selects which samples to keep. It is given a list of
:class:`PruneItem` (a lightweight, evalscope-agnostic view of a sample) plus a
:class:`PruneContext` carrying the target ratio, an RNG, and an optional
:class:`Signals` provider with offline per-item statistics computed from already
scored reference models.

The key design choice: strategies consume **offline signals about the items**
(difficulty, cross-model discrimination, embeddings) and never the
model-under-test. A subset chosen to be informative about a *panel* of reference
models generalises to a new, unseen model — which is exactly the
"defensible for a fourth model" requirement.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


def _stable_seed(*parts: Any) -> int:
    """A process-independent seed (Python's str hash is randomized per run)."""
    digest = hashlib.md5('-'.join(map(str, parts)).encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


@dataclass
class PruneItem:
    """A minimal, framework-agnostic view of one benchmark sample.

    ``id`` must match the join key used by the offline :class:`Signals`
    (evalscope's ``sample.id`` after ``reindex()``, which equals the per-sample
    ``index`` in the shipped predictions/reviews).
    """

    id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@runtime_checkable
class Signals(Protocol):
    """Offline per-item statistics derived from reference-model eval results."""

    models: Sequence[str]

    def has(self, item_id: int) -> bool:
        """Whether any reference model produced a score for this item."""
        ...

    def scores(self, item_id: int) -> Dict[str, float]:
        """Map of reference-model name -> score (1=correct/pass, 0=incorrect)."""
        ...

    def difficulty(self, item_id: int) -> Optional[float]:
        """Fraction of reference models that got the item wrong (0..1), or None."""
        ...

    def judge_variance(self, item_id: int) -> Optional[float]:
        """Estimated grader/judge variance for this item (higher = noisier)."""
        ...


@dataclass
class PruneContext:
    """Everything a strategy needs beyond the items themselves."""

    dataset_name: str = 'unknown'
    metric_name: str = 'acc'
    seed: int = 42
    signals: Optional[Signals] = None
    strategy_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def rng(self) -> np.random.Generator:
        # Deterministic per (seed, dataset) so runs are reproducible across
        # processes (Python's built-in str hash is salted per interpreter).
        return np.random.default_rng(_stable_seed(self.seed, self.dataset_name))


def target_count(n: int, ratio: float) -> int:
    """How many samples to keep for ``n`` items at ``ratio`` in (0, 1]."""
    if not 0 < ratio <= 1:
        raise ValueError(f'prune ratio must be in (0, 1], got {ratio}')
    return max(1, min(n, round(n * ratio)))


class PruningStrategy(ABC):
    """Base class for pruning strategies.

    Subclasses register themselves with ``@register_pruning_strategy('name')`` and
    implement :meth:`select`, returning the indices (into ``items``) to keep.
    """

    #: Stable name used in ``--dataset-args '{"pruning_strategy": "..."}'``.
    name: str = 'base'

    @abstractmethod
    def select(self, items: List[PruneItem], ratio: float, ctx: PruneContext) -> List[int]:
        """Return the indices of the items to KEEP (length ≈ ratio * len(items))."""
        raise NotImplementedError

    # ---- shared helpers -------------------------------------------------

    @staticmethod
    def _score_matrix(items: List[PruneItem], ctx: PruneContext) -> Optional[np.ndarray]:
        """Build an (n_items, n_models) matrix of reference scores, or None.

        Missing (item, model) entries are filled with the column mean so the
        matrix is dense; items with no signal at all get the global mean.
        """
        sig = ctx.signals
        if sig is None or not getattr(sig, 'models', None):
            return None
        models = list(sig.models)
        n, m = len(items), len(models)
        mat = np.full((n, m), np.nan, dtype=float)
        for i, it in enumerate(items):
            sc = sig.scores(it.id) if sig.has(it.id) else {}
            for j, mod in enumerate(models):
                if mod in sc:
                    mat[i, j] = sc[mod]
        if np.all(np.isnan(mat)):
            return None
        # Impute NaNs with per-column mean (then global mean for empty columns).
        col_mean = np.nanmean(mat, axis=0)
        global_mean = float(np.nanmean(mat))
        col_mean = np.where(np.isnan(col_mean), global_mean, col_mean)
        idx = np.where(np.isnan(mat))
        mat[idx] = np.take(col_mean, idx[1])
        return mat
