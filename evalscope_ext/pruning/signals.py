"""Offline per-item signals built from already-scored *reference* models.

The shipped challenge data gives, for a panel of reference models, a per-sample
score (``pass`` for the LiveCodeBench sandbox, ``acc`` for the AA-LCR LLM judge),
joined to the sample by ``index``. From that panel we derive, per item:

* the score each reference model got (the discrimination signal),
* difficulty = fraction of the panel that got it wrong,
* a judge-variance proxy for LLM-judged benchmarks (verdict fragility).

Crucially these are statistics about the *items*, computed from models other than
the one under test, so a subset chosen from them does not overfit to any single
model. ``SignalStore`` caches to a compact JSON (a few hundred KB) so the
evalscope runtime can prune without the multi-GB raw prediction files.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# Map an evalscope benchmark name -> how its shipped reference files look.
BENCHMARK_SPECS = {
    'live_code_bench': {
        'review_prefix': 'live_code_bench_v5__',
        'judged': False,  # deterministic sandbox grader
    },
    'aa_lcr': {
        'review_prefix': 'aa_lcr__',
        'judged': True,  # non-deterministic LLM judge
    },
}


def _read_scores(path: str) -> Dict[int, float]:
    """Read an evalscope/challenge review .jsonl -> {index: score in [0,1]}."""
    out: Dict[int, float] = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj['index']
            value = obj['sample_score']['score']['value']
            # value is e.g. {"pass": 1.0} or {"acc": 0.0}; take the first number.
            score = next(float(v) for v in value.values() if isinstance(v, (int, float)))
            out[idx] = score
    return out


class SignalStore:
    """Implements the :class:`~evalscope_ext.pruning.base.Signals` protocol."""

    def __init__(
        self,
        scores_by_model: Dict[str, Dict[int, float]],
        judged: bool = False,
        meta: Optional[Dict[int, dict]] = None,
    ) -> None:
        self.models: List[str] = sorted(scores_by_model)
        self._scores = scores_by_model
        self._judged = judged
        self._meta = meta or {}
        # Precompute the per-item panel view.
        self._items = sorted({i for s in scores_by_model.values() for i in s})
        self._panel: Dict[int, Dict[str, float]] = {}
        for i in self._items:
            self._panel[i] = {m: self._scores[m][i] for m in self.models if i in self._scores[m]}

    # ---- Signals protocol ----------------------------------------------

    def has(self, item_id: int) -> bool:
        return item_id in self._panel and len(self._panel[item_id]) > 0

    def scores(self, item_id: int) -> Dict[str, float]:
        return dict(self._panel.get(item_id, {}))

    def difficulty(self, item_id: int) -> Optional[float]:
        sc = self._panel.get(item_id)
        if not sc:
            return None
        return 1.0 - sum(sc.values()) / len(sc)

    def judge_variance(self, item_id: int) -> Optional[float]:
        if not self._judged:
            return 0.0
        sc = self._panel.get(item_id)
        if not sc:
            return None
        # Verdict fragility proxy: binomial variance of the panel pass-rate.
        # Items the panel splits ~50/50 are the most likely to flip under a
        # resampled LLM judge. (A true estimate would need repeated judgments;
        # this is the best single-pass proxy — see Handout A.)
        p = sum(sc.values()) / len(sc)
        return p * (1.0 - p)

    # ---- construction / caching ----------------------------------------

    @property
    def item_ids(self) -> List[int]:
        return list(self._items)

    @classmethod
    def from_evals(cls, evals_part1_dir: str, benchmark: str) -> 'SignalStore':
        """Build from the shipped ``Evals/Part 1`` directory for a benchmark."""
        spec = BENCHMARK_SPECS[benchmark]
        review_dir = os.path.join(evals_part1_dir, 'reviews')
        scores_by_model: Dict[str, Dict[int, float]] = {}
        for fname in os.listdir(review_dir):
            if not fname.startswith(spec['review_prefix']) or not fname.endswith('.jsonl'):
                continue
            model = fname[len(spec['review_prefix']):-len('.jsonl')]
            scores_by_model[model] = _read_scores(os.path.join(review_dir, fname))
        if not scores_by_model:
            raise FileNotFoundError(
                f'No reviews for {benchmark} (prefix {spec["review_prefix"]!r}) under {review_dir}'
            )
        return cls(scores_by_model, judged=spec['judged'])

    def to_cache(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'judged': self._judged,
            'models': self.models,
            # store as {index: {model: score}}
            'panel': {str(i): self._panel[i] for i in self._items},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)

    @classmethod
    def from_cache(cls, path: str) -> 'SignalStore':
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)
        scores_by_model: Dict[str, Dict[int, float]] = {m: {} for m in payload['models']}
        for idx_str, per_model in payload['panel'].items():
            idx = int(idx_str)
            for m, s in per_model.items():
                scores_by_model[m][idx] = float(s)
        return cls(scores_by_model, judged=bool(payload.get('judged', False)))
