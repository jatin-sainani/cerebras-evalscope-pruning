"""PruningMixin — applies a pruning strategy on the loaded dataset.

Hooks the single cleanest extension point: it overrides ``load_dataset`` to call
the base adapter's loader, then replaces each subset with a pruned
``MemoryDataset``. Pruning happens on the *whole* loaded subset (selection is a
global decision over coverage/discrimination), which is why a per-sample
``sample_filter`` is the wrong layer.

Prune configuration is read from the dataset args in two accepted forms:

    # the literal challenge run-contract form (top-level keys)
    --dataset-args '{"pruning_strategy": "balanced", "prune_ratio": 0.1}'

    # the evalscope-idiomatic extra_params form (also validated)
    --dataset-args '{"extra_params": {"pruning_strategy": "balanced", "prune_ratio": 0.1}}'

Both work: top-level keys are read straight off ``task_config.dataset_args`` and
take precedence over ``extra_params`` defaults declared on the pruned meta.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning.base import PruneContext, PruneItem
from evalscope_ext.pruning.registry import get_pruning_strategy
from evalscope_ext.pruning.signals import SignalStore

logger = get_logger()

SIGNALS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'signals')


@dataclass
class PruneConfig:
    strategy: Optional[str]
    ratio: float
    seed: int
    strategy_args: Dict[str, Any]
    signals_path: Optional[str]


class PruningMixin:
    """Mix in before a ``DataAdapter`` subclass to add ``*_pruned`` behaviour."""

    #: Base benchmark name whose signal cache to use (defaults to name minus '_pruned').
    pruning_base_name: Optional[str] = None

    def _base_name(self) -> str:
        if self.pruning_base_name:
            return self.pruning_base_name
        name = self.name  # type: ignore[attr-defined]
        return name[:-len('_pruned')] if name.endswith('_pruned') else name

    def _prune_config(self) -> PruneConfig:
        # Top-level dataset args (the literal run-contract form).
        top: Dict[str, Any] = {}
        tc = getattr(self, '_task_config', None)
        if tc is not None and getattr(tc, 'dataset_args', None):
            top = tc.dataset_args.get(self.name, {}) or {}  # type: ignore[attr-defined]
        ep = self.extra_params or {}  # type: ignore[attr-defined]

        def pick(key, default=None):
            if key in top:
                return top[key]
            return ep.get(key, default)

        seed_default = getattr(self, 'seed', None) or 42
        return PruneConfig(
            strategy=pick('pruning_strategy'),
            ratio=float(pick('prune_ratio', 0.1)),
            seed=int(pick('prune_seed', seed_default)),
            strategy_args=pick('strategy_args', {}) or {},
            signals_path=pick('signals_path'),
        )

    def _load_signals(self, cfg: PruneConfig) -> Optional[SignalStore]:
        path = cfg.signals_path or os.path.join(SIGNALS_DIR, f'{self._base_name()}.json')
        path = os.path.abspath(path)
        if not os.path.exists(path):
            logger.warning(
                f'[prune] no signal cache at {path}; pruning will fall back to '
                f'coverage over item structure only. Build one with '
                f'`python -m evalscope_ext.tools.build_signals`.'
            )
            return None
        return SignalStore.from_cache(path)

    def load_dataset(self) -> DatasetDict:  # type: ignore[override]
        dataset_dict: DatasetDict = super().load_dataset()  # type: ignore[misc]
        cfg = self._prune_config()
        if not cfg.strategy:
            logger.info('[prune] no pruning_strategy given; returning full dataset.')
            return dataset_dict

        strategy = get_pruning_strategy(cfg.strategy)
        signals = self._load_signals(cfg)
        base_name = self._base_name()

        for subset in list(dataset_dict.keys()):
            ds = dataset_dict[subset]
            n = len(ds)
            if n == 0:
                continue
            # Ensure stable 0..n-1 ids that align with the offline signal index
            # (both derive from the upstream dataset's canonical order).
            ds.reindex()
            samples = list(ds)
            items = [PruneItem(id=int(s.id), metadata=dict(s.metadata or {})) for s in samples]

            if signals is not None:
                covered = sum(1 for it in items if signals.has(it.id))
                if covered < 0.5 * len(items):
                    logger.warning(
                        f'[prune] signal cache covers only {covered}/{len(items)} samples of '
                        f'subset {subset!r}; selection quality may degrade.'
                    )

            ctx = PruneContext(
                dataset_name=base_name,
                metric_name=(self.metric_list[0] if self.metric_list else 'acc'),  # type: ignore[attr-defined]
                seed=cfg.seed,
                signals=signals,
                strategy_args=cfg.strategy_args,
            )
            keep_idx = strategy.select(items, cfg.ratio, ctx)
            kept = _select_samples(samples, keep_idx)
            pruned = MemoryDataset(samples=kept, name=getattr(ds, 'name', subset))
            pruned.reindex()
            dataset_dict[subset] = pruned
            logger.info(
                f"[prune] {self.name}/{subset}: {cfg.strategy} kept {len(kept)}/{n} "
                f"({len(kept)/n:.0%}) samples."
            )
        return dataset_dict


def _select_samples(samples: List[Any], keep_idx: List[int]) -> List[Any]:
    keep = sorted(set(int(i) for i in keep_idx if 0 <= int(i) < len(samples)))
    return [samples[i] for i in keep]
