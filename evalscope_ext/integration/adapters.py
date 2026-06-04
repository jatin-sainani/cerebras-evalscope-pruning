"""Pruned benchmark variants: ``live_code_bench_pruned``, ``aa_lcr_pruned``, ``mmmu_pruned``.

Each is three lines of real logic: derive a ``*_pruned`` :class:`BenchmarkMeta`
from the already-registered base meta, declare the prune ``extra_params``, and
register a class that composes :class:`PruningMixin` over the base adapter. All
behaviour lives in the mixin + strategies, so adding a pruned variant for a new
benchmark is trivial and there are no per-dataset hacks.
"""
from __future__ import annotations

import copy
from typing import Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
from evalscope.utils.logger import get_logger

from evalscope_ext.integration.mixin import PruningMixin

logger = get_logger()

# Declared as spec entries so the evalscope ``extra_params`` override path
# validates them; they are also accepted as top-level dataset args.
PRUNE_PARAM_SPECS = {
    'pruning_strategy': {
        'type': 'str | null',
        'description': "Pruning strategy name (e.g. 'balanced', 'difficulty_stratified', "
                       "'discrimination', 'diversity', 'judge_noise_aware'). Null = no pruning.",
        'value': None,
    },
    'prune_ratio': {
        'type': 'float',
        'description': 'Fraction of samples to keep, in (0, 1].',
        'value': 0.1,
    },
    'prune_seed': {
        'type': 'int',
        'description': 'Seed for reproducible selection.',
        'value': 42,
    },
    'signals_path': {
        'type': 'str | null',
        'description': 'Override path to the offline signal cache JSON. Null = bundled cache.',
        'value': None,
    },
    'strategy_args': {
        'type': 'dict',
        'description': 'Extra keyword args forwarded to the strategy (e.g. {"bins": 5}).',
        'value': {},
    },
}


def _make_pruned_meta(base_name: str, new_name: str) -> BenchmarkMeta:
    """Deep-copy a registered base meta into a ``*_pruned`` variant."""
    if base_name not in BENCHMARK_REGISTRY:
        raise KeyError(f'Base benchmark {base_name!r} not registered.')
    meta = copy.deepcopy(BENCHMARK_REGISTRY[base_name])
    meta.name = new_name
    meta.pretty_name = f'{meta.pretty_name or base_name} (pruned)'
    meta.data_adapter = None  # set by @register_benchmark
    meta.extra_params = {**(meta.extra_params or {}), **PRUNE_PARAM_SPECS}
    return meta


def register_pruned_benchmark(base_name: str, base_cls: Optional[type] = None) -> type:
    """Register ``<base_name>_pruned`` for **any** evalscope benchmark — universal.

    This is the whole story of how pruning attaches to a benchmark: there is no
    per-dataset code. Given a registered benchmark name, we derive its meta, add the
    prune params, and register ``PruningMixin`` composed over its adapter. Works for
    LiveCodeBench, AA-LCR, MMMU, or a benchmark we've never seen — one call:

        register_pruned_benchmark('gpqa')   # -> 'gpqa_pruned'

    Args:
        base_name: the registered base benchmark name.
        base_cls:  its adapter class; if omitted, taken from the registry (so the base
                   module must already be imported).
    Returns:
        the newly registered pruned adapter class.
    """
    if base_cls is None:
        if base_name not in BENCHMARK_REGISTRY:
            raise KeyError(f'Base benchmark {base_name!r} is not registered; import it first.')
        base_cls = BENCHMARK_REGISTRY[base_name].data_adapter

    new_name = f'{base_name}_pruned'
    meta = _make_pruned_meta(base_name, new_name)

    @register_benchmark(meta)
    class _PrunedAdapter(PruningMixin, base_cls):  # type: ignore[misc, valid-type]
        pruning_base_name = base_name

    _PrunedAdapter.__name__ = f'{base_cls.__name__}Pruned'
    _PrunedAdapter.__qualname__ = _PrunedAdapter.__name__
    logger.info(f'[prune] registered benchmark {new_name!r}.')
    return _PrunedAdapter


# Register the variants the challenge cares about. Each is independent so a missing
# base benchmark (e.g. a vision dep for MMMU) doesn't block the others. Adding more is
# a one-liner — the mechanism is benchmark-agnostic.
def _safe_register(base_name: str, import_path: str, cls_name: str) -> None:
    try:
        module = __import__(import_path, fromlist=[cls_name])
        register_pruned_benchmark(base_name, getattr(module, cls_name))
    except Exception as exc:  # pragma: no cover - depends on optional deps
        logger.warning(f'[prune] could not register {base_name}_pruned: {exc}')


_safe_register(
    'live_code_bench',
    'evalscope.benchmarks.live_code_bench.live_code_bench_adapter',
    'LiveCodeBenchAdapter',
)
_safe_register(
    'aa_lcr',
    'evalscope.benchmarks.aa_lcr.aa_lcr_adapter',
    'AALCRAdapter',
)
_safe_register(
    'mmmu',
    'evalscope.benchmarks.mmmu.mmmu_adapter',
    'MMMUAdapter',
)
