"""evalscope_ext — a dataset-pruning extension for evalscope.

Pruning selects the smallest sample subset of a benchmark that still preserves the
"is this model good enough?" verdict, so a sales/deployment team can get a fast
go/no-go answer at a fraction of the eval cost.

The package is split into three layers:

* ``evalscope_ext.pruning`` — the framework-agnostic core (strategies, registry,
  offline signals). Depends only on the standard library + numpy, so it is unit
  testable without installing the full evalscope stack.
* ``evalscope_ext.integration`` — the thin glue that plugs the core into evalscope
  as ``*_pruned`` benchmark variants. Importing it triggers ``@register_benchmark``
  side effects, so ``evalscope eval --datasets live_code_bench_pruned`` works.
* ``evalscope_ext.tools`` — CLI utilities (``compare_runs``, offline validation).

Importing ``evalscope_ext`` registers the pruned benchmarks. The registration is
guarded so that ``import evalscope_ext`` still succeeds (for using the core or the
tools) even when the heavy evalscope dependencies are not installed.
"""

from evalscope_ext.pruning import (  # noqa: F401
    PruneContext,
    PruneItem,
    PruningStrategy,
    get_pruning_strategy,
    list_pruning_strategies,
    register_pruning_strategy,
)

__all__ = [
    'PruneContext',
    'PruneItem',
    'PruningStrategy',
    'register_pruning_strategy',
    'get_pruning_strategy',
    'list_pruning_strategies',
]


def _register_benchmarks() -> None:
    """Import the integration adapters to fire benchmark registration.

    Wrapped in a try/except so that the core and tools remain importable in
    environments where evalscope itself cannot be imported (e.g. a CI box that
    only exercises the framework-agnostic pruning logic).
    """
    try:
        from evalscope_ext import integration  # noqa: F401
    except Exception:  # pragma: no cover - depends on optional heavy deps
        # evalscope (or one of its deps) is unavailable; the pruned benchmarks
        # simply won't be registered, but the pruning core stays usable.
        pass


_register_benchmarks()
