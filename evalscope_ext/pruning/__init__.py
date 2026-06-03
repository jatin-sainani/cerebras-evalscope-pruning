"""Framework-agnostic pruning core: items, context, strategies, registry.

Nothing here imports evalscope, so it can be unit-tested standalone and reused by
the offline-validation tooling. The evalscope glue lives in
``evalscope_ext.integration``.
"""

from evalscope_ext.pruning.base import PruneContext, PruneItem, PruningStrategy
from evalscope_ext.pruning.registry import (
    get_pruning_strategy,
    list_pruning_strategies,
    register_pruning_strategy,
)

# Importing the strategies module triggers their @register_pruning_strategy
# decorators so they are discoverable by name.
from evalscope_ext.pruning import strategies  # noqa: F401,E402

__all__ = [
    'PruneItem',
    'PruneContext',
    'PruningStrategy',
    'register_pruning_strategy',
    'get_pruning_strategy',
    'list_pruning_strategies',
]
