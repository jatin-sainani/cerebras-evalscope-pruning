"""evalscope glue: pruned benchmark variants.

Importing this package registers ``live_code_bench_pruned``, ``aa_lcr_pruned`` and
``mmmu_pruned`` with evalscope's benchmark registry (via ``@register_benchmark``),
so they become selectable through ``evalscope eval --datasets <name>_pruned``.

This module imports evalscope; it is only loaded when evalscope is installed.
"""

from evalscope_ext.integration import adapters  # noqa: F401  (registration side effects)
from evalscope_ext.integration.adapters import register_pruned_benchmark  # noqa: F401

__all__ = ['register_pruned_benchmark']
