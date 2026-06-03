# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import importlib
import os
import time

from evalscope.utils import get_logger

logger = get_logger()

# Using glob to find all files matching the pattern
pattern = os.path.join(os.path.dirname(__file__), '*', '**', '*_adapter.py')
files = glob.glob(pattern, recursive=True)

import_times = []

for file_path in files:
    if file_path.endswith('.py') and not os.path.basename(file_path).startswith('_'):
        # Convert file path to a module path
        relative_path = os.path.relpath(file_path, os.path.dirname(__file__))
        module_path = relative_path[:-3].replace(os.path.sep, '.')  # strip '.py' and convert to module path
        full_path = f'evalscope.benchmarks.{module_path}'

        start_time = time.perf_counter()
        importlib.import_module(full_path)
        end_time = time.perf_counter()

        import_times.append((full_path, end_time - start_time))


def _load_plugin_benchmarks():
    """Import out-of-tree benchmark providers so their @register_benchmark fires.

    Two discovery channels, both optional:
      * the ``EVALSCOPE_PLUGINS`` env var: a comma-separated list of importable
        module names (e.g. ``EVALSCOPE_PLUGINS=evalscope_ext``);
      * the ``evalscope.plugins`` entry-point group, for installed packages.

    This lets an extension such as ``evalscope_ext`` register ``*_pruned``
    benchmarks with the CLI without editing the in-tree glob above.
    """
    modules = [m.strip() for m in os.environ.get('EVALSCOPE_PLUGINS', '').split(',') if m.strip()]
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on user env
            logger.warning(f'Failed to import EVALSCOPE_PLUGINS module {name!r}: {exc}')

    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group = eps.select(group='evalscope.plugins') if hasattr(eps, 'select') else eps.get('evalscope.plugins', [])
        for ep in group:
            try:
                ep.load()
            except Exception as exc:  # pragma: no cover
                logger.warning(f'Failed to load evalscope.plugins entry point {ep!r}: {exc}')
    except Exception:  # pragma: no cover
        pass


_load_plugin_benchmarks()
