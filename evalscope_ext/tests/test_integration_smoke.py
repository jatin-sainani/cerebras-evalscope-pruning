"""Smoke test for the evalscope integration.

Skips automatically when evalscope (or one of its heavy/compiled deps) is not
installed, so the pruning core can be tested in a minimal environment. In a full
evalscope install this verifies the pruned benchmarks register and that prune
config resolves from dataset args.
"""
import pytest

evalscope = pytest.importorskip('evalscope')


def test_pruned_benchmarks_registered():
    import evalscope_ext.integration  # noqa: F401  (registration side effects)
    from evalscope.api.registry import BENCHMARK_REGISTRY

    # live_code_bench is always available; its pruned variant must register.
    assert 'live_code_bench_pruned' in BENCHMARK_REGISTRY
    meta = BENCHMARK_REGISTRY['live_code_bench_pruned']
    assert 'pruning_strategy' in meta.extra_params
    assert 'prune_ratio' in meta.extra_params


def test_mixin_in_mro():
    import evalscope_ext.integration  # noqa: F401
    from evalscope.api.registry import BENCHMARK_REGISTRY
    from evalscope_ext.integration.mixin import PruningMixin

    adapter_cls = BENCHMARK_REGISTRY['live_code_bench_pruned'].data_adapter
    assert PruningMixin in adapter_cls.__mro__


def test_prune_config_reads_top_level_and_extra_params():
    """The literal run-contract (top-level) form and extra_params form both resolve."""
    import evalscope_ext.integration  # noqa: F401
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    cfg = TaskConfig(
        model='dummy',
        datasets=['live_code_bench_pruned'],
        dataset_args={'live_code_bench_pruned': {'pruning_strategy': 'balanced', 'prune_ratio': 0.1}},
    )
    adapter = get_benchmark('live_code_bench_pruned', cfg)
    pc = adapter._prune_config()
    assert pc.strategy == 'balanced'
    assert pc.ratio == 0.1


def test_mixin_prunes_loaded_dataset_end_to_end():
    """Exercise PruningMixin.load_dataset over real Sample/MemoryDataset objects.

    Uses a stub base adapter so no model inference or dataset download is needed;
    the pruning transform itself (reindex -> select -> rebuild) runs for real
    against the bundled LiveCodeBench signal cache.
    """
    from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
    from evalscope_ext.integration.mixin import PruningMixin

    class _FakeBase:
        name = 'live_code_bench_pruned'
        metric_list = ['acc']
        seed = 42
        _task_config = None

        def __init__(self, dd):
            self._dd = dd
            # extra_params drives the prune config in this harness.
            self.extra_params = {
                'pruning_strategy': 'difficulty_stratified',
                'prune_ratio': 0.1,
                'prune_seed': 42,
                'strategy_args': {},
                'signals_path': None,
            }

        def load_dataset(self):
            return self._dd

    class _Pruned(PruningMixin, _FakeBase):
        pruning_base_name = 'live_code_bench'

    n = 315  # matches the shipped LCB signal index range 0..314
    ds = MemoryDataset([Sample(input='q', metadata={}) for _ in range(n)], name='release_v5')
    adapter = _Pruned(DatasetDict({'release_v5': ds}))

    out = adapter.load_dataset()
    pruned = out['release_v5']
    assert len(pruned) == round(0.1 * n)  # 32 kept
    # Rebuilt dataset is reindexed 0..k-1.
    assert [s.id for s in pruned] == list(range(len(pruned)))
