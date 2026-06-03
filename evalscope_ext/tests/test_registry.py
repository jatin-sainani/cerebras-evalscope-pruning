import pytest

from evalscope_ext.pruning.base import PruneContext, PruneItem, PruningStrategy
from evalscope_ext.pruning.registry import (
    get_pruning_strategy,
    list_pruning_strategies,
    register_pruning_strategy,
)


def test_get_unknown_raises():
    with pytest.raises(ValueError, match='not registered'):
        get_pruning_strategy('does_not_exist')


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match='already registered'):
        @register_pruning_strategy('balanced')  # already taken
        class _Dup(PruningStrategy):
            def select(self, items, ratio, ctx):
                return []


def test_register_and_use_custom_strategy():
    @register_pruning_strategy('first_k_test')
    class FirstK(PruningStrategy):
        def select(self, items, ratio, ctx):
            k = max(1, round(len(items) * ratio))
            return list(range(k))

    assert 'first_k_test' in list_pruning_strategies()
    strat = get_pruning_strategy('first_k_test')
    items = [PruneItem(id=i) for i in range(10)]
    assert strat.select(items, 0.3, PruneContext()) == [0, 1, 2]
    assert strat.name == 'first_k_test'
