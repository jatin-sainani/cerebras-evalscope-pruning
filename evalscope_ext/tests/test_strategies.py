import numpy as np
import pytest

from evalscope_ext.pruning.base import PruneContext, PruneItem
from evalscope_ext.pruning.registry import get_pruning_strategy, list_pruning_strategies
from evalscope_ext.pruning.strategies.discrimination import discrimination_scores

ALL = ['balanced', 'difficulty_stratified', 'discrimination', 'diversity', 'judge_noise_aware']


def test_all_strategies_registered():
    assert set(ALL).issubset(set(list_pruning_strategies()))


@pytest.mark.parametrize('name', ALL)
@pytest.mark.parametrize('ratio', [0.05, 0.1, 0.25, 0.5])
def test_select_count_and_validity(synthetic_panel, name, ratio):
    items, sig = synthetic_panel
    ctx = PruneContext(dataset_name='syn', signals=sig)
    keep = get_pruning_strategy(name).select(items, ratio, ctx)
    expected = max(1, round(len(items) * ratio))
    assert len(keep) == expected
    assert len(set(keep)) == len(keep)  # no duplicates
    assert all(0 <= i < len(items) for i in keep)


@pytest.mark.parametrize('name', ALL)
def test_reproducible(synthetic_panel, name):
    items, sig = synthetic_panel
    ctx = PruneContext(dataset_name='syn', signals=sig)
    a = get_pruning_strategy(name).select(items, 0.1, ctx)
    b = get_pruning_strategy(name).select(items, 0.1, ctx)
    assert a == b


def test_discrimination_avoids_unanimous_items(synthetic_panel):
    """Items everyone passes/fails carry no go/no-go signal and should be dropped."""
    items, sig = synthetic_panel
    ctx = PruneContext(dataset_name='syn', signals=sig)
    keep = set(get_pruning_strategy('discrimination').select(items, 0.1, ctx))
    # Unanimous items have difficulty 0 or 1.
    unanimous = {it.id for it in items if sig.difficulty(it.id) in (0.0, 1.0)}
    overlap = keep & unanimous
    # Discrimination should pick almost no unanimous items.
    assert len(overlap) <= 1


def test_discrimination_scores_peak_midrange():
    # Fisher info p(1-p) is maximal at p=0.5 and zero at the extremes.
    mat = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=float)
    scores = discrimination_scores(mat)
    assert scores[0] == pytest.approx(0.0)  # all fail -> no signal
    assert scores[1] == pytest.approx(0.0)  # all pass -> no signal
    # Split items (rows 2, 3) carry the discrimination signal.
    assert scores[2] > scores[0]
    assert scores[3] > scores[1]
    assert scores.max() > 0


def test_stratified_preserves_difficulty_distribution(synthetic_panel):
    """The kept set's mean difficulty should track the full set's (unbiased)."""
    items, sig = synthetic_panel
    ctx = PruneContext(dataset_name='syn', signals=sig)
    keep = get_pruning_strategy('difficulty_stratified').select(items, 0.2, ctx)
    full_mean = np.mean([sig.difficulty(it.id) for it in items])
    kept_mean = np.mean([sig.difficulty(items[i].id) for i in keep])
    assert abs(kept_mean - full_mean) < 0.08


def test_no_signals_falls_back_to_coverage():
    # With no reference panel, strategies must still return a valid subset.
    items = [PruneItem(id=i, metadata={}) for i in range(50)]
    ctx = PruneContext(dataset_name='syn', signals=None)
    for name in ALL:
        keep = get_pruning_strategy(name).select(items, 0.2, ctx)
        assert len(keep) == 10
        assert all(0 <= i < 50 for i in keep)


def test_ratio_one_keeps_everything(synthetic_panel):
    items, sig = synthetic_panel
    ctx = PruneContext(dataset_name='syn', signals=sig)
    keep = get_pruning_strategy('balanced').select(items, 1.0, ctx)
    assert sorted(keep) == list(range(len(items)))
