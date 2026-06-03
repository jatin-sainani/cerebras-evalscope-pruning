"""Shared fixtures: a synthetic reference panel with known structure."""
import numpy as np
import pytest

from evalscope_ext.pruning.base import PruneItem
from evalscope_ext.pruning.signals import SignalStore


@pytest.fixture
def synthetic_panel():
    """A 200-item, 4-model panel with a controlled difficulty spread.

    Item difficulties span [0,1]; each model's correctness is drawn from a
    logistic of (model_ability - item_difficulty), so stronger models pass more
    and item discrimination peaks at mid difficulty — exactly the structure the
    strategies should exploit.
    """
    rng = np.random.default_rng(0)
    n = 200
    abilities = {'m_weak': -1.0, 'm_mid1': 0.0, 'm_mid2': 0.2, 'm_strong': 1.2}
    difficulty = np.linspace(-2.5, 2.5, n)  # latent difficulty
    scores_by_model = {}
    for m, a in abilities.items():
        p = 1.0 / (1.0 + np.exp(-(a - difficulty)))
        scores_by_model[m] = {i: float(rng.random() < p[i]) for i in range(n)}
    sig = SignalStore(scores_by_model, judged=False)
    items = [PruneItem(id=i) for i in range(n)]
    return items, sig
