"""Tests for the MMMU image-encoder probe (Part B)."""
import os

import pytest

from evalscope_ext.pruning.base import PruneContext, PruneItem
from evalscope_ext.pruning.registry import get_pruning_strategy, list_pruning_strategies
from evalscope_ext.pruning.strategies.mmmu_encoder import (
    ENCODER_STRESS_AXES,
    axes_for,
    primary_axis,
)
from evalscope_ext.tools import mmmu_probe


def test_img_type_maps_to_axes():
    assert primary_axis({'img_type': ['Tables']}) == 'text_in_image'
    assert primary_axis({'img_type': ['Plots and Charts']}) == 'chart_diagram'
    assert primary_axis({'img_type': ['Pathological Images']}) == 'fine_grained'
    assert primary_axis({'img_type': ['Photographs']}) == 'natural_image'
    # multi-image is detected from image count.
    assert 'multi_image' in axes_for({'img_type': ['Photographs'], 'n_images': 2})


def test_img_type_accepts_stringified_list():
    # The shipped predictions store img_type as a stringified list.
    assert primary_axis({'img_type': "['Tables']"}) == 'text_in_image'


def test_probe_strategy_registered():
    assert 'mmmu_encoder_probe' in list_pruning_strategies()


def test_probe_strategy_balances_axes():
    types = ['Tables', 'Plots and Charts', 'Pathological Images', 'Photographs']
    items = [
        PruneItem(id=i, metadata={'img_type': [types[i % len(types)]], 'topic_difficulty': 'Medium'})
        for i in range(100)
    ]
    keep = get_pruning_strategy('mmmu_encoder_probe').select(items, 0.2, PruneContext(dataset_name='mmmu'))
    assert len(keep) == 20
    axes_covered = {primary_axis(items[i].metadata) for i in keep}
    assert axes_covered == {'text_in_image', 'chart_diagram', 'fine_grained', 'natural_image'}


# --- real shipped-data tests (skip when the Evals/MMMU tree is unavailable) ---

def _find_mmmu():
    candidates = [
        os.environ.get('EVALSCOPE_EXT_MMMU', ''),
        os.path.join(os.path.dirname(__file__), '..', '..', '..',
                     'ai-model-quality-challenge', 'Evals', 'MMMU'),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, 'predictions')):
            return c
    return None


MMMU = _find_mmmu()
needs_mmmu = pytest.mark.skipif(MMMU is None, reason='shipped Evals/MMMU data not available')


@needs_mmmu
def test_load_shipped_and_offline_report():
    items = mmmu_probe.load_shipped(MMMU)
    assert len(items) > 0
    stats = mmmu_probe.offline_axis_report(items)
    axes = {s.axis for s in stats}
    assert axes.issubset(set(ENCODER_STRESS_AXES))
    assert len(axes) >= 4
    assert all(s.accuracy is None or 0.0 <= s.accuracy <= 1.0 for s in stats)


@needs_mmmu
def test_select_probe_is_axis_balanced_on_real_data():
    items = mmmu_probe.load_shipped(MMMU)
    probe = mmmu_probe.select_probe(items, per_axis=8)
    cov = {}
    for it in probe:
        ax = primary_axis(it.metadata)
        cov[ax] = cov.get(ax, 0) + 1
    assert len(cov) >= 4
    assert min(cov.values()) >= 4
