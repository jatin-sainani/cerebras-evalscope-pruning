"""Real-data validation: principled pruning must beat the forbidden baselines.

Runs the leave-one-model-out evaluation on the shipped challenge data when it is
available (set ``EVALSCOPE_EXT_EVALS`` to the ``Evals/Part 1`` directory, or rely
on the default sibling-repo path). Skips cleanly when the data is absent, so the
fork's test suite is green for reviewers who don't have the synthetic Evals tree.
"""
import os

import pytest

from evalscope_ext.tools.validate_offline import evaluate

_CANDIDATES = [
    os.environ.get('EVALSCOPE_EXT_EVALS', ''),
    os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'ai-model-quality-challenge', 'Evals', 'Part 1',
    ),
]


def _find_evals():
    for c in _CANDIDATES:
        if c and os.path.isdir(os.path.join(c, 'reviews')):
            return c
    return None


EVALS = _find_evals()
needs_data = pytest.mark.skipif(EVALS is None, reason='shipped Evals/Part 1 data not available')


@needs_data
def test_lcb_principled_beats_baselines():
    results = {r.strategy: r for r in evaluate(EVALS, 'live_code_bench', ratio=0.1)}
    random_mae = results['random'].mae

    # Distribution-preserving pruning estimates the held-out model's true score
    # far better than uniform random (the headline claim).
    assert results['difficulty_stratified'].mae < random_mae
    assert results['balanced'].mae < random_mae

    # The forbidden top-k-hardest baseline is badly biased — much worse than
    # both random and our strategies.
    assert results['hardest'].mae > results['difficulty_stratified'].mae
    assert results['hardest'].mae > random_mae

    # Pruned to 10% of the benchmark.
    assert results['difficulty_stratified'].n_kept == round(0.1 * results['difficulty_stratified'].n_full)


@needs_data
def test_aa_lcr_beats_random():
    results = {r.strategy: r for r in evaluate(EVALS, 'aa_lcr', ratio=0.1)}
    # On the noisy LLM-judged benchmark the best principled strategy still beats
    # random; we assert at least one does (variance is high at n=10).
    principled = ['balanced', 'discrimination', 'difficulty_stratified', 'judge_noise_aware']
    assert any(results[s].mae < results['random'].mae for s in principled)
