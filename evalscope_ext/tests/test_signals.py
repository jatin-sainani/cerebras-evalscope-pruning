import json
import os

from evalscope_ext.pruning.signals import SignalStore


def _write_reviews(path, scores, key='acc'):
    with open(path, 'w', encoding='utf-8') as f:
        for idx, s in scores.items():
            f.write(json.dumps({'index': idx, 'sample_score': {'score': {'value': {key: s}}}}) + '\n')


def test_from_evals_and_signal_values(tmp_path):
    part1 = tmp_path / 'Part 1'
    reviews = part1 / 'reviews'
    reviews.mkdir(parents=True)
    # aa_lcr: two models, 3 items.
    _write_reviews(reviews / 'aa_lcr__modelA.jsonl', {0: 1.0, 1: 0.0, 2: 1.0})
    _write_reviews(reviews / 'aa_lcr__modelB.jsonl', {0: 1.0, 1: 0.0, 2: 0.0})

    sig = SignalStore.from_evals(str(part1), 'aa_lcr')
    assert sig.models == ['modelA', 'modelB']
    assert sig.scores(0) == {'modelA': 1.0, 'modelB': 1.0}
    assert sig.difficulty(0) == 0.0   # both pass
    assert sig.difficulty(1) == 1.0   # both fail
    assert sig.difficulty(2) == 0.5   # split
    # aa_lcr is judged -> verdict fragility proxy p(1-p); item 2 (p=0.5) is max.
    assert sig.judge_variance(2) > sig.judge_variance(0)
    assert sig.judge_variance(0) == 0.0


def test_deterministic_grader_has_zero_judge_variance(tmp_path):
    part1 = tmp_path / 'Part 1'
    reviews = part1 / 'reviews'
    reviews.mkdir(parents=True)
    _write_reviews(reviews / 'live_code_bench_v5__m1.jsonl', {0: 1.0, 1: 0.0}, key='pass')
    _write_reviews(reviews / 'live_code_bench_v5__m2.jsonl', {0: 0.0, 1: 0.0}, key='pass')
    sig = SignalStore.from_evals(str(part1), 'live_code_bench')
    assert all(sig.judge_variance(i) == 0.0 for i in sig.item_ids)


def test_cache_round_trip(tmp_path):
    scores = {'a': {0: 1.0, 1: 0.0}, 'b': {0: 0.0, 1: 1.0}}
    sig = SignalStore(scores, judged=True)
    path = str(tmp_path / 'sig.json')
    sig.to_cache(path)
    assert os.path.exists(path)
    restored = SignalStore.from_cache(path)
    assert restored.models == sig.models
    assert restored.scores(0) == sig.scores(0)
    assert restored.difficulty(1) == sig.difficulty(1)
