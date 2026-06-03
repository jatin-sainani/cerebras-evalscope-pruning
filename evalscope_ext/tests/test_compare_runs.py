import json
import os

import pytest

from evalscope_ext.tools.compare_runs import compare


def _write_run(root, model, dataset, scores):
    d = os.path.join(root, 'reviews', model)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'{dataset}.jsonl'), 'w', encoding='utf-8') as f:
        for i, s in enumerate(scores):
            f.write(json.dumps({'index': i, 'sample_score': {'score': {'value': {'acc': s}}}}) + '\n')


def test_verdict_preserved_when_subset_matches(tmp_path):
    full = tmp_path / 'full'
    pruned = tmp_path / 'pruned'
    # Full: 10 items, score 0.7. Pruned subset of 4 also ~0.7.
    _write_run(str(full), 'modelA', 'live_code_bench', [1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    _write_run(str(pruned), 'modelA', 'live_code_bench_pruned', [1, 1, 1, 0])

    report = compare(str(full), str(pruned), threshold=0.5)
    assert len(report.comparisons) == 1
    c = report.comparisons[0]
    assert c.dataset == 'live_code_bench_pruned'
    assert c.full_score == 0.7
    assert c.verdict_full and c.verdict_pruned  # both GO
    assert report.verdict_agreement() == 1.0
    assert c.ci[0] <= c.pruned_score <= c.ci[1]


def test_rank_correlation_across_models(tmp_path):
    full = tmp_path / 'full'
    pruned = tmp_path / 'pruned'
    # Strong, mid, weak models; pruned subset preserves ordering.
    _write_run(str(full), 'strong', 'live_code_bench', [1] * 9 + [0])
    _write_run(str(full), 'mid', 'live_code_bench', [1] * 6 + [0] * 4)
    _write_run(str(full), 'weak', 'live_code_bench', [1] * 3 + [0] * 7)
    _write_run(str(pruned), 'strong', 'live_code_bench_pruned', [1, 1, 1, 1])
    _write_run(str(pruned), 'mid', 'live_code_bench_pruned', [1, 1, 0, 0])
    _write_run(str(pruned), 'weak', 'live_code_bench_pruned', [1, 0, 0, 0])

    report = compare(str(full), str(pruned), threshold=0.5)
    assert report.rank_spearman['live_code_bench'] == pytest.approx(1.0)


def test_judged_dataset_widens_ci(tmp_path):
    full = tmp_path / 'full'
    pruned = tmp_path / 'pruned'
    _write_run(str(full), 'm', 'aa_lcr', [1, 0] * 25)
    _write_run(str(pruned), 'm', 'aa_lcr_pruned', [1, 0, 1, 0])
    report = compare(str(full), str(pruned), threshold=0.5)
    c = report.comparisons[0]
    assert c.judged is True
    # The CI should be non-degenerate and around 0.5.
    assert c.ci[0] < 0.5 < c.ci[1]
