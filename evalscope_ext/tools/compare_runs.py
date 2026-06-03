"""Compare a full eval run against a pruned one and report whether the pruned
subset preserves the go/no-go verdict.

Reads two evalscope output directories (each containing ``reviews/<model>/<dataset>.jsonl``
per-sample scores), pairs them by model and base dataset name (stripping a
``_pruned`` suffix), and reports per model/dataset:

* full score, pruned score, and their delta;
* a bootstrap CI on the pruned score (estimation uncertainty from the subset);
* verdict agreement at a go/no-go threshold (and a flag if the CI straddles it);
* judge-noise-widened CI for LLM-judged benchmarks;
* cross-model rank correlation (Spearman) when >= 2 models are present.

Usage (matches the challenge run contract):
    python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/ \
        --threshold 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Benchmarks graded by a non-deterministic LLM judge: their pruned score CI is
# additionally widened by an estimate of judge noise.
JUDGED_DATASETS = {'aa_lcr', 'aa_lcr_pruned'}


def _read_review_scores(path: str) -> List[float]:
    """Read per-sample scores from an evalscope reviews .jsonl."""
    scores: List[float] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                value = obj['sample_score']['score']['value']
            except (KeyError, TypeError):
                continue
            num = next((float(v) for v in value.values() if isinstance(v, (int, float))), None)
            if num is not None:
                scores.append(num)
    return scores


def _discover(run_dir: str) -> Dict[Tuple[str, str], List[float]]:
    """Map (model, dataset) -> scores for every reviews file under a run dir."""
    out: Dict[Tuple[str, str], List[float]] = {}
    for path in glob.glob(os.path.join(run_dir, '**', 'reviews', '*', '*.jsonl'), recursive=True):
        model = os.path.basename(os.path.dirname(path))
        dataset = os.path.splitext(os.path.basename(path))[0]
        out[(model, dataset)] = _read_review_scores(path)
    # Also support a flatter reviews/<model>/<dataset>.jsonl layout.
    for path in glob.glob(os.path.join(run_dir, 'reviews', '*', '*.jsonl')):
        model = os.path.basename(os.path.dirname(path))
        dataset = os.path.splitext(os.path.basename(path))[0]
        out.setdefault((model, dataset), _read_review_scores(path))
    return out


def _base_dataset(name: str) -> str:
    return name[:-len('_pruned')] if name.endswith('_pruned') else name


def _bootstrap_ci(scores: List[float], n_boot: int, seed: int, extra_var: float = 0.0) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean; ``extra_var`` widens for judge noise."""
    if not scores:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    means = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    if extra_var > 0:
        means = means + rng.normal(0.0, np.sqrt(extra_var), size=n_boot)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _spearman(a: List[float], b: List[float]) -> float:
    if len(a) < 2:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom else float('nan')


@dataclass
class Comparison:
    model: str
    dataset: str
    full_score: float
    pruned_score: float
    n_full: int
    n_pruned: int
    ci: Tuple[float, float]
    verdict_full: bool
    verdict_pruned: bool
    ci_straddles: bool
    judged: bool


@dataclass
class Report:
    threshold: float
    comparisons: List[Comparison] = field(default_factory=list)
    rank_spearman: Dict[str, float] = field(default_factory=dict)

    def verdict_agreement(self) -> float:
        if not self.comparisons:
            return float('nan')
        agree = sum(c.verdict_full == c.verdict_pruned for c in self.comparisons)
        return agree / len(self.comparisons)


def compare(
    full_dir: str,
    pruned_dir: str,
    threshold: float = 0.5,
    n_boot: int = 2000,
    seed: int = 0,
) -> Report:
    full = _discover(full_dir)
    pruned = _discover(pruned_dir)

    report = Report(threshold=threshold)
    # Index full by (model, base_dataset) for pairing.
    full_by_key = {(m, _base_dataset(d)): s for (m, d), s in full.items()}

    # Group pruned comparisons by base dataset for rank correlation.
    by_dataset_full: Dict[str, List[Tuple[str, float]]] = {}
    by_dataset_pruned: Dict[str, List[Tuple[str, float]]] = {}

    for (model, dataset), p_scores in sorted(pruned.items()):
        base = _base_dataset(dataset)
        f_scores = full_by_key.get((model, base))
        if not f_scores or not p_scores:
            continue
        full_score = float(np.mean(f_scores))
        pruned_score = float(np.mean(p_scores))
        judged = base in {_base_dataset(d) for d in JUDGED_DATASETS}
        # Judge-noise term: a Bernoulli-flip variance proxy on the subset mean.
        extra_var = (pruned_score * (1 - pruned_score) / max(len(p_scores), 1)) if judged else 0.0
        ci = _bootstrap_ci(p_scores, n_boot, seed, extra_var=extra_var)
        comp = Comparison(
            model=model,
            dataset=dataset,
            full_score=full_score,
            pruned_score=pruned_score,
            n_full=len(f_scores),
            n_pruned=len(p_scores),
            ci=ci,
            verdict_full=full_score >= threshold,
            verdict_pruned=pruned_score >= threshold,
            ci_straddles=ci[0] <= threshold <= ci[1],
            judged=judged,
        )
        report.comparisons.append(comp)
        by_dataset_full.setdefault(base, []).append((model, full_score))
        by_dataset_pruned.setdefault(base, []).append((model, pruned_score))

    for base, full_list in by_dataset_full.items():
        pruned_list = dict(by_dataset_pruned.get(base, []))
        models = [m for m, _ in full_list if m in pruned_list]
        if len(models) >= 2:
            fm = dict(full_list)
            report.rank_spearman[base] = _spearman(
                [fm[m] for m in models], [pruned_list[m] for m in models]
            )
    return report


def format_report(report: Report) -> str:
    lines = [
        f'\nVerdict threshold: score >= {report.threshold:g}  ->  GO',
        f'{"model":18s} {"dataset":22s} {"full":>6s} {"pruned":>7s} {"delta":>7s} '
        f'{"95% CI":>15s} {"verdict":>10s}',
        '-' * 92,
    ]
    for c in report.comparisons:
        delta = c.pruned_score - c.full_score
        agree = '=' if c.verdict_full == c.verdict_pruned else 'FLIP!'
        v = f'{"GO" if c.verdict_full else "NO"}->{"GO" if c.verdict_pruned else "NO"} {agree}'
        ci = f'[{c.ci[0]:.2f},{c.ci[1]:.2f}]'
        straddle = ' *' if c.ci_straddles else ''
        lines.append(
            f'{c.model:18s} {c.dataset:22s} {c.full_score:6.3f} {c.pruned_score:7.3f} '
            f'{delta:+7.3f} {ci:>15s} {v:>10s}{straddle}'
        )
    lines.append('-' * 92)
    if report.comparisons:
        n_full = report.comparisons[0].n_full
        n_pruned = report.comparisons[0].n_pruned
        lines.append(
            f'Kept {n_pruned}/{n_full} samples ({n_pruned / max(n_full,1):.0%}). '
            f'Verdict agreement: {report.verdict_agreement():.0%}.'
        )
    for base, rho in report.rank_spearman.items():
        lines.append(f'Cross-model rank correlation ({base}): Spearman = {rho:.2f}')
    if any(c.ci_straddles for c in report.comparisons):
        lines.append('* pruned-score CI straddles the threshold: subset cannot resolve the verdict here — keep more samples.')
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Compare a full vs pruned evalscope run.')
    ap.add_argument('--full', required=True, help='Output dir of the full run.')
    ap.add_argument('--pruned', required=True, help='Output dir of the pruned run.')
    ap.add_argument('--threshold', type=float, default=0.5, help='Go/no-go score threshold.')
    ap.add_argument('--bootstrap', type=int, default=2000, help='Bootstrap resamples for the CI.')
    ap.add_argument('--json', action='store_true', help='Emit machine-readable JSON instead of a table.')
    args = ap.parse_args(argv)

    report = compare(args.full, args.pruned, threshold=args.threshold, n_boot=args.bootstrap)
    if args.json:
        print(json.dumps({
            'threshold': report.threshold,
            'verdict_agreement': report.verdict_agreement(),
            'rank_spearman': report.rank_spearman,
            'comparisons': [c.__dict__ for c in report.comparisons],
        }, indent=2, default=list))
    else:
        print(format_report(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
