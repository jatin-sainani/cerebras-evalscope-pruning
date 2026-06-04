"""Part B — a working MMMU image-encoder probe.

Pipeline (all runnable):

1. **Tag** every MMMU item by the encoder-stress axis its image exercises, from MMMU's own
   ``img_type`` field + image count (``pruning.strategies.mmmu_encoder``).
2. **Select** an axis-balanced, difficulty-spread probe with the registered
   ``mmmu_encoder_probe`` strategy — the same machinery as Part A.
3. **Measure** the encoder with a paired **hard-vs-control** contrast through the standard
   OpenAI ``image_url`` interface: each hard item is paired with a control that needs the
   same reasoning but minimal visual fidelity. The per-axis accuracy *gap* isolates encoder
   degradation from reasoning ability.

Three entry points:

* ``offline_axis_report`` / CLI default — runs on the **shipped** ``Evals/MMMU`` data (one
  reference model), showing per-axis accuracy + difficulty and proving the probe covers every
  stress axis. No model calls, no network.
* ``load_hf`` — pulls real images from the full ``MMMU/MMMU`` HF dataset (needs ``datasets``).
* ``measure_encoder`` — the live paired-contrast measurement against any OpenAI-compatible
  endpoint (needs ``openai`` + an endpoint; ``--measure`` on the CLI).

    # offline, on shipped data
    python -m evalscope_ext.tools.mmmu_probe --evals "/path/to/Evals/MMMU" --per-axis 8

    # live, against an OpenAI-compatible vision endpoint
    python -m evalscope_ext.tools.mmmu_probe --measure --hf-subjects Math Accounting \\
        --per-axis 5 --base-url https://api.openai.com/v1 --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from evalscope_ext.pruning.base import PruneContext, PruneItem
from evalscope_ext.pruning.registry import get_pruning_strategy
from evalscope_ext.pruning.strategies.mmmu_encoder import (
    ENCODER_STRESS_AXES,
    _norm_img_types,
    axes_for,
    primary_axis,
)


@dataclass
class MMMUItem:
    id: str
    subject: str
    img_type: List[str]
    question_type: str
    difficulty: str
    n_images: int = 1
    correct: Optional[float] = None  # reference-model correctness, when known
    # Populated only for HF-loaded items (needed for the live measurement):
    question: str = ''
    options: List[str] = field(default_factory=list)
    answer: str = ''
    explanation: str = ''
    images: List[bytes] = field(default_factory=list)

    @property
    def metadata(self) -> Dict:
        return {
            'img_type': self.img_type,
            'question_type': self.question_type,
            'topic_difficulty': self.difficulty,
            'n_images': self.n_images,
        }


# --------------------------------------------------------------------------- loading

def load_shipped(mmmu_dir: str, model: Optional[str] = None) -> List[MMMUItem]:
    """Load the shipped reference rows (predictions metadata + review correctness)."""
    pred_root = os.path.join(mmmu_dir, 'predictions')
    rev_root = os.path.join(mmmu_dir, 'reviews')
    if model is None:
        model = sorted(os.listdir(pred_root))[0]
    items: List[MMMUItem] = []
    for pred_path in sorted(glob.glob(os.path.join(pred_root, model, '*.jsonl'))):
        subject = os.path.splitext(os.path.basename(pred_path))[0]
        rev_path = os.path.join(rev_root, model, os.path.basename(pred_path))
        correctness: Dict[int, Optional[float]] = {}
        if os.path.exists(rev_path):
            with open(rev_path, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    val = o['sample_score']['score']['value']
                    correctness[o['index']] = next(
                        (float(v) for v in val.values() if isinstance(v, (int, float))), None
                    )
        with open(pred_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                md = o.get('metadata', {})
                its = _norm_img_types(md.get('img_type'))
                items.append(
                    MMMUItem(
                        id=md.get('id', f'{subject}_{o["index"]}'),
                        subject=subject,
                        img_type=its,
                        question_type=md.get('question_type', 'multiple-choice'),
                        difficulty=md.get('topic_difficulty', 'Medium'),
                        n_images=max(1, len(its)),
                        correct=correctness.get(o['index']),
                    )
                )
    return items


def load_hf(subjects: List[str], split: str = 'validation', max_per_subject: Optional[int] = None) -> List[MMMUItem]:
    """Load real items (with images) from the full ``MMMU/MMMU`` HF dataset."""
    import io

    from datasets import load_dataset  # optional dependency

    items: List[MMMUItem] = []
    for subject in subjects:
        ds = load_dataset('MMMU/MMMU', subject, split=split)
        for i, rec in enumerate(ds):
            if max_per_subject and i >= max_per_subject:
                break
            imgs: List[bytes] = []
            for n in range(1, 8):
                img = rec.get(f'image_{n}')
                if img is not None:
                    buf = io.BytesIO()
                    img.convert('RGB').save(buf, format='PNG')
                    imgs.append(buf.getvalue())
            its = _norm_img_types(rec.get('img_type'))
            options = rec.get('options', []) or []
            if isinstance(options, str):
                try:
                    import ast
                    options = ast.literal_eval(options)
                except (ValueError, SyntaxError):
                    options = [options]
            items.append(
                MMMUItem(
                    id=rec.get('id', f'{subject}_{i}'),
                    subject=subject,
                    img_type=list(its),
                    question_type=rec.get('question_type', 'multiple-choice'),
                    difficulty=rec.get('topic_difficulty', 'Medium'),
                    n_images=len(imgs),
                    question=rec.get('question', ''),
                    options=list(options),
                    answer=str(rec.get('answer', '')),
                    explanation=rec.get('explanation', '') or '',
                    images=imgs,
                )
            )
    return items


# --------------------------------------------------------------------------- selection

def select_probe(items: List[MMMUItem], per_axis: int, seed: int = 42) -> List[MMMUItem]:
    """Axis-balanced, difficulty-spread probe via the registered strategy."""
    n_axes = len(ENCODER_STRESS_AXES)
    ratio = min(1.0, (per_axis * n_axes) / max(len(items), 1))
    prune_items = [PruneItem(id=i, metadata=it.metadata) for i, it in enumerate(items)]
    ctx = PruneContext(dataset_name='mmmu', seed=seed)
    keep = get_pruning_strategy('mmmu_encoder_probe').select(prune_items, ratio, ctx)
    return [items[i] for i in keep]


# --------------------------------------------------------------------------- offline report

@dataclass
class AxisStat:
    axis: str
    n: int
    accuracy: Optional[float]
    mean_difficulty: float


def offline_axis_report(items: List[MMMUItem]) -> List[AxisStat]:
    """Per-axis count, reference-model accuracy, and mean difficulty.

    Demonstrates *which* image categories stress the encoder (low accuracy / high
    difficulty) using the shipped reference model — no live calls required.
    """
    diff_map = {'easy': 0.2, 'medium': 0.5, 'hard': 0.8}
    by_axis: Dict[str, List[MMMUItem]] = defaultdict(list)
    for it in items:
        for ax in axes_for(it.metadata):
            by_axis[ax].append(it)
    stats: List[AxisStat] = []
    for ax in ENCODER_STRESS_AXES:
        bucket = by_axis.get(ax, [])
        if not bucket:
            continue
        accs = [b.correct for b in bucket if b.correct is not None]
        diffs = [diff_map.get(str(b.difficulty).lower(), 0.5) for b in bucket]
        stats.append(
            AxisStat(
                axis=ax,
                n=len(bucket),
                accuracy=(sum(accs) / len(accs)) if accs else None,
                mean_difficulty=sum(diffs) / len(diffs),
            )
        )
    return stats


# --------------------------------------------------------------------------- live measurement

LETTERS = 'ABCDEFGHIJ'


def _format_question(it: MMMUItem) -> str:
    q = it.question
    if it.options:
        opts = '\n'.join(f'{LETTERS[i]}. {o}' for i, o in enumerate(it.options))
        q = f'{q}\n\n{opts}\n\nAnswer with the option letter only.'
    return q


def _control_text(it: MMMUItem) -> str:
    """A control that conveys the same reasoning content with minimal visual demand.

    Substitutes the image with its textual description (the MMMU explanation when present),
    so a model that relies on reasoning rather than the encoder can still answer. Comparing
    accuracy here vs on the real image isolates the encoder's contribution.
    """
    desc = it.explanation.strip() or f'(An image of type {", ".join(it.img_type) or "unknown"}.)'
    return f'{_format_question(it)}\n\n[Image description]: {desc}'


def _image_content(images: List[bytes]) -> List[dict]:
    import base64

    parts = []
    for raw in images:
        b64 = base64.b64encode(raw).decode('ascii')
        parts.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}})
    return parts


def _extract_letter(text: str, n_options: int) -> str:
    m = re.search(r'\b([A-J])\b', text.upper())
    if m and LETTERS.index(m.group(1)) < n_options:
        return m.group(1)
    return text.strip()[:1].upper()


@dataclass
class EncoderReport:
    per_axis_gap: Dict[str, float]
    acc_hard: Dict[str, float]
    acc_control: Dict[str, float]
    n: int

    @property
    def encoder_quality_index(self) -> float:
        """1 - mean per-axis gap; 1.0 = encoder adds no error vs the text control."""
        if not self.per_axis_gap:
            return float('nan')
        return 1.0 - sum(self.per_axis_gap.values()) / len(self.per_axis_gap)


def measure_encoder(client, items: List[MMMUItem], model: str, max_tokens: int = 16) -> EncoderReport:
    """Live paired hard-vs-control measurement through the OpenAI chat interface.

    ``client`` is an ``openai.OpenAI``-compatible client. For each item we ask the model the
    same question twice — once with the real image(s), once with a text-only control — and
    record correctness. The per-axis gap = acc(control) - acc(hard) is the encoder-stress
    signal. Only multiple-choice items with a known answer are scored.
    """
    def ask(content) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': content}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ''

    hard_correct: Dict[str, List[int]] = defaultdict(list)
    ctrl_correct: Dict[str, List[int]] = defaultdict(list)
    scored = 0
    for it in items:
        if it.question_type != 'multiple-choice' or not it.options or not it.answer:
            continue
        ax = primary_axis(it.metadata)
        hard_content = [{'type': 'text', 'text': _format_question(it)}, *_image_content(it.images)]
        ctrl_content = _control_text(it)
        gold = it.answer.strip().upper()[:1]
        hard_correct[ax].append(int(_extract_letter(ask(hard_content), len(it.options)) == gold))
        ctrl_correct[ax].append(int(_extract_letter(ask(ctrl_content), len(it.options)) == gold))
        scored += 1

    acc_hard = {a: sum(v) / len(v) for a, v in hard_correct.items() if v}
    acc_control = {a: sum(v) / len(v) for a, v in ctrl_correct.items() if v}
    gap = {a: max(0.0, acc_control[a] - acc_hard[a]) for a in acc_hard if a in acc_control}
    return EncoderReport(per_axis_gap=gap, acc_hard=acc_hard, acc_control=acc_control, n=scored)


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='MMMU image-encoder probe.')
    ap.add_argument('--evals', help='Path to shipped Evals/MMMU dir (offline report).')
    ap.add_argument('--per-axis', type=int, default=8, help='Probe items per encoder-stress axis.')
    ap.add_argument('--measure', action='store_true', help='Run the live paired-contrast measurement.')
    ap.add_argument('--hf-subjects', nargs='*', default=['Math'], help='MMMU subjects to pull from HF for --measure.')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    args = ap.parse_args(argv)

    if args.measure:
        from openai import OpenAI

        items = load_hf(args.hf_subjects, split=args.split)
        probe = select_probe(items, per_axis=args.per_axis)
        print(f'Selected {len(probe)} probe items across {len(args.hf_subjects)} subjects.')
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        report = measure_encoder(client, probe, model=args.model)
        print(f'\nEncoder probe -- {report.n} items, model={args.model}')
        print(f'{"axis":16s} {"acc(hard)":>10s} {"acc(control)":>13s} {"gap":>7s}')
        for ax in ENCODER_STRESS_AXES:
            if ax in report.acc_hard:
                print(f'{ax:16s} {report.acc_hard[ax]:10.2f} {report.acc_control.get(ax, float("nan")):13.2f} '
                      f'{report.per_axis_gap.get(ax, float("nan")):7.2f}')
        print(f'\nEncoder-quality index (1 - mean gap): {report.encoder_quality_index:.3f}')
        return 0

    if not args.evals:
        ap.error('provide --evals for the offline report, or --measure for the live probe.')
    items = load_shipped(args.evals)
    probe = select_probe(items, per_axis=args.per_axis)
    print(f'Loaded {len(items)} MMMU reference items; selected an axis-balanced probe of {len(probe)}.')
    print(f'\n{"axis":16s} {"items":>6s} {"ref acc":>8s} {"mean diff":>10s}   (full reference set)')
    print('-' * 52)
    for st in offline_axis_report(items):
        acc = f'{st.accuracy:.2f}' if st.accuracy is not None else '   --'
        print(f'{st.axis:16s} {st.n:6d} {acc:>8s} {st.mean_difficulty:10.2f}')
    cov: Dict[str, int] = defaultdict(int)
    for it in probe:
        cov[primary_axis(it.metadata)] += 1
    print(f'\nProbe coverage by primary axis: {dict(cov)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
