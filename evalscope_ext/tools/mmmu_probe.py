"""Part B — a forward-looking MMMU image-encoder probe (design + selector).

Generic MMMU accuracy mixes *reasoning* ability with *image-encoder* quality. To tell a
customer whether a candidate's vision encoder is good enough — through the standard OpenAI
``image_url`` interface only — we (1) tag every item by the encoder-stress axis it exercises,
(2) select an axis-balanced probe with the same pruning machinery used for Part A, and
(3) measure the encoder via a paired hard-vs-control contrast (the per-axis accuracy gap
isolates encoder degradation from reasoning).

This module implements (1) and (2) — they run on item metadata alone, no model needed. Step
(3) requires a live endpoint and is specified in :data:`PAIRED_CONTRAST_PROTOCOL` and
Handout A. It deliberately reuses ``PruneItem`` + the registered strategies so the probe is
not a bespoke one-off.
"""
from __future__ import annotations

from typing import Any, Dict, List

from evalscope_ext.pruning.base import PruneContext, PruneItem
from evalscope_ext.pruning.registry import get_pruning_strategy

# Image properties that stress an encoder rather than the reasoner.
ENCODER_STRESS_AXES = [
    'high_resolution',   # large native images lost to the model's fixed downscale
    'text_in_image',     # OCR load — dense text/tables the encoder must read
    'fine_grained',      # small objects, counting, precise localisation
    'multi_image',       # cross-image binding within one prompt
    'chart_diagram',     # structured visuals (plots, schematics, flowcharts)
]

PAIRED_CONTRAST_PROTOCOL = """\
For each selected hard-visual item, construct a matched CONTROL that needs the same
reasoning with minimal visual fidelity (e.g. supply the key fact as text, or a clean
high-res re-render). Query both through the OpenAI image_url interface. The per-axis
accuracy GAP = acc(control) - acc(hard) isolates encoder degradation: a model that holds
on controls but collapses on hard items has a weak encoder, independent of reasoning.
Report per-axis gaps + an aggregate encoder-quality index (1 - mean gap).
"""


def tag_axes(metadata: Dict[str, Any]) -> Dict[str, float]:
    """Heuristically score each encoder-stress axis in [0,1] from item metadata.

    Cheap, model-free signals available per MMMU row: number of images, native
    image dimensions, an OCR/text-density proxy, and the subject/subfield. Replace
    these heuristics with a one-off offline pass (an OCR model, an aspect/resolution
    read) when wiring against the real ~12K dataset.
    """
    n_images = int(metadata.get('num_images', metadata.get('image_count', 1)) or 1)
    width = float(metadata.get('image_width', 0) or 0)
    height = float(metadata.get('image_height', 0) or 0)
    ocr_tokens = float(metadata.get('ocr_token_count', 0) or 0)
    subfield = str(metadata.get('subfield', metadata.get('subject', ''))).lower()

    megapixels = (width * height) / 1e6
    chart_like = any(k in subfield for k in ('chart', 'diagram', 'plot', 'table', 'schemat', 'graph'))
    return {
        'high_resolution': min(1.0, megapixels / 4.0),         # >=4MP saturates
        'text_in_image': min(1.0, ocr_tokens / 200.0),          # >=200 OCR tokens saturates
        'fine_grained': min(1.0, megapixels / 8.0),             # detail tends to live in big images
        'multi_image': 1.0 if n_images > 1 else 0.0,
        'chart_diagram': 1.0 if chart_like else 0.0,
    }


def dominant_axis(metadata: Dict[str, Any]) -> str:
    scores = tag_axes(metadata)
    return max(scores, key=scores.get)


def select_probe(
    items: List[PruneItem],
    ratio: float = 0.05,
    per_axis_strategy: str = 'diversity',
    seed: int = 42,
) -> List[int]:
    """Select an axis-balanced probe: split the budget across encoder-stress axes,
    then cover each axis's items with a registered pruning strategy.

    Each ``PruneItem.metadata`` should carry the MMMU fields ``tag_axes`` reads.
    Items are bucketed by their dominant stress axis so no axis is dropped (random
    sampling would under-represent the rare OCR-heavy / multi-image items).
    """
    buckets: Dict[str, List[int]] = {a: [] for a in ENCODER_STRESS_AXES}
    for i, it in enumerate(items):
        buckets[dominant_axis(it.metadata)].append(i)

    total_k = max(len(ENCODER_STRESS_AXES), round(len(items) * ratio))
    per_axis = max(1, total_k // len(ENCODER_STRESS_AXES))
    strat = get_pruning_strategy(per_axis_strategy)

    chosen: List[int] = []
    for axis, idxs in buckets.items():
        if not idxs:
            continue
        sub_items = [items[i] for i in idxs]
        ctx = PruneContext(dataset_name=f'mmmu-{axis}', seed=seed)
        local_ratio = min(1.0, per_axis / len(sub_items))
        local = strat.select(sub_items, local_ratio, ctx)
        chosen.extend(idxs[j] for j in local)
    return sorted(set(chosen))
