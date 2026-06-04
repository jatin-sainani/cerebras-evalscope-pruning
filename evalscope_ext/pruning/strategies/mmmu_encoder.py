"""MMMU image-encoder probe selection (Part B).

Generic MMMU accuracy conflates *reasoning* with *image-encoder* quality. To probe the
encoder specifically and cheaply, we (1) tag each item by the encoder-stress axis its image
exercises — read straight from MMMU's own ``img_type`` field plus the image count — and
(2) select an **axis-balanced, difficulty-spread** probe. Random sampling of 12K items would
bury the encoder-stressing categories (dense tables, multi-image binding, fine detail are a
small slice); balancing across axes guarantees each failure mode is probed.

This is a registered pruning strategy, so it runs through the same universal pruned-benchmark
machinery as Part A:

    evalscope eval --datasets mmmu_pruned \\
        --dataset-args '{"pruning_strategy": "mmmu_encoder_probe", "prune_ratio": 0.05}'

The measurement that turns a probe set into an encoder-quality number (a paired hard-vs-control
contrast through the OpenAI image interface) lives in ``evalscope_ext.tools.mmmu_probe``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from evalscope_ext.pruning.base import PruneContext, PruneItem, PruningStrategy, target_count
from evalscope_ext.pruning.registry import register_pruning_strategy

# The encoder-stress axes, in priority order (an item is assigned to its highest-priority
# matching axis for budget allocation, but reported under all that apply).
ENCODER_STRESS_AXES = [
    'text_in_image',   # OCR load: dense text/tables the encoder must read
    'multi_image',     # cross-image binding within one prompt
    'fine_grained',    # small objects, domain imagery, precise localisation/counting
    'chart_diagram',   # structured visuals (plots, schematics, blueprints)
    'natural_image',   # general perception (photos, paintings)
]

# MMMU's 30+ img_type categories -> our 5 encoder-stress axes.
IMG_TYPE_TO_AXIS: Dict[str, str] = {
    # text-in-image / OCR
    'Tables': 'text_in_image', 'Screenshots': 'text_in_image', 'Mathematical Notations': 'text_in_image',
    'Icons and Symbols': 'text_in_image', 'Logos and Branding': 'text_in_image', 'Poster': 'text_in_image',
    'Advertisements': 'text_in_image', 'Sheet Music': 'text_in_image',
    # charts / diagrams / structured
    'Diagrams': 'chart_diagram', 'Plots and Charts': 'chart_diagram', 'Technical Blueprints': 'chart_diagram',
    'Trees and Graphs': 'chart_diagram', 'Geometric Shapes': 'chart_diagram', 'Maps': 'chart_diagram',
    'Sketches and Drafts': 'chart_diagram', 'DNA Sequences': 'chart_diagram',
    # fine-grained / domain imagery
    'Medical Images': 'fine_grained', 'Microscopic Images': 'fine_grained', 'Pathological Images': 'fine_grained',
    'Body Scans: MRI, CT scans, and X-rays': 'fine_grained', 'Chemical Structures': 'fine_grained',
    # natural images
    'Photographs': 'natural_image', 'Paintings': 'natural_image', 'Portraits': 'natural_image',
    'Sculpture': 'natural_image', 'Comics and Cartoons': 'natural_image', 'Landscapes': 'natural_image',
    'Other': 'natural_image',
}


def _norm_img_types(raw) -> List[str]:
    """img_type may be a list, a stringified list, or a single string."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith('['):
            import ast
            try:
                raw = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                raw = [s]
        else:
            raw = [s]
    return [str(x) for x in raw]


def axes_for(metadata: Dict) -> List[str]:
    """The encoder-stress axes an item exercises, from its metadata."""
    img_types = _norm_img_types(metadata.get('img_type'))
    axes = {IMG_TYPE_TO_AXIS.get(t, 'natural_image') for t in img_types} if img_types else set()
    n_images = int(metadata.get('n_images', len(img_types) if len(img_types) > 1 else 1) or 1)
    if n_images > 1 or len(img_types) > 1:
        axes.add('multi_image')
    return [a for a in ENCODER_STRESS_AXES if a in axes] or ['natural_image']


def primary_axis(metadata: Dict) -> str:
    """Highest-priority axis, used for disjoint budget allocation."""
    return axes_for(metadata)[0]


_DIFFICULTY_MAP = {'easy': 0.2, 'medium': 0.5, 'hard': 0.8}


def _difficulty(item: PruneItem, ctx: PruneContext) -> float:
    if ctx.signals is not None and ctx.signals.has(item.id):
        d = ctx.signals.difficulty(item.id)
        if d is not None:
            return d
    td = str(item.metadata.get('topic_difficulty', '')).lower()
    return _DIFFICULTY_MAP.get(td, 0.5)


def _even_spread(order: Sequence[int], take: int) -> List[int]:
    """Pick ``take`` items evenly spaced along a difficulty-sorted order."""
    if take >= len(order):
        return list(order)
    idx = np.linspace(0, len(order) - 1, take).round().astype(int)
    return [order[i] for i in dict.fromkeys(idx)]  # dedupe, keep order


@register_pruning_strategy('mmmu_encoder_probe')
class MMMUEncoderProbe(PruningStrategy):
    """Axis-balanced, difficulty-spread probe selection for image-encoder stress."""

    def select(self, items: List[PruneItem], ratio: float, ctx: PruneContext) -> List[int]:
        n = len(items)
        k = target_count(n, ratio)

        # Bucket items by primary axis.
        buckets: Dict[str, List[int]] = {a: [] for a in ENCODER_STRESS_AXES}
        for i, it in enumerate(items):
            buckets[primary_axis(it.metadata)].append(i)
        present = [a for a in ENCODER_STRESS_AXES if buckets[a]]
        if not present:
            return sorted(range(min(k, n)))

        # Balance the budget across axes (equal share, remainder to the largest axes),
        # capped by each axis's availability.
        base = k // len(present)
        alloc = {a: min(base, len(buckets[a])) for a in present}
        leftover = k - sum(alloc.values())
        for a in sorted(present, key=lambda x: -len(buckets[x])):
            if leftover <= 0:
                break
            room = len(buckets[a]) - alloc[a]
            add = min(room, leftover)
            alloc[a] += add
            leftover -= add

        chosen: List[int] = []
        for a in present:
            take = alloc[a]
            if take <= 0:
                continue
            order = sorted(buckets[a], key=lambda i: _difficulty(items[i], ctx))
            chosen.extend(_even_spread(order, take))
        return sorted(chosen[:k])
