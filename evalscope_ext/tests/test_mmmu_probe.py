from evalscope_ext.pruning.base import PruneItem
from evalscope_ext.tools.mmmu_probe import (
    ENCODER_STRESS_AXES,
    dominant_axis,
    select_probe,
    tag_axes,
)


def test_tag_axes_detects_stress():
    ocr = tag_axes({'ocr_token_count': 400, 'image_width': 800, 'image_height': 600})
    assert ocr['text_in_image'] == 1.0
    multi = tag_axes({'num_images': 3})
    assert multi['multi_image'] == 1.0
    chart = tag_axes({'subfield': 'Economics charts'})
    assert chart['chart_diagram'] == 1.0
    hires = tag_axes({'image_width': 3000, 'image_height': 2000})
    assert hires['high_resolution'] == 1.0


def test_dominant_axis():
    assert dominant_axis({'num_images': 4}) == 'multi_image'
    assert dominant_axis({'ocr_token_count': 500}) == 'text_in_image'


def test_select_probe_is_axis_balanced():
    # Build 250 items spread across all five axes.
    items = []
    specs = [
        {'image_width': 3000, 'image_height': 2000},   # high_resolution
        {'ocr_token_count': 400},                       # text_in_image
        {'num_images': 3},                              # multi_image
        {'subfield': 'diagram'},                        # chart_diagram
        {'image_width': 1500, 'image_height': 1500},    # fine_grained-ish
    ]
    for i in range(250):
        items.append(PruneItem(id=i, metadata=dict(specs[i % len(specs)])))

    probe = select_probe(items, ratio=0.1, seed=0)
    assert len(probe) > 0
    # Every axis that has items should be represented in the probe.
    covered = {dominant_axis(items[i].metadata) for i in probe}
    present = {dominant_axis(it.metadata) for it in items}
    assert covered == present
    assert covered.issubset(set(ENCODER_STRESS_AXES))
