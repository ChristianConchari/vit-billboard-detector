import pytest

from vit.eval.localization import (
    IOU_THRESHOLDS,
    ap_per_iou_threshold,
    boxes_by_image,
    edge_bias,
    match_to_ground_truth,
)

GROUND_TRUTH = {
    "images": [{"id": 1, "file_name": "a_1.jpg", "width": 640, "height": 480}],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [100, 100, 200, 100],
            "area": 20000,
            "iscrowd": 0,
        }
    ],
    "categories": [{"id": 0, "name": "billboard"}],
}


def _detection(bbox, score=0.9):
    return {"image_id": 1, "category_id": 0, "bbox": bbox, "score": score}


def test_perfect_boxes_score_full_ap_at_every_iou_threshold():
    aps = ap_per_iou_threshold(GROUND_TRUTH, [_detection([100, 100, 200, 100])])

    assert list(aps) == list(IOU_THRESHOLDS)
    assert all(ap == pytest.approx(1.0) for ap in aps.values())


def test_slightly_small_boxes_only_lose_ap_at_high_iou():
    aps = ap_per_iou_threshold(
        GROUND_TRUTH, [_detection([110, 105, 180, 90])]
    )  # IoU 0.81

    assert aps[0.5] == pytest.approx(1.0) and aps[0.8] == pytest.approx(1.0)
    assert aps[0.85] == 0.0 and aps[0.95] == 0.0


def test_boxes_by_image_converts_to_xyxy_and_filters_by_score():
    grouped = boxes_by_image(
        [_detection([0, 0, 10, 20], 0.9), _detection([5, 5, 1, 1], 0.1)], 0.5
    )

    assert grouped == {1: [[0, 0, 10, 20]]}


def test_low_iou_candidates_are_not_matched():
    pairs = match_to_ground_truth({1: [[0, 0, 10, 10]]}, {1: [[20, 20, 30, 30]]})

    assert pairs == []


def test_edge_bias_reports_boxes_shrunk_on_every_side():
    ground_truth = [100, 100, 300, 200]
    shrunk = [110, 105, 290, 195]  # 5% of width and height inside on each side

    bias = edge_bias(match_to_ground_truth({1: [ground_truth]}, {1: [shrunk]}))

    assert (bias.left, bias.right) == (pytest.approx(-0.05), pytest.approx(-0.05))
    assert (bias.top, bias.bottom) == (pytest.approx(-0.05), pytest.approx(-0.05))
    assert bias.width_ratio == pytest.approx(
        0.9
    ) and bias.height_ratio == pytest.approx(0.9)
    assert bias.matches == 1


def test_edge_bias_needs_matches():
    with pytest.raises(ValueError):
        edge_bias([])
