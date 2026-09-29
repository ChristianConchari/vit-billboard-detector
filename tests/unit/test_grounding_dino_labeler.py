from vit.labeling.grounding_dino_labeler import BoxProposal, deduplicate_proposals


def test_deduplicate_keeps_highest_score_among_overlapping_boxes():
    proposals = [
        BoxProposal(label="billboard", score=0.55, box_xyxy=(100, 100, 300, 300)),
        BoxProposal(label="advertising sign", score=0.90, box_xyxy=(102, 101, 301, 299)),
        BoxProposal(label="ooh ad", score=0.40, box_xyxy=(105, 103, 298, 302)),
    ]

    kept = deduplicate_proposals(proposals, iou_threshold=0.5)

    assert len(kept) == 1
    assert kept[0].score == 0.90


def test_deduplicate_keeps_non_overlapping_boxes_separate():
    proposals = [
        BoxProposal(label="billboard", score=0.8, box_xyxy=(0, 0, 100, 100)),
        BoxProposal(label="billboard", score=0.7, box_xyxy=(500, 500, 600, 600)),
    ]

    kept = deduplicate_proposals(proposals, iou_threshold=0.5)

    assert len(kept) == 2


def test_deduplicate_empty_list_returns_empty_list():
    assert deduplicate_proposals([], iou_threshold=0.5) == []
