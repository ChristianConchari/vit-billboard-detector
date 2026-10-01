import pytest
from PIL import Image

from vit.eval.coco_evaluation import coco_metrics, evaluate_detector
from vit.inference.detection import BoxProposal


def _ground_truth(categories=None):
    return {
        "images": [{"id": 1, "file_name": "a_1.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0}
        ],
        "categories": categories or [{"id": 0, "name": "billboard"}],
    }


class FixedDetector:
    def __init__(self, proposals):
        self.proposals = proposals

    def predict(self, image):
        return self.proposals


@pytest.fixture
def image_dir(tmp_path):
    Image.new("RGB", (640, 480)).save(tmp_path / "a_1.jpg")
    return tmp_path


def test_perfect_detection_scores_full_map(image_dir):
    detector = FixedDetector([BoxProposal("billboard", 0.9, (100, 100, 300, 250))])

    metrics = evaluate_detector(detector, _ground_truth(), image_dir)

    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["AP50"] == pytest.approx(1.0)


def test_proposal_label_text_is_ignored(image_dir):
    detector = FixedDetector([BoxProposal("advertising sign", 0.9, (100, 100, 300, 250))])

    assert evaluate_detector(detector, _ground_truth(), image_dir)["mAP"] == pytest.approx(1.0)


def test_misplaced_detection_scores_zero(image_dir):
    detector = FixedDetector([BoxProposal("billboard", 0.9, (400, 300, 600, 450))])

    assert evaluate_detector(detector, _ground_truth(), image_dir)["mAP"] == pytest.approx(0.0)


def test_no_detections_scores_zero():
    assert coco_metrics(_ground_truth(), [])["mAP"] == 0.0


def test_multi_category_ground_truth_is_rejected(image_dir):
    categories = [{"id": 0, "name": "billboard"}, {"id": 1, "name": "poster"}]

    with pytest.raises(ValueError):
        evaluate_detector(FixedDetector([]), _ground_truth(categories), image_dir)
