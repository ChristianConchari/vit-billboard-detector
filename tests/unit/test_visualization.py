from PIL import Image

from vit.eval.visualization import TITLE_BAR_HEIGHT, crop_bottom, draw_detections, side_by_side
from vit.inference.detection import BoxProposal


def test_draw_detections_adds_title_bar_above_the_image():
    panel = draw_detections(
        Image.new("RGB", (200, 100)),
        [[10, 10, 50, 50]],
        [BoxProposal("billboard", 0.9, (10, 10, 60, 60))],
        "t",
    )

    assert panel.size == (200, 100 + TITLE_BAR_HEIGHT)


def test_crop_bottom_removes_only_the_bottom_band():
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    image.paste((255, 255, 255), (0, 90, 200, 100))

    cropped = crop_bottom(image, 10)

    assert cropped.size == (200, 90)
    assert cropped.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_side_by_side_scales_panels_to_max_width():
    combined = side_by_side([Image.new("RGB", (1920, 1080))] * 2, max_panel_width=960)

    assert combined.size == (1920, 540)
