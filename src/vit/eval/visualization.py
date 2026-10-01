"""Draw ground truth and predicted boxes for qualitative comparison of detectors."""
from PIL import Image, ImageDraw, ImageFont

from vit.inference.detection import BoxProposal

GROUND_TRUTH_COLOR = (46, 204, 113)
PREDICTION_COLOR = (231, 76, 60)
TITLE_BAR_HEIGHT = 48


def draw_detections(
    image: Image.Image,
    ground_truth_xywh: list[list[float]],
    proposals: list[BoxProposal],
    title: str,
) -> Image.Image:
    """Return a copy of `image` with ground truth (green), predictions (red) and a title bar."""
    line_width = max(2, image.width // 400)
    font = ImageFont.load_default(size=max(16, image.width // 60))

    canvas = Image.new("RGB", (image.width, image.height + TITLE_BAR_HEIGHT), (255, 255, 255))
    canvas.paste(image, (0, TITLE_BAR_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black", font=font)

    for x, y, w, h in ground_truth_xywh:
        top = y + TITLE_BAR_HEIGHT
        draw.rectangle([x, top, x + w, top + h], outline=GROUND_TRUTH_COLOR, width=line_width)

    for proposal in proposals:
        x_min, y_min, x_max, y_max = proposal.box_xyxy
        top, bottom = y_min + TITLE_BAR_HEIGHT, y_max + TITLE_BAR_HEIGHT
        draw.rectangle([x_min, top, x_max, bottom], outline=PREDICTION_COLOR, width=line_width)
        draw.text((x_min + 4, top + 2), f"{proposal.score:.2f}", fill=PREDICTION_COLOR, font=font)

    return canvas


def crop_bottom(image: Image.Image, height: int) -> Image.Image:
    return image.crop((0, 0, image.width, image.height - height))


def side_by_side(panels: list[Image.Image], max_panel_width: int = 960) -> Image.Image:
    scale = min(1.0, max_panel_width / panels[0].width)
    resized = [p.resize((round(p.width * scale), round(p.height * scale))) for p in panels]
    combined = Image.new("RGB", (sum(p.width for p in resized), resized[0].height), (255, 255, 255))
    x_offset = 0
    for panel in resized:
        combined.paste(panel, (x_offset, 0))
        x_offset += panel.width
    return combined
