"""Draw boxes and attention maps for qualitative comparison of detectors."""

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from vit.eval.attention_maps import DetectionAttention
from vit.inference.detection import BoxProposal

GROUND_TRUTH_COLOR = (46, 204, 113)
PREDICTION_COLOR = (231, 76, 60)
TITLE_BAR_HEIGHT = 48
ATTENTION_COLORMAP = colormaps["inferno"]


def draw_detections(
    image: Image.Image,
    ground_truth_xywh: list[list[float]],
    proposals: list[BoxProposal],
    title: str,
) -> Image.Image:
    """Copy of `image` with ground truth (green), predictions (red) and a title bar."""
    canvas = draw_predictions(image, proposals)
    draw = ImageDraw.Draw(canvas)
    for x, y, w, h in ground_truth_xywh:
        draw.rectangle(
            [x, y, x + w, y + h], outline=GROUND_TRUTH_COLOR, width=_line_width(image)
        )
    return _with_title(canvas, title)


def draw_predictions(image: Image.Image, proposals: list[BoxProposal]) -> Image.Image:
    """Return a copy of `image` with each predicted box and its score."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for proposal in proposals:
        _draw_box(draw, image, proposal.box_xyxy, f"{proposal.score:.2f}")
    return canvas


def draw_encoder_attention(
    image: Image.Image, detection: DetectionAttention, title: str
) -> Image.Image:
    """Overlay the encoder self-attention of the token under the detection's center."""
    attention = detection.encoder_attention / detection.encoder_attention.max()
    heatmap = Image.fromarray(np.uint8(ATTENTION_COLORMAP(attention)[..., :3] * 255))
    heatmap = heatmap.resize(image.size, Image.Resampling.BILINEAR)
    canvas = Image.blend(image, heatmap, alpha=0.55)
    _draw_box(
        ImageDraw.Draw(canvas), image, detection.box_xyxy, f"{detection.score:.2f}"
    )
    return _with_title(canvas, title)


def draw_sampling_points(
    image: Image.Image, detection: DetectionAttention, title: str
) -> Image.Image:
    """Draw the decoder query's deformable sampling points, scaled by weight."""
    canvas = Image.blend(image, Image.new("RGB", image.size), alpha=0.35)
    draw = ImageDraw.Draw(canvas)
    weights = detection.sampling_weights / detection.sampling_weights.max()
    max_radius = image.width / 80

    for (x, y), weight in sorted(
        zip(detection.sampling_points_xy, weights, strict=True), key=lambda p: p[1]
    ):
        if not (0 <= x < image.width and 0 <= y < image.height):
            continue
        radius = max(2.0, max_radius * float(np.sqrt(weight)))
        color = tuple(int(c * 255) for c in ATTENTION_COLORMAP(0.3 + 0.7 * weight)[:3])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    _draw_box(draw, image, detection.box_xyxy, f"{detection.score:.2f}")
    return _with_title(canvas, title)


def crop_bottom(image: Image.Image, height: int) -> Image.Image:
    return image.crop((0, 0, image.width, image.height - height))


def side_by_side(panels: list[Image.Image], max_panel_width: int = 960) -> Image.Image:
    scale = min(1.0, max_panel_width / panels[0].width)
    resized = [
        p.resize((round(p.width * scale), round(p.height * scale))) for p in panels
    ]
    combined = Image.new(
        "RGB", (sum(p.width for p in resized), resized[0].height), (255, 255, 255)
    )
    x_offset = 0
    for panel in resized:
        combined.paste(panel, (x_offset, 0))
        x_offset += panel.width
    return combined


def _with_title(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new(
        "RGB", (image.width, image.height + TITLE_BAR_HEIGHT), (255, 255, 255)
    )
    canvas.paste(image, (0, TITLE_BAR_HEIGHT))
    ImageDraw.Draw(canvas).text((10, 8), title, fill="black", font=_font(image))
    return canvas


def _draw_box(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    box_xyxy: tuple[float, float, float, float],
    label: str,
) -> None:
    draw.rectangle(box_xyxy, outline=PREDICTION_COLOR, width=_line_width(image))
    draw.text(
        (box_xyxy[0] + 4, box_xyxy[1] + 2),
        label,
        fill=PREDICTION_COLOR,
        font=_font(image),
    )


def _line_width(image: Image.Image) -> int:
    return max(2, image.width // 400)


def _font(image: Image.Image) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=max(16, image.width // 60))
