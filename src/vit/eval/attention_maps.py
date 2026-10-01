"""Extract RT-DETR attention for each detection, for qualitative interpretation.

Two views per detection:
- Encoder (AIFI): dense multi-head self-attention over the stride-32 feature
  grid, read from the token under the detected box center, like a ViT
  attention map.
- Decoder: the deformable cross-attention of the detection's query in the last
  decoder layer, i.e. the image points it samples and how much weight each
  gets. Sampling locations aren't part of the model output, so they are
  recomputed from the layer inputs with the layer's own projections.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessor, PreTrainedModel


@dataclass
class DetectionAttention:
    box_xyxy: tuple[float, float, float, float]
    score: float
    encoder_attention: np.ndarray  # (grid_height, grid_width), sums to 1
    sampling_points_xy: np.ndarray  # (num_points, 2), pixels
    sampling_weights: np.ndarray  # (num_points,)


def deformable_sampling(
    attention_module: torch.nn.Module, layer_kwargs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute sampling locations and weights of RT-DETR's deformable attention.

    Mirrors RTDetrMultiscaleDeformableAttention.forward for 4-d (box) reference
    points. Returns locations (batch, queries, heads, levels, points, 2) in
    normalized image coordinates and weights (batch, queries, heads, levels, points).
    """
    queries = layer_kwargs["hidden_states"] + layer_kwargs["position_embeddings"]
    reference_boxes = layer_kwargs["reference_points"]
    batch_size, num_queries, _ = queries.shape
    heads, levels, points = (
        attention_module.n_heads,
        attention_module.n_levels,
        attention_module.n_points,
    )

    offsets = attention_module.sampling_offsets(queries).view(
        batch_size, num_queries, heads, levels, points, 2
    )
    weights = torch.softmax(
        attention_module.attention_weights(queries).view(batch_size, num_queries, heads, -1), dim=-1
    ).view(batch_size, num_queries, heads, levels, points)

    centers = reference_boxes[:, :, None, :, None, :2]
    sizes = reference_boxes[:, :, None, :, None, 2:]
    locations = centers + offsets / points * sizes * 0.5
    return locations, weights


def token_at(center_xy: tuple[float, float], grid_height: int, grid_width: int) -> int:
    """Index of the flattened grid token containing a normalized (x, y) point."""
    column = min(int(center_xy[0] * grid_width), grid_width - 1)
    row = min(int(center_xy[1] * grid_height), grid_height - 1)
    return row * grid_width + column


@torch.no_grad()
def explain_detections(
    model: PreTrainedModel,
    image_processor: BaseImageProcessor,
    image: Image.Image,
    device: torch.device,
    score_threshold: float,
    max_detections: int = 3,
) -> list[DetectionAttention]:
    """Run RT-DETR once and return attention views for its top detections.

    The model must be loaded with attn_implementation="eager" so the encoder
    returns its attention weights.
    """
    model.eval()
    cross_attention = model.model.decoder.layers[-1].encoder_attn
    captured: dict[str, Any] = {}
    hook = cross_attention.register_forward_hook(
        lambda module, args, kwargs, output: captured.update(kwargs), with_kwargs=True
    )
    try:
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs, output_attentions=True)
    finally:
        hook.remove()

    locations, weights = deformable_sampling(cross_attention, captured)
    encoder_attention = outputs.encoder_attentions[-1][0].mean(dim=0)
    grid_height, grid_width = captured["spatial_shapes_list"][-1]

    scores = outputs.logits[0].sigmoid().max(dim=-1).values
    top = scores.topk(max_detections)
    image_size = np.array([image.width, image.height])

    detections = []
    for score, query in zip(top.values.tolist(), top.indices.tolist()):
        if score < score_threshold:
            break
        cx, cy, w, h = outputs.pred_boxes[0, query].tolist()
        token = token_at((cx, cy), grid_height, grid_width)
        detections.append(
            DetectionAttention(
                box_xyxy=(
                    (cx - w / 2) * image.width,
                    (cy - h / 2) * image.height,
                    (cx + w / 2) * image.width,
                    (cy + h / 2) * image.height,
                ),
                score=score,
                encoder_attention=encoder_attention[token]
                .reshape(grid_height, grid_width)
                .cpu()
                .numpy(),
                sampling_points_xy=locations[0, query].reshape(-1, 2).cpu().numpy() * image_size,
                sampling_weights=weights[0, query].reshape(-1).cpu().numpy(),
            )
        )
    return detections
