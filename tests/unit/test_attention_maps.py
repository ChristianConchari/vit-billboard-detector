import numpy as np
import pytest
import torch
from PIL import Image

from vit.eval.attention_maps import DetectionAttention, deformable_sampling, token_at
from vit.eval.visualization import (
    TITLE_BAR_HEIGHT,
    draw_encoder_attention,
    draw_sampling_points,
)


class FakeDeformableAttention(torch.nn.Module):
    n_heads, n_levels, n_points = 1, 1, 2

    def __init__(self, offsets: list[float]):
        super().__init__()
        self.sampling_offsets = torch.nn.Linear(1, 4)
        self.attention_weights = torch.nn.Linear(1, 2)
        torch.nn.init.zeros_(self.sampling_offsets.weight)
        self.sampling_offsets.bias.data = torch.tensor(offsets)
        torch.nn.init.zeros_(self.attention_weights.weight)
        torch.nn.init.zeros_(self.attention_weights.bias)


def test_token_at_maps_normalized_point_to_flat_grid_index():
    assert token_at((0.0, 0.0), 20, 20) == 0
    assert token_at((0.99, 0.0), 20, 20) == 19
    assert token_at((0.5, 0.5), 20, 20) == 10 * 20 + 10
    assert token_at((1.0, 1.0), 20, 20) == 399


def test_deformable_sampling_offsets_points_relative_to_the_reference_box():
    module = FakeDeformableAttention(offsets=[0.0, 0.0, 2.0, -2.0])
    kwargs = {
        "hidden_states": torch.zeros(1, 1, 1),
        "position_embeddings": torch.zeros(1, 1, 1),
        "reference_points": torch.tensor([[[[0.5, 0.5, 0.2, 0.4]]]]),
    }

    locations, weights = deformable_sampling(module, kwargs)

    points = locations.reshape(-1, 2)
    assert points[0].tolist() == pytest.approx([0.5, 0.5])
    assert points[1].tolist() == pytest.approx(
        [0.5 + 2.0 / 2 * 0.2 * 0.5, 0.5 - 2.0 / 2 * 0.4 * 0.5]
    )
    assert weights.reshape(-1).tolist() == pytest.approx([0.5, 0.5])


def _detection() -> DetectionAttention:
    return DetectionAttention(
        box_xyxy=(10, 10, 60, 40),
        score=0.3,
        encoder_attention=np.random.default_rng(0).random((4, 8)),
        sampling_points_xy=np.array([[20.0, 20.0], [500.0, 500.0]]),
        sampling_weights=np.array([0.7, 0.3]),
    )


def test_attention_panels_keep_image_size_plus_title_bar():
    image = Image.new("RGB", (200, 100))

    for draw in (draw_encoder_attention, draw_sampling_points):
        assert draw(image, _detection(), "t").size == (200, 100 + TITLE_BAR_HEIGHT)
