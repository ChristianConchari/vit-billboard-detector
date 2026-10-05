import pytest
import torch

from vit.models.rtdetr import freeze_backbone
from vit.train.trainer import build_optimizer

TRAIN_CFG = {
    "learning_rate": 1e-4,
    "backbone_learning_rate": 1e-5,
    "weight_decay": 1e-4,
}


class TinyDetector(torch.nn.Module):
    """Mimics RT-DETR's layout: model.model.backbone plus the rest of the network."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(4, 4)
        self.model.decoder = torch.nn.Linear(4, 2)


def _group_sizes_and_lrs(optimizer):
    return [(len(group["params"]), group["lr"]) for group in optimizer.param_groups]


def test_unfrozen_backbone_gets_its_own_lower_learning_rate():
    optimizer = build_optimizer(TinyDetector(), TRAIN_CFG)

    assert _group_sizes_and_lrs(optimizer) == [(2, 1e-4), (2, 1e-5)]


def test_frozen_backbone_is_left_out_of_the_optimizer():
    model = TinyDetector()
    freeze_backbone(model)

    optimizer = build_optimizer(model, TRAIN_CFG)

    assert _group_sizes_and_lrs(optimizer) == [(2, 1e-4)]
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
