"""Fine-tuning loop for RT-DETR with per-epoch validation mAP and MLflow tracking."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from vit.data.coco_detection import CocoDetectionDataset, DetectionCollator
from vit.eval.coco_evaluation import evaluate_detector
from vit.inference.rtdetr_detector import RtDetrDetector
from vit.models.rtdetr import freeze_backbone, init_classification_bias, load_rtdetr
from vit.transforms.augmentations import build_train_augmentations
from vit.utils.logging import get_logger
from vit.utils.tracking import prefixed, require_active_run

logger = get_logger(__name__, log_file="training.log")


def train_rtdetr(config: dict[str, Any], run_name: str | None = None) -> Path:
    """Fine-tune RT-DETR as described by `config` and return the best checkpoint dir.

    Per-epoch metrics go to the active MLflow run, which the caller owns.
    """
    require_active_run()
    data_cfg, train_cfg = config["data"], config["training"]
    seed_everything(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, image_processor = load_rtdetr(
        config["model"]["pretrained_checkpoint"], config["model"]["label_names"]
    )
    init_classification_bias(model)
    if train_cfg["freeze_backbone"]:
        freeze_backbone(model)
    model.to(device)

    train_loader = DataLoader(
        CocoDetectionDataset(
            data_cfg["train_annotations"],
            data_cfg["image_dir"],
            config["model"]["label_names"],
            augmentations=build_train_augmentations(config["augmentation"]),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=DetectionCollator(image_processor),
    )
    val_ground_truth = json.loads(Path(data_cfg["val_annotations"]).read_text())
    val_detector = RtDetrDetector(
        model, image_processor, device, score_threshold=config["evaluation"]["score_threshold"]
    )

    optimizer = build_optimizer(model, train_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] * len(train_loader)
    )

    run_name = run_name or new_run_name()
    best_dir = Path(train_cfg["checkpoint_dir"]) / run_name / "best"
    best_map = -1.0

    mlflow.log_params({"train.images": len(train_loader.dataset), "train.device": str(device)})
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, train_cfg["max_grad_norm"]
        )
        val_metrics = evaluate_detector(val_detector, val_ground_truth, data_cfg["image_dir"])
        mlflow.log_metrics({"train/loss": train_loss, **prefixed(val_metrics, "val")}, step=epoch)
        logger.info(
            "epoch %d/%d | train_loss %.4f | val mAP %.4f | val AP50 %.4f",
            epoch,
            train_cfg["epochs"],
            train_loss,
            val_metrics["mAP"],
            val_metrics["AP50"],
        )

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            model.save_pretrained(best_dir)
            image_processor.save_pretrained(best_dir)
            mlflow.log_metrics({"val/best_mAP": best_map, "val/best_epoch": epoch}, step=epoch)

    mlflow.log_param("train.best_checkpoint", str(best_dir))
    logger.info("Best val mAP %.4f, checkpoint saved to %s", best_map, best_dir)
    return best_dir


def build_optimizer(model: PreTrainedModel, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with a separate, lower learning rate for the (unfrozen) pretrained backbone.

    Updating the backbone at the head's learning rate tends to wreck its
    pretrained features, so it gets `backbone_learning_rate` instead.
    """
    backbone_ids = {id(p) for p in model.model.backbone.parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {
            "params": [p for p in trainable if id(p) not in backbone_ids],
            "lr": train_cfg["learning_rate"],
        },
        {
            "params": [p for p in trainable if id(p) in backbone_ids],
            "lr": train_cfg["backbone_learning_rate"],
        },
    ]
    return torch.optim.AdamW(
        [group for group in groups if group["params"]], weight_decay=train_cfg["weight_decay"]
    )


def train_one_epoch(
    model: PreTrainedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = [{k: v.to(device) for k, v in target.items()} for target in batch["labels"]]
        loss = model(pixel_values=batch["pixel_values"].to(device), labels=labels).loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def new_run_name() -> str:
    return f"rtdetr-{datetime.now():%Y%m%d-%H%M%S}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
