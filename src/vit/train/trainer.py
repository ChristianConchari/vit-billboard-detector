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

logger = get_logger(__name__, log_file="training.log")


def train_rtdetr(config: dict[str, Any]) -> Path:
    """Fine-tune RT-DETR as described by `config` and return the best checkpoint dir."""
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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] * len(train_loader)
    )

    run_name = f"rtdetr-{datetime.now():%Y%m%d-%H%M%S}"
    best_dir = Path(train_cfg["checkpoint_dir"]) / run_name / "best"
    best_map = -1.0

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten(config))
        mlflow.log_params({"train_images": len(train_loader.dataset), "device": str(device)})

        for epoch in range(1, train_cfg["epochs"] + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, train_cfg["max_grad_norm"]
            )
            val_metrics = evaluate_detector(val_detector, val_ground_truth, data_cfg["image_dir"])
            mlflow.log_metrics(
                {"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}},
                step=epoch,
            )
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
                mlflow.log_metrics({"best_val_mAP": best_map, "best_epoch": epoch}, step=epoch)

        mlflow.log_param("best_checkpoint", str(best_dir))

    logger.info("Best val mAP %.4f, checkpoint saved to %s", best_map, best_dir)
    return best_dir


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config sections into dotted keys for MLflow params."""
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat
