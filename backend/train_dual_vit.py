# -*- coding: utf-8 -*-
"""
train_dual_vit.py -- Training script for the Dual-ViT MRI forgery detector.

Usage
-----
python train_dual_vit.py \
    --data_dir   /path/to/processed \
    --output_dir ./models/dual_vit_checkpoint \
    --epochs     20 \
    --warmup_epochs 5 \
    --batch_size 8 \
    --lr         2e-5

Dataset layout expected under --data_dir:
    train/
        real/   *.png  *.jpg  ...
        fake/   *.png  *.jpg  ...
    val/
        real/
        fake/

Two-phase training strategy
----------------------------
Phase 1 (warmup, epochs 1..warmup_epochs):
    Both ViT encoders are FROZEN.  Only the fusion head is trained.
    This teaches the MLP to use CLS-token features before the encoders adapt,
    preventing the randomly-initialised head from corrupting pretrained weights.

Phase 2 (full fine-tuning, remaining epochs):
    All parameters are unfrozen.  A differential LR is applied:
        - Fusion head : lr (full learning rate)
        - Encoders    : lr / 10  (gentle nudge to preserve pretrained features)

Class imbalance handling
------------------------
Cross-entropy loss is weighted inversely to class frequency so the model
cannot cheat by predicting the majority class (fake) for every sample.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoImageProcessor, ViTModel  # type: ignore

# Allow importing model_service and ela from the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_service import DualViTClassifier  # noqa: E402
from ela import generate_ela_image           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DualViTDataset(Dataset):
    """
    Loads images from root/{split}/{real,fake}/*.ext
    and generates ELA on-the-fly for Branch 2.

    Label convention (matches model_service.py and the original notebook):
        fake = 0
        real = 1
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    LABEL_MAP = {"real": 1, "fake": 0}

    def __init__(self, root: str, split: str, processor) -> None:
        self.processor = processor
        self.samples: List[Tuple[Path, int]] = []

        split_dir = Path(root) / split
        for class_name, label in self.LABEL_MAP.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                logger.warning("Missing directory: %s  (skipping)", class_dir)
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((p, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {split_dir}. "
                "Expected layout: {split}/real/ and {split}/fake/"
            )

        counts = Counter(label for _, label in self.samples)
        logger.info(
            "%s split: %d samples  (fake=%d  real=%d)",
            split, len(self.samples), counts[0], counts[1],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning("Could not open %s: %s -- using blank image.", path, exc)
            image = Image.new("RGB", (224, 224), color=0)

        ela_img = generate_ela_image(image).convert("RGB")

        t_orig = self.processor(images=image,   return_tensors="pt")["pixel_values"].squeeze(0)
        t_ela  = self.processor(images=ela_img, return_tensors="pt")["pixel_values"].squeeze(0)

        return t_orig, t_ela, label


def make_weighted_sampler(dataset: DualViTDataset) -> WeightedRandomSampler:
    """
    Over-sample the minority class so every batch sees a balanced mix.
    This complements the weighted loss -- together they are very effective
    against the 6:1 fake:real imbalance in this dataset.
    """
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    total  = len(labels)
    weight_per_class = {cls: total / count for cls, count in counts.items()}
    sample_weights = [weight_per_class[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def set_encoders_frozen(model: DualViTClassifier, frozen: bool) -> None:
    """Freeze or unfreeze both ViT encoder branches."""
    for param in model.encoder_orig.parameters():
        param.requires_grad = not frozen
    for param in model.encoder_ela.parameters():
        param.requires_grad = not frozen
    state = "FROZEN" if frozen else "UNFROZEN"
    logger.info("Encoder branches: %s", state)


def build_optimizer(model: DualViTClassifier, lr: float, encoders_frozen: bool):
    """
    Differential learning rate:
        - Fusion head  : lr        (trains faster -- starts from random init)
        - Encoders     : lr / 10   (trains slowly -- preserves pretrained features)
    """
    head_params   = list(model.fusion_head.parameters())
    enc_params     = (
        list(model.encoder_orig.parameters()) +
        list(model.encoder_ela.parameters())
    )

    if encoders_frozen:
        param_groups = [{"params": head_params, "lr": lr}]
    else:
        param_groups = [
            {"params": head_params, "lr": lr},
            {"params": enc_params,  "lr": lr / 10},
        ]

    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


def compute_class_weights(dataset: DualViTDataset, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    n_classes = 2  # fake=0, real=1
    total = len(labels)
    weights = [total / (n_classes * counts[cls]) for cls in range(n_classes)]
    logger.info("Class weights: fake=%.3f  real=%.3f", weights[0], weights[1])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: DualViTClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer=None,
    device: torch.device = torch.device("cpu"),
    train: bool = True,
) -> Tuple[float, float]:
    """Single epoch of training or evaluation. Returns (loss, accuracy)."""
    model.train(train)
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for t_orig, t_ela, labels in loader:
            t_orig  = t_orig.to(device)
            t_ela   = t_ela.to(device)
            labels  = labels.to(device)

            logits = model(pixel_values_orig=t_orig, pixel_values_ela=t_ela)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)
    if device.type == "cpu":
        logger.warning(
            "No GPU detected -- training on CPU will be slow (~2-4 hrs for 20 epochs). "
            "Consider running on a machine with a GPU or in Google Colab."
        )

    backbone = "google/vit-base-patch16-224"
    logger.info("Loading ViT-Base backbone: %s", backbone)
    processor = AutoImageProcessor.from_pretrained(backbone)
    enc_orig  = ViTModel.from_pretrained(backbone)
    enc_ela   = ViTModel.from_pretrained(backbone)

    model = DualViTClassifier(enc_orig, enc_ela, hidden_dim=512, dropout=0.3)
    model.to(device)

    # --- Datasets ---
    train_ds = DualViTDataset(args.data_dir, "train", processor)
    val_ds   = DualViTDataset(args.data_dir, "val",   processor)

    sampler   = make_weighted_sampler(train_ds)
    train_dl  = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_dl    = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # --- Loss with class weighting ---
    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    optimizer    = None
    scheduler    = None

    for epoch in range(1, args.epochs + 1):

        # ---- Phase transition ----
        if epoch == 1:
            # Phase 1: freeze encoders, train only fusion head
            set_encoders_frozen(model, frozen=True)
            optimizer = build_optimizer(model, args.lr, encoders_frozen=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.warmup_epochs
            )
            logger.info("--- Phase 1: Fusion-head warmup (%d epochs) ---", args.warmup_epochs)

        elif epoch == args.warmup_epochs + 1:
            # Phase 2: unfreeze encoders with differential LR
            set_encoders_frozen(model, frozen=False)
            optimizer = build_optimizer(model, args.lr, encoders_frozen=False)
            remaining = args.epochs - args.warmup_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining
            )
            logger.info("--- Phase 2: Full fine-tuning (%d epochs) ---", remaining)

        # ---- Train ----
        train_loss, train_acc = run_epoch(
            model, train_dl, criterion, optimizer, device, train=True
        )
        scheduler.step()

        # ---- Validate ----
        val_loss, val_acc = run_epoch(
            model, val_dl, criterion, device=device, train=False
        )

        logger.info(
            "Epoch %2d/%d | train loss=%.4f acc=%.3f | val loss=%.4f acc=%.3f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        # ---- Save best checkpoint ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "dual_vit.pt"
            torch.save(model.state_dict(), str(ckpt_path))
            logger.info("  >>> Best checkpoint saved (val_acc=%.3f) -> %s", val_acc, ckpt_path)

    # Save the processor config so the server can load it automatically.
    processor.save_pretrained(str(output_dir))

    logger.info("=" * 60)
    logger.info("Training complete.  Best val_acc = %.3f", best_val_acc)
    logger.info("Checkpoint saved to: %s", output_dir)
    logger.info("")
    logger.info("Next step -- set this in backend/.env:")
    logger.info("  DUAL_MODEL_PATH=%s", output_dir.resolve())
    logger.info("Then restart the backend:  uvicorn server:app --reload --port 8000")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Dual-ViT MRI forgery detector")
    p.add_argument(
        "--data_dir", required=True,
        help="Root dataset directory containing train/ and val/ sub-folders"
    )
    p.add_argument(
        "--output_dir", default="models/dual_vit_checkpoint",
        help="Where to save the checkpoint and processor config"
    )
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--warmup_epochs", type=int,   default=5,
                   help="Epochs to train only the fusion head (encoders frozen)")
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--workers",       type=int,   default=0,
                   help="DataLoader num_workers (use 0 on Windows to avoid multiprocessing issues)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
