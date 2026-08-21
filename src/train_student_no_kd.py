"""
Control experiment: trains the exact same student ARCHITECTURE as
config.STUDENT_CFG, but with plain supervised RMSE loss on ground-truth
labels only -- no teacher, no soft-target loss, no feature-alignment loss.

This isolates what knowledge distillation itself contributes. Without this
comparison, "Student (Distilled) beats Teacher" is ambiguous: it could mean
KD is working well, or it could just mean a small Transformer generalizes
well on its own regardless of distillation. Comparing this checkpoint
against the KD-trained student in evaluate.py answers that directly.

Usage:
    python -m src.train_student_no_kd --subset FD001
    python -m src.train_student_no_kd --subset all
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src import config
from src.data.dataset import get_dataloaders
from src.models.student_model import build_student
from src.utils.metrics import rmse, nasa_score
from src.utils.train_helpers import build_scheduler, augment_batch


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()
    return rmse(targets, preds), nasa_score(targets, preds)


def train_one_subset(subset: str, device: str, force: bool = False):
    cfg = config.get_student_config(subset)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"student_no_kd_{subset}.pt")
    data_path = os.path.join(config.PROCESSED_DATA_DIR, f"{subset}.npz")

    if not force and os.path.exists(ckpt_path):
        is_stale = (
            os.path.exists(data_path)
            and os.path.getmtime(data_path) > os.path.getmtime(ckpt_path)
        )
        if not is_stale:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            print(f"[{subset}][student-no-kd] checkpoint already exists and "
                  f"looks up to date -> {ckpt_path} (test_rmse={ckpt['test_rmse']:.3f}). "
                  f"Skipping training. Use --force to retrain anyway.")
            return ckpt["test_rmse"], ckpt["test_score"]
        else:
            print(f"[{subset}][student-no-kd] checkpoint is older than the "
                  f"processed data -- retraining")

    # Same seed as the KD student and teacher, so architecture is the ONLY
    # thing that differs from train_student_kd.py's result -- not init luck.
    torch.manual_seed(config.RANDOM_SEED)

    train_loader, val_loader, test_loader = get_dataloaders(subset, cfg.batch_size)
    model = build_student(cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X = augment_batch(X, cfg)
            optimizer.zero_grad()
            pred = model(X)
            loss = torch.sqrt(criterion(pred, y) + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        train_rmse = running_loss / len(train_loader.dataset)
        val_rmse, val_score = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        print(f"[{subset}][student-no-kd] epoch {epoch:03d} "
              f"train_rmse={train_rmse:.3f} val_rmse={val_rmse:.3f} "
              f"val_score={val_score:.1f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[{subset}][student-no-kd] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_rmse, test_score = evaluate(model, test_loader, device)
    print(f"[{subset}][student-no-kd] TEST rmse={test_rmse:.3f} score={test_score:.1f}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg, "test_rmse": test_rmse,
                "test_score": test_score}, ckpt_path)
    print(f"[{subset}][student-no-kd] saved checkpoint -> {ckpt_path}")
    return test_rmse, test_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true",
                         help="Retrain even if an up-to-date checkpoint already exists.")
    args = parser.parse_args()

    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    for s in subsets:
        train_one_subset(s, args.device, force=args.force)


if __name__ == "__main__":
    main()