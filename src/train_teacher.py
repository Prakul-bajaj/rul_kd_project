"""
Train the high-capacity Transformer teacher model (Sec. 7.2).

Usage:
    python -m src.train_teacher --subset FD001
    python -m src.train_teacher --subset FD001 --device cuda

IMPORTANT -- single source of truth for teacher training:
train_teacher_model() below is the *only* place the teacher training loop
is implemented. src/teacher_arch_sweep.py and the CV harness both call it
directly with a custom TeacherConfig instead of duplicating the loop, the
same pattern train_student_kd.py's train_student() already establishes.
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src import config
from src.config import TeacherConfig
from src.data.dataset import get_dataloaders
from src.models.teacher_transformer import build_teacher
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


def train_teacher_model(subset: str, device: str, teacher_cfg: TeacherConfig = None,
                         verbose: bool = True, log_prefix: str = "teacher",
                         train_loader=None, val_loader=None, test_loader=None,
                         seed: int = None):
    """Train one teacher on one subset. Reusable entry point for both the
    CLI (train_one_subset) and src/teacher_arch_sweep.py.

    Args:
        teacher_cfg: defaults to config.TEACHER_CFG if not given. Pass a
            dataclasses.replace(...) variant to sweep architecture while
            keeping the seed/optimizer/schedule identical.
        train_loader/val_loader/test_loader: pass pre-built loaders (e.g.
            from a CV fold) to override the default get_dataloaders(subset)
            split.

    Returns: (model, test_rmse, test_score)
    """
    cfg = teacher_cfg if teacher_cfg is not None else config.TEACHER_CFG
    torch.manual_seed(seed if seed is not None else config.RANDOM_SEED)

    if train_loader is None or val_loader is None or test_loader is None:
        train_loader, val_loader, test_loader = get_dataloaders(subset, cfg.batch_size)
    model = build_teacher(cfg).to(device)

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

        if verbose:
            print(f"[{subset}][{log_prefix}] epoch {epoch:03d} "
                  f"train_rmse={train_rmse:.3f} val_rmse={val_rmse:.3f} "
                  f"val_score={val_score:.1f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                if verbose:
                    print(f"[{subset}][{log_prefix}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_rmse, test_score = evaluate(model, test_loader, device)
    if verbose:
        print(f"[{subset}][{log_prefix}] TEST rmse={test_rmse:.3f} score={test_score:.1f}")

    return model, test_rmse, test_score


def train_one_subset(subset: str, device: str, force: bool = False):
    """CLI entry point: trains the *default* teacher (config.TEACHER_CFG)
    and saves its checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"teacher_{subset}.pt")
    data_path = os.path.join(config.PROCESSED_DATA_DIR, f"{subset}.npz")

    if not force and os.path.exists(ckpt_path):
        is_stale = (
            os.path.exists(data_path)
            and os.path.getmtime(data_path) > os.path.getmtime(ckpt_path)
        )
        if not is_stale:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            print(f"[{subset}][teacher] checkpoint already exists and looks "
                  f"up to date -> {ckpt_path} (test_rmse={ckpt['test_rmse']:.3f}). "
                  f"Skipping training. Use --force to retrain anyway.")
            return ckpt["test_rmse"], ckpt["test_score"]
        else:
            print(f"[{subset}][teacher] checkpoint is older than the "
                  f"processed data -- retraining")

    model, test_rmse, test_score = train_teacher_model(subset, device)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config.TEACHER_CFG,
                "test_rmse": test_rmse, "test_score": test_score}, ckpt_path)
    print(f"[{subset}][teacher] saved checkpoint -> {ckpt_path}")
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
