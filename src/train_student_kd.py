"""
Train the lightweight student model with knowledge distillation from a
pre-trained teacher (Sec. 7.3 + 7.4).

Prerequisite: run train_teacher.py first so checkpoints/teacher_{subset}.pt
exists.

Usage:
    python -m src.train_student_kd --subset FD001
    python -m src.train_student_kd --subset all

IMPORTANT -- single source of truth for KD training:
train_student() below is the *only* place the student training loop is
implemented. trade_off_sweep.py imports and reuses this exact function for
every compression variant it sweeps, instead of duplicating the loop. This
guarantees that a sweep variant configured identically to config.STUDENT_CFG
reproduces the same trained result as running this script directly (same
architecture -> same manual_seed -> same init -> same data order -> same
training procedure). Do not copy this loop elsewhere; extend this function
with new arguments instead.
"""

import argparse
import copy
import os

import numpy as np
import torch
from torch.optim import AdamW

from src import config
from src.config import StudentConfig
from src.data.dataset import get_dataloaders
from src.models.teacher_transformer import build_teacher
from src.models.student_model import build_student
from src.distillation.kd_loss import KnowledgeDistillationLoss
from src.utils.metrics import rmse, nasa_score
from src.utils.train_helpers import build_scheduler, augment_batch


def load_teacher(subset: str, device: str):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"teacher_{subset}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run `python -m src.train_teacher "
            f"--subset {subset}` first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Rebuild from the checkpoint's own saved config, not the current global
    # default -- window size / sensor count / architecture all vary by
    # subset (see config.get_teacher_config), so the global TEACHER_CFG
    # would silently mismatch FD002's actual trained shape.
    teacher_cfg = ckpt.get("config") or config.get_teacher_config(subset)
    teacher = build_teacher(teacher_cfg).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def evaluate_student(student, loader, device):
    student.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = student(X)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()
    return rmse(targets, preds), nasa_score(targets, preds)


def train_student(subset: str, device: str, student_cfg: StudentConfig = None,
                   verbose: bool = True, log_prefix: str = "student",
                   train_loader=None, val_loader=None, test_loader=None,
                   teacher_model=None, seed: int = None):
    """Train one student on one subset via knowledge distillation.

    This is the single reusable entry point for KD student training --
    both train_student_kd.py's CLI and trade_off_sweep.py call this
    function directly rather than each having their own copy of the loop.

    Args:
        subset: e.g. "FD001"
        device: "cpu" or "cuda"
        student_cfg: a StudentConfig instance; defaults to config.STUDENT_CFG
            if not provided. Pass a dataclasses.replace(...) variant to sweep
            architecture size while keeping everything else (seed, epochs,
            early-stopping patience, optimizer, KD loss weights) identical.
        verbose: print per-epoch progress if True
        log_prefix: label used in printed log lines, so sweep runs are
            distinguishable from the default training run in the console.

    Returns:
        (student_model, kd_loss_fn, test_rmse, test_score)
    """
    s_cfg = student_cfg if student_cfg is not None else config.get_student_config(subset)
    kd_cfg = config.KD_CFG
    # t_cfg is only used below to compute teacher_pooled_dim for the KD
    # feature projector -- if an already-built teacher_model was passed in
    # (e.g. an ensemble), prefer its actual config over a freshly-looked-up
    # default so the projector dimension always matches what's really there.
    t_cfg = (teacher_model.cfg if teacher_model is not None and hasattr(teacher_model, "cfg")
             else config.get_teacher_config(subset))

    # Reseed on every call so any student config, trained via this function,
    # gets the same weight initialization and data-shuffle order as any other
    # call with matching hyperparameters -- this is what makes results
    # reproducible/comparable across separate script invocations. Pass
    # `seed` to deliberately vary init/data order instead (ensembling).
    torch.manual_seed(seed if seed is not None else config.RANDOM_SEED)

    if train_loader is None or val_loader is None or test_loader is None:
        train_loader, val_loader, test_loader = get_dataloaders(subset, s_cfg.batch_size)

    teacher = teacher_model if teacher_model is not None else load_teacher(subset, device)
    student = build_student(s_cfg).to(device)

    student_pooled_dim = s_cfg.gru_hidden if s_cfg.use_gru_head else s_cfg.d_model
    teacher_pooled_dim = t_cfg.gru_hidden if getattr(t_cfg, "use_gru_head", False) else t_cfg.d_model
    kd_loss_fn = KnowledgeDistillationLoss(
        kd_cfg, student_dim=student_pooled_dim, teacher_dim=teacher_pooled_dim
    ).to(device)

    optimizer = AdamW(
        list(student.parameters()) + list(kd_loss_fn.parameters()),
        lr=s_cfg.lr, weight_decay=s_cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, s_cfg)

    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, s_cfg.epochs + 1):
        student.train()
        running = {"loss_task": 0.0, "loss_soft": 0.0, "loss_feature": 0.0, "loss_total": 0.0}
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X = augment_batch(X, s_cfg)
            with torch.no_grad():
                teacher_pred, teacher_feats = teacher(X, return_features=True)

            optimizer.zero_grad()
            student_pred, student_feats = student(X, return_features=True)
            loss, logs = kd_loss_fn(student_pred, teacher_pred, y, student_feats, teacher_feats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(kd_loss_fn.parameters()), config.GRAD_CLIP_NORM
            )
            optimizer.step()

            for k in running:
                running[k] += logs[k]
            n_batches += 1

        avg = {k: v / n_batches for k, v in running.items()}
        val_rmse, val_score = evaluate_student(student, val_loader, device)
        scheduler.step(val_rmse)

        if verbose:
            print(f"[{subset}][{log_prefix}] epoch {epoch:03d} "
                  f"task={avg['loss_task']:.3f} soft={avg['loss_soft']:.3f} "
                  f"feat={avg['loss_feature']:.3f} total={avg['loss_total']:.3f} "
                  f"| val_rmse={val_rmse:.3f} val_score={val_score:.1f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(student.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= s_cfg.patience:
                if verbose:
                    print(f"[{subset}][{log_prefix}] early stopping at epoch {epoch}")
                break

    student.load_state_dict(best_state)
    test_rmse, test_score = evaluate_student(student, test_loader, device)
    if verbose:
        print(f"[{subset}][{log_prefix}] TEST rmse={test_rmse:.3f} score={test_score:.1f}")

    return student, kd_loss_fn, test_rmse, test_score


def train_one_subset(subset: str, device: str, force: bool = False):
    """CLI entry point: trains the per-subset student (config.get_student_config(subset))
    and saves its checkpoint. This is what `comparison_{subset}.csv` reports."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"student_{subset}.pt")
    data_path = os.path.join(config.PROCESSED_DATA_DIR, f"{subset}.npz")
    teacher_path = os.path.join(config.CHECKPOINT_DIR, f"teacher_{subset}.pt")

    if not force and os.path.exists(ckpt_path):
        is_stale = (
            (os.path.exists(data_path) and os.path.getmtime(data_path) > os.path.getmtime(ckpt_path))
            or (os.path.exists(teacher_path) and os.path.getmtime(teacher_path) > os.path.getmtime(ckpt_path))
        )
        if not is_stale:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            print(f"[{subset}][student] checkpoint already exists and looks "
                  f"up to date -> {ckpt_path} (test_rmse={ckpt['test_rmse']:.3f}). "
                  f"Skipping training. Use --force to retrain anyway.")
            return ckpt["test_rmse"], ckpt["test_score"]
        else:
            print(f"[{subset}][student] checkpoint is older than the "
                  f"processed data or the teacher checkpoint -- retraining")

    s_cfg = config.get_student_config(subset)
    student, _, test_rmse, test_score = train_student(subset, device, student_cfg=s_cfg)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save({"state_dict": student.state_dict(), "config": s_cfg,
                "test_rmse": test_rmse, "test_score": test_score}, ckpt_path)
    print(f"[{subset}][student] saved checkpoint -> {ckpt_path}")
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