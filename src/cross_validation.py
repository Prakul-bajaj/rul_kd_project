"""
K-fold-by-unit cross-validation harness (project review, Sec. "Data" /
"Teacher overfitting"): a single fixed 80/20 train/val split means the
"best epoch" chosen during training -- and the final reported RMSE -- are
both partly a function of which handful of engines happened to land in
that one validation slice. This reuses train_teacher_model() /
train_student() / train_student_no_kd's loop unchanged (same pattern as
trade_off_sweep.py and teacher_arch_sweep.py: no duplicated training
loop), just re-folding the training pool by unit_id for each run and
reporting mean +/- std across folds instead of one number.

The held-out official C-MAPSS test set is NOT folded -- every fold trains
on a different train/val split of the *training* pool but is evaluated on
the same fixed test set, so the spread across folds isolates "how much
does the train/val split itself affect the model" from "how good is the
model" (test conditions never change).

Usage:
    python -m src.cross_validation --subset FD001 --model teacher
    python -m src.cross_validation --subset FD002 --model student --folds 5
    python -m src.cross_validation --subset all --model student_no_kd
"""

import argparse
import dataclasses
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import CMAPSSDataset, load_split
from src.train_teacher import train_teacher_model
from src.train_student_kd import train_student, load_teacher
from src.models.student_model import build_student
from src.utils.metrics import rmse, nasa_score
from src.utils.run_record import save_config_snapshot


def unit_kfold_splits(unit_ids: np.ndarray, n_folds: int, seed: int):
    """Yield (train_mask, val_mask) for each fold, splitting by unique
    unit_id (never by window) so no engine leaks across a fold's
    train/val boundary."""
    rng = np.random.default_rng(seed)
    unique_units = np.unique(unit_ids)
    rng.shuffle(unique_units)
    folds = np.array_split(unique_units, n_folds)
    for k in range(n_folds):
        val_units = set(folds[k].tolist())
        val_mask = np.array([u in val_units for u in unit_ids])
        yield ~val_mask, val_mask


def _make_loader(X, y, batch_size, shuffle):
    ds = CMAPSSDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _fixed_test_loader(subset: str, batch_size: int):
    data = load_split(subset)
    return _make_loader(data["X_test"], data["y_test"], batch_size, shuffle=False)


def run_teacher_cv(subset: str, device: str, n_folds: int = None, teacher_cfg=None):
    n_folds = n_folds or config.CV_N_FOLDS
    cfg = teacher_cfg if teacher_cfg is not None else config.TEACHER_CFG
    data = load_split(subset)
    X_all, y_all, units_all = data["X_trainval"], data["y_trainval"], data["units_trainval"]
    test_loader = _fixed_test_loader(subset, cfg.batch_size)

    rows = []
    for k, (train_mask, val_mask) in enumerate(unit_kfold_splits(units_all, n_folds, config.RANDOM_SEED)):
        train_loader = _make_loader(X_all[train_mask], y_all[train_mask], cfg.batch_size, shuffle=True)
        val_loader = _make_loader(X_all[val_mask], y_all[val_mask], cfg.batch_size, shuffle=False)
        _, test_rmse, test_score = train_teacher_model(
            subset, device, teacher_cfg=cfg, verbose=False, log_prefix=f"cv-teacher-fold{k}",
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        )
        print(f"[{subset}][cv-teacher] fold {k+1}/{n_folds}: test_rmse={test_rmse:.3f} test_score={test_score:.1f}")
        rows.append({"fold": k, "test_rmse": test_rmse, "test_score": test_score})
    return pd.DataFrame(rows)


def run_student_cv(subset: str, device: str, n_folds: int = None, student_cfg=None, use_kd: bool = True):
    n_folds = n_folds or config.CV_N_FOLDS
    cfg = student_cfg if student_cfg is not None else config.STUDENT_CFG
    data = load_split(subset)
    X_all, y_all, units_all = data["X_trainval"], data["y_trainval"], data["units_trainval"]
    test_loader = _fixed_test_loader(subset, cfg.batch_size)

    teacher = load_teacher(subset, device) if use_kd else None

    rows = []
    for k, (train_mask, val_mask) in enumerate(unit_kfold_splits(units_all, n_folds, config.RANDOM_SEED)):
        train_loader = _make_loader(X_all[train_mask], y_all[train_mask], cfg.batch_size, shuffle=True)
        val_loader = _make_loader(X_all[val_mask], y_all[val_mask], cfg.batch_size, shuffle=False)
        if use_kd:
            _, _, test_rmse, test_score = train_student(
                subset, device, student_cfg=cfg, verbose=False, log_prefix=f"cv-student-fold{k}",
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                teacher_model=teacher,
            )
        else:
            test_rmse, test_score = _train_no_kd_fold(subset, device, cfg, train_loader, val_loader, test_loader)
        tag = "student" if use_kd else "student-no-kd"
        print(f"[{subset}][cv-{tag}] fold {k+1}/{n_folds}: test_rmse={test_rmse:.3f} test_score={test_score:.1f}")
        rows.append({"fold": k, "test_rmse": test_rmse, "test_score": test_score})
    return pd.DataFrame(rows)


def _train_no_kd_fold(subset, device, cfg, train_loader, val_loader, test_loader):
    """Inline no-KD training using the fold's loaders -- mirrors
    train_student_no_kd.py's loop (plain RMSE, no teacher) but that
    module's train_one_subset() doesn't accept loader overrides since its
    only caller is its own CLI; duplicating the ~15-line loop here for CV
    is simpler than threading loader overrides through a checkpoint-caching
    CLI wrapper never meant to be reused that way."""
    import copy
    import torch.nn as nn
    from torch.optim import AdamW
    from src.utils.train_helpers import build_scheduler, augment_batch

    torch.manual_seed(config.RANDOM_SEED)
    model = build_student(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.MSELoss()

    best_val_rmse, best_state, epochs_no_improve = float("inf"), None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X = augment_batch(X, cfg)
            optimizer.zero_grad()
            loss = torch.sqrt(criterion(model(X), y) + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()

        val_rmse, _ = _eval_loader(model, val_loader, device)
        scheduler.step(val_rmse)
        if val_rmse < best_val_rmse:
            best_val_rmse, best_state, epochs_no_improve = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return _eval_loader(model, test_loader, device)


def _eval_loader(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds.append(model(X).cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()
    return rmse(targets, preds), nasa_score(targets, preds)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "mean_test_rmse": df["test_rmse"].mean(), "std_test_rmse": df["test_rmse"].std(),
        "mean_test_score": df["test_score"].mean(), "std_test_score": df["test_score"].std(),
        "n_folds": len(df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", help="FD001 | FD002 | FD003 | FD004 | all")
    parser.add_argument("--model", default="teacher", choices=["teacher", "student", "student_no_kd"])
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    for s in subsets:
        if args.model == "teacher":
            df = run_teacher_cv(s, args.device, n_folds=args.folds)
        elif args.model == "student":
            df = run_student_cv(s, args.device, n_folds=args.folds, use_kd=True)
        else:
            df = run_student_cv(s, args.device, n_folds=args.folds, use_kd=False)

        summary = summarize(df)
        print(f"\n[{s}][cv-{args.model}] {args.folds or config.CV_N_FOLDS}-fold summary: "
              f"rmse={summary['mean_test_rmse']:.3f}+/-{summary['std_test_rmse']:.3f} "
              f"score={summary['mean_test_score']:.1f}+/-{summary['std_test_score']:.1f}")

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(config.RESULTS_DIR, f"cv_{s}_{args.model}.csv")
        df.to_csv(out_path, index=False)
        save_config_snapshot(out_path, extra={"subset": s, "model": args.model, **summary})
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
