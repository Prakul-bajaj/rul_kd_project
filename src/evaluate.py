"""
Final evaluation / benchmarking (Sec. 7.5): compares teacher, student, and
LSTM baseline on RMSE, NASA score, parameter count, model size, FLOPs, and
inference time, and writes results/comparison_{subset}.csv.

Prerequisite: teacher and student checkpoints already trained.

Usage:
    python -m src.evaluate --subset FD001
    python -m src.evaluate --subset all
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src import config
from src.data.dataset import get_dataloaders
from src.models.teacher_transformer import build_teacher
from src.models.student_model import build_student
from src.models.lstm_baseline import LSTMBaseline
from src.utils.metrics import rmse, nasa_score
from src.utils.complexity import profile_model
from src.utils.run_record import save_config_snapshot


def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(preds).flatten(), np.concatenate(targets).flatten()


def evaluate_subset(subset: str, device: str, train_baseline: bool = True):
    _, _, test_loader = get_dataloaders(subset, batch_size=256)

    rows = []

    # --- Teacher -----------------------------------------------------
    teacher_ckpt = os.path.join(config.CHECKPOINT_DIR, f"teacher_{subset}.pt")
    if os.path.exists(teacher_ckpt):
        ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        teacher = build_teacher(config.TEACHER_CFG).to(device)
        teacher.load_state_dict(ckpt["state_dict"])
        preds, targets = predict(teacher, test_loader, device)
        prof = profile_model(teacher, config.TEACHER_CFG.window_size,
                              config.TEACHER_CFG.input_dim, device)
        rows.append({"model": "Teacher (Transformer)", "subset": subset,
                      "rmse": rmse(targets, preds), "score": nasa_score(targets, preds),
                      **prof})
    else:
        print(f"[warn] no teacher checkpoint for {subset}, skipping")

    # --- Student -------------------------------------------------------
    student_ckpt = os.path.join(config.CHECKPOINT_DIR, f"student_{subset}.pt")
    if os.path.exists(student_ckpt):
        ckpt = torch.load(student_ckpt, map_location=device, weights_only=False)
        student = build_student(config.STUDENT_CFG).to(device)
        student.load_state_dict(ckpt["state_dict"])
        preds, targets = predict(student, test_loader, device)
        prof = profile_model(student, config.STUDENT_CFG.window_size,
                              config.STUDENT_CFG.input_dim, device)
        rows.append({"model": "Student (Distilled)", "subset": subset,
                      "rmse": rmse(targets, preds), "score": nasa_score(targets, preds),
                      **prof})
    else:
        print(f"[warn] no student checkpoint for {subset}, skipping")

    # --- Student, no KD (control: same architecture, ground-truth only) --
    no_kd_ckpt = os.path.join(config.CHECKPOINT_DIR, f"student_no_kd_{subset}.pt")
    if os.path.exists(no_kd_ckpt):
        ckpt = torch.load(no_kd_ckpt, map_location=device, weights_only=False)
        student_no_kd = build_student(config.STUDENT_CFG).to(device)
        student_no_kd.load_state_dict(ckpt["state_dict"])
        preds, targets = predict(student_no_kd, test_loader, device)
        prof = profile_model(student_no_kd, config.STUDENT_CFG.window_size,
                              config.STUDENT_CFG.input_dim, device)
        rows.append({"model": "Student (No KD)", "subset": subset,
                      "rmse": rmse(targets, preds), "score": nasa_score(targets, preds),
                      **prof})
    else:
        print(f"[warn] no student-no-kd checkpoint for {subset} -- run "
              f"`python -m src.train_student_no_kd --subset {subset}` to "
              f"include the no-KD control in this comparison")

    # --- Ensembles (only if every seed member checkpoint already exists --
    #     evaluate.py never trains ensemble members itself; run
    #     `python -m src.ensemble --subset X --model teacher/student` first) --
    for kind, builder, cfg in [("teacher", build_teacher, config.TEACHER_CFG),
                                ("student", build_student, config.STUDENT_CFG)]:
        member_paths = [os.path.join(config.CHECKPOINT_DIR, f"{kind}_{subset}_seed{seed}.pt")
                         for seed in config.ENSEMBLE_SEEDS]
        if all(os.path.exists(p) for p in member_paths):
            from src.ensemble import ModelEnsemble
            members = []
            for p in member_paths:
                ckpt = torch.load(p, map_location=device, weights_only=False)
                m = builder(cfg).to(device)
                m.load_state_dict(ckpt["state_dict"])
                members.append(m)
            ens = ModelEnsemble(members).to(device)
            preds, targets = predict(ens, test_loader, device)
            prof = profile_model(ens, cfg.window_size, cfg.input_dim, device)
            rows.append({"model": f"{kind.capitalize()} (Ensemble x{len(members)})", "subset": subset,
                          "rmse": rmse(targets, preds), "score": nasa_score(targets, preds), **prof})

    # --- LSTM baseline (trained here if not already saved) -----------
    baseline_ckpt = os.path.join(config.CHECKPOINT_DIR, f"lstm_baseline_{subset}.pt")
    if train_baseline:
        baseline = _get_or_train_baseline(subset, baseline_ckpt, device)
        preds, targets = predict(baseline, test_loader, device)
        prof = profile_model(baseline, config.TEACHER_CFG.window_size,
                              config.TEACHER_CFG.input_dim, device)
        rows.append({"model": "LSTM Baseline", "subset": subset,
                      "rmse": rmse(targets, preds), "score": nasa_score(targets, preds),
                      **prof})

    df = pd.DataFrame(rows)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"comparison_{subset}.csv")
    df.to_csv(out_path, index=False)
    cfg_path = save_config_snapshot(out_path, extra={"subset": subset})
    print(f"\n=== {subset} comparison ===")
    print(df.to_string(index=False))
    print(f"saved -> {out_path}")
    print(f"config snapshot -> {cfg_path}")
    return df


def _get_or_train_baseline(subset, ckpt_path, device):
    baseline = LSTMBaseline(config.TEACHER_CFG.input_dim).to(device)

    data_path = os.path.join(config.PROCESSED_DATA_DIR, f"{subset}.npz")
    is_stale = (
        os.path.exists(ckpt_path)
        and os.path.exists(data_path)
        and os.path.getmtime(data_path) > os.path.getmtime(ckpt_path)
    )
    if is_stale:
        print(f"[{subset}] baseline checkpoint is older than the processed "
              f"data (preprocessing changed since it was trained) -- "
              f"retraining instead of reusing it")

    if os.path.exists(ckpt_path) and not is_stale:
        baseline.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return baseline

    print(f"[{subset}] training a fresh LSTM baseline")
    train_loader, val_loader, _ = get_dataloaders(subset, batch_size=256)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    best_val, best_state = float("inf"), None
    for epoch in range(1, 51):
        baseline.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = torch.sqrt(criterion(baseline(X), y) + 1e-8)
            loss.backward()
            optimizer.step()

        preds, targets = predict(baseline, val_loader, device)
        val_rmse = rmse(targets, preds)
        if val_rmse < best_val:
            best_val, best_state = val_rmse, {k: v.clone() for k, v in baseline.state_dict().items()}

    baseline.load_state_dict(best_state)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(best_state, ckpt_path)
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    all_dfs = []
    for s in subsets:
        all_dfs.append(evaluate_subset(s, args.device, train_baseline=not args.no_baseline))

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(config.RESULTS_DIR, "comparison_all.csv")
    combined.to_csv(combined_path, index=False)
    save_config_snapshot(combined_path, extra={"subsets": subsets})
    print(f"\nCombined results saved -> {combined_path}")


if __name__ == "__main__":
    main()