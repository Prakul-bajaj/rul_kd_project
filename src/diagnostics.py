"""
Near-failure vs. far-from-failure error diagnostic (project review, Sec.
"Data" / "Evaluation & Diagnostics").

The single fastest check for whether a bad FD002/FD004 result is a genuine
model problem or an operating-condition confusion problem: bucket
predictions by true RUL (near-failure vs. mid-life vs. healthy/capped) and,
for multi-condition subsets, also by operating-condition cluster. If error
concentrates in the healthy/early-life bucket (and/or in specific
conditions) rather than being roughly even across an engine's life, that's
the signature of the model leaning on operating-condition regime rather
than genuine degradation signal.

Runs on the *validation* split by default rather than test: val windows
come from the sliding-window training pool (many windows per engine across
its full life), giving a much richer per-bucket sample than the C-MAPSS
test protocol's single truncated window per unit.

Usage:
    python -m src.diagnostics --subset FD002 --model teacher
    python -m src.diagnostics --subset FD002 --model teacher --split test
    python -m src.diagnostics --subset all --model student
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src import config
from src.data.preprocessing import train_val_split_by_unit
from src.models.teacher_transformer import build_teacher
from src.models.student_model import build_student
from src.models.lstm_baseline import LSTMBaseline
from src.utils.metrics import rmse, nasa_score
from src.utils.run_record import save_config_snapshot

RUL_BUCKET_EDGES = [0, 15, 30, 60, 90, config.RUL_MAX_CAP + 1]
RUL_BUCKET_LABELS = ["0-15 (near-failure)", "15-30", "30-60", "60-90", "90+ (healthy/capped)"]


def _load_model(subset: str, model_name: str, device: str):
    ckpt_map = {
        "teacher": ("teacher", build_teacher, config.TEACHER_CFG),
        "student": ("student", build_student, config.STUDENT_CFG),
        "student_no_kd": ("student_no_kd", build_student, config.STUDENT_CFG),
    }
    if model_name == "baseline":
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"lstm_baseline_{subset}.pt")
        if not os.path.exists(ckpt_path):
            return None
        model = LSTMBaseline(config.TEACHER_CFG.input_dim).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return model

    ckpt_key, builder, cfg = ckpt_map[model_name]
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_key}_{subset}.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = builder(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def _predict(model, X: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds).flatten()


def load_split_with_conditions(subset: str, split: str):
    """Return X, y_true, cond_id for the requested split, recovering the
    per-window condition id even for train/val (which weren't saved
    directly -- see diagnostics.py module docstring: the val split is
    deterministically reproducible from units_trainval via the same seed
    used at preprocessing time, so no extra preprocessing output was
    needed just for this)."""
    path = os.path.join(config.PROCESSED_DATA_DIR, f"{subset}.npz")
    data = np.load(path)

    if split == "test":
        return data["X_test"], data["y_test"], data["conds_test"]

    train_mask, val_mask = train_val_split_by_unit(
        data["units_trainval"], config.VAL_SPLIT_RATIO, config.RANDOM_SEED
    )
    mask = val_mask if split == "val" else train_mask
    return data["X_trainval"][mask], data["y_trainval"][mask], data["conds_trainval"][mask]


def bucket_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                 edges=RUL_BUCKET_EDGES, labels=RUL_BUCKET_LABELS) -> pd.DataFrame:
    rows = []
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        rows.append({
            "rul_bucket": label, "n": n,
            "rmse": rmse(y_true[mask], y_pred[mask]) if n > 0 else float("nan"),
            "score": nasa_score(y_true[mask], y_pred[mask]) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def condition_rmse(y_true: np.ndarray, y_pred: np.ndarray, cond_id: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in sorted(np.unique(cond_id)):
        mask = cond_id == c
        n = int(mask.sum())
        rows.append({
            "condition_id": int(c), "n": n,
            "rmse": rmse(y_true[mask], y_pred[mask]) if n > 0 else float("nan"),
            "score": nasa_score(y_true[mask], y_pred[mask]) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def run_diagnostic(subset: str, model_name: str, device: str, split: str = "val"):
    model = _load_model(subset, model_name, device)
    if model is None:
        print(f"[{subset}][{model_name}] no checkpoint found -- skipping")
        return None

    X, y_true, cond_id = load_split_with_conditions(subset, split)
    y_pred = _predict(model, X, device)

    overall_rmse, overall_score = rmse(y_true, y_pred), nasa_score(y_true, y_pred)
    print(f"\n=== {subset} [{model_name}] near-failure diagnostic (split={split}) ===")
    print(f"overall: rmse={overall_rmse:.3f} score={overall_score:.1f} n={len(y_true)}")

    by_rul = bucket_rmse(y_true, y_pred)
    print("\nBy true RUL (near-failure vs. healthy):")
    print(by_rul.to_string(index=False))

    by_cond = None
    if subset in config.MULTI_CONDITION_SUBSETS:
        by_cond = condition_rmse(y_true, y_pred, cond_id)
        print("\nBy operating-condition cluster:")
        print(by_cond.to_string(index=False))
        healthy_rmse = by_rul.loc[by_rul["rul_bucket"] == RUL_BUCKET_LABELS[-1], "rmse"].iloc[0]
        near_failure_rmse = by_rul.loc[by_rul["rul_bucket"] == RUL_BUCKET_LABELS[0], "rmse"].iloc[0]
        cond_spread = by_cond["rmse"].max() - by_cond["rmse"].min()
        healthy_worse = healthy_rmse > 1.5 * near_failure_rmse
        wide_cond_spread = cond_spread > 0.3 * by_cond["rmse"].mean()
        verdict = (
            "signature of operating-condition confusion (error concentrated in "
            "healthy/early-life windows and/or uneven across conditions) -- the "
            "condition-related fixes are worth pursuing here"
            if (healthy_worse or wide_cond_spread) else
            "no strong operating-condition-confusion signature -- error looks "
            "roughly even across life-stage and condition, so condition-related "
            "fixes are less likely to be the highest-leverage next step here"
        )
        print(f"\nsignature check: healthy_rmse={healthy_rmse:.2f} vs "
              f"near_failure_rmse={near_failure_rmse:.2f}, condition RMSE "
              f"spread={cond_spread:.2f} (mean condition rmse={by_cond['rmse'].mean():.2f}) "
              f"-> {verdict}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"diagnostic_{subset}_{model_name}_{split}.csv")
    by_rul.to_csv(out_path, index=False)
    if by_cond is not None:
        by_cond.to_csv(out_path.replace(".csv", "_by_condition.csv"), index=False)
    save_config_snapshot(out_path, extra={
        "subset": subset, "model": model_name, "split": split,
        "overall_rmse": overall_rmse, "overall_score": overall_score,
    })
    print(f"saved -> {out_path}")
    return by_rul, by_cond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="all", help="FD001 | FD002 | FD003 | FD004 | all")
    parser.add_argument("--model", default="teacher",
                         choices=["teacher", "student", "student_no_kd", "baseline"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    for s in subsets:
        run_diagnostic(s, args.model, args.device, args.split)


if __name__ == "__main__":
    main()
