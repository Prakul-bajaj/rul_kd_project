"""
10-run best/ensemble robustness protocol (design-decision 9, FD001/FD002
literature pass): Noot et al. 2025 ("P01", DAST) and Wang et al. 2025
("P04", TBiGNet) both train 10 independent runs per configuration and
report either the single best (by validation loss) or an ensemble of the
best 6-of-10 (by validation loss) -- not formal k-fold, which neither
paper (nor A-DDF, "P03") uses. This replaces the earlier k-fold-only
robustness plan with that literature-matched protocol.

Methodological note: selection MUST be by validation performance, not test
performance -- picking whichever of the 10 runs happens to score best on
the test set would be test-set leakage across repeated attempts. Each
trained model is therefore re-evaluated on the val set here (the model
returned by train_teacher_model/train_student is already the best-val
checkpoint for its own run, so this reproduces the exact val_rmse that
would have been used to select it, without needing to change those
functions' return signatures).

Usage:
    python -m src.multi_seed_eval --subset FD001 --model teacher
    python -m src.multi_seed_eval --subset FD002 --model student
    python -m src.multi_seed_eval --subset FD001 --model teacher --runs 10
"""

import argparse
import dataclasses
import os

import pandas as pd
import torch

from src import config
from src.data.dataset import get_dataloaders
from src.train_teacher import train_teacher_model, evaluate as evaluate_teacher
from src.train_student_kd import train_student, load_teacher, evaluate_student
from src.ensemble import ModelEnsemble
from src.utils.run_record import save_config_snapshot


def run_multi_seed_teacher(subset: str, device: str, teacher_cfg=None, seeds=None, epochs: int = None):
    seeds = seeds or config.MULTI_SEED_SEEDS
    t_cfg = teacher_cfg or config.get_teacher_config(subset)
    if epochs is not None:
        t_cfg = dataclasses.replace(t_cfg, epochs=epochs)

    train_loader, val_loader, test_loader = get_dataloaders(subset, t_cfg.batch_size)

    rows, models = [], []
    for seed in seeds:
        model, test_rmse, test_score = train_teacher_model(
            subset, device, teacher_cfg=t_cfg, verbose=False, log_prefix=f"multiseed-teacher-seed{seed}",
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, seed=seed,
        )
        val_rmse, val_score = evaluate_teacher(model, val_loader, device)
        print(f"[{subset}][multi-seed-teacher] seed={seed}: val_rmse={val_rmse:.3f} test_rmse={test_rmse:.3f}")
        rows.append({"seed": seed, "val_rmse": val_rmse, "val_score": val_score,
                      "test_rmse": test_rmse, "test_score": test_score})
        models.append(model)

    return _summarize(subset, "teacher", rows, models, test_loader, device)


def run_multi_seed_student(subset: str, device: str, student_cfg=None, seeds=None, epochs: int = None):
    """KD student (the literature protocol reports the actual model being
    proposed, not a no-KD control -- see cross_validation.py / kd_ablation.py
    for the no-KD comparisons, which is a different, already-answered
    question from 'how noisy is this model across seeds')."""
    seeds = seeds or config.MULTI_SEED_SEEDS
    s_cfg = student_cfg or config.get_student_config(subset)
    if epochs is not None:
        s_cfg = dataclasses.replace(s_cfg, epochs=epochs)

    train_loader, val_loader, test_loader = get_dataloaders(subset, s_cfg.batch_size)
    teacher = load_teacher(subset, device)

    rows, models = [], []
    for seed in seeds:
        model, _, test_rmse, test_score = train_student(
            subset, device, student_cfg=s_cfg, verbose=False, log_prefix=f"multiseed-student-seed{seed}",
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            teacher_model=teacher, seed=seed,
        )
        val_rmse, val_score = evaluate_student(model, val_loader, device)
        print(f"[{subset}][multi-seed-student] seed={seed}: val_rmse={val_rmse:.3f} test_rmse={test_rmse:.3f}")
        rows.append({"seed": seed, "val_rmse": val_rmse, "val_score": val_score,
                      "test_rmse": test_rmse, "test_score": test_score})
        models.append(model)

    return _summarize(subset, "student", rows, models, test_loader, device)


def _summarize(subset: str, model_name: str, rows: list, models: list, test_loader, device: str):
    df = pd.DataFrame(rows)

    # (a) single best: lowest val_rmse across the N runs (P01/P04 "simple model")
    best_idx = df["val_rmse"].idxmin()
    best_row = df.loc[best_idx]

    # (b) ensemble of the best top_k-of-N by val_rmse (P01 "ensemble model")
    top_k = min(config.MULTI_SEED_ENSEMBLE_TOP_K, len(models))
    top_idx = df["val_rmse"].nsmallest(top_k).index.tolist()
    ensemble = ModelEnsemble([models[i] for i in top_idx]).to(device)
    ens_rmse, ens_score = evaluate_teacher(ensemble, test_loader, device)  # generic eval loop, model-agnostic

    print(f"[{subset}][multi-seed-{model_name}] mean/std test_rmse across {len(df)} runs: "
          f"{df['test_rmse'].mean():.3f} +/- {df['test_rmse'].std():.3f}")
    print(f"[{subset}][multi-seed-{model_name}] SINGLE BEST (seed={int(best_row['seed'])}, chosen by "
          f"val_rmse={best_row['val_rmse']:.3f}): test_rmse={best_row['test_rmse']:.3f}")
    print(f"[{subset}][multi-seed-{model_name}] ENSEMBLE (best {top_k}-of-{len(df)} by val_rmse, "
          f"seeds={[int(df.loc[i,'seed']) for i in top_idx]}): test_rmse={ens_rmse:.3f}")

    summary = {
        "subset": subset, "model": model_name, "n_runs": len(df),
        "mean_test_rmse": df["test_rmse"].mean(), "std_test_rmse": df["test_rmse"].std(),
        "mean_val_rmse": df["val_rmse"].mean(), "std_val_rmse": df["val_rmse"].std(),
        "best_seed": int(best_row["seed"]), "best_seed_val_rmse": best_row["val_rmse"],
        "best_seed_test_rmse": best_row["test_rmse"],
        "ensemble_top_k": top_k, "ensemble_test_rmse": ens_rmse, "ensemble_test_score": ens_score,
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "all"])
    parser.add_argument("--model", default="teacher", choices=["teacher", "student"])
    parser.add_argument("--runs", type=int, default=None, help="override MULTI_SEED_N_RUNS (default 10)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = config.MULTI_SEED_SEEDS[:args.runs] if args.runs else config.MULTI_SEED_SEEDS
    subsets = ["FD001", "FD002"] if args.subset == "all" else [args.subset]

    summaries = []
    for subset in subsets:
        if args.model == "teacher":
            df, summary = run_multi_seed_teacher(subset, args.device, seeds=seeds, epochs=args.epochs)
        else:
            df, summary = run_multi_seed_student(subset, args.device, seeds=seeds, epochs=args.epochs)

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(config.RESULTS_DIR, f"multi_seed_{subset}_{args.model}.csv")
        df.to_csv(out_path, index=False)
        save_config_snapshot(out_path, extra={"subset": subset, "model": args.model, "seeds": seeds, **summary})
        print(f"saved -> {out_path}")
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(config.RESULTS_DIR, f"multi_seed_summary_{args.model}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
