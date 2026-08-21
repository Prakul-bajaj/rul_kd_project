"""
Teacher architecture sweep (project review, Sec. "Teacher -- Architecture" /
"Systematic Architecture Sweep"): the teacher's 128/8/4 configuration was
hand-picked once and never validated against alternatives, unlike the
student which went through a real sweep. This mirrors trade_off_sweep.py's
approach exactly -- it imports and calls train_teacher_model() from
train_teacher.py rather than duplicating the training loop, so a sweep
variant configured identically to config.TEACHER_CFG reproduces the same
result as `python -m src.train_teacher --subset X`.

Critically, every candidate is also compared against the LSTM baseline on
the same subset: the review's proposed bar is that a teacher config only
counts as "good enough to distil from" if it beats the baseline. Candidates
that don't are flagged in the output rather than silently ranked alongside
ones that do.

Usage:
    python -m src.teacher_arch_sweep --subset FD001
    python -m src.teacher_arch_sweep --subset all
"""

import argparse
import dataclasses
import os

import pandas as pd
import torch

from src import config
from src.config import TeacherConfig
from src.train_teacher import train_teacher_model
from src.evaluate import _get_or_train_baseline, predict
from src.data.dataset import get_dataloaders
from src.utils.metrics import rmse, nasa_score
from src.utils.complexity import profile_model
from src.utils.run_record import save_config_snapshot


# The first entry matches config.TEACHER_CFG's current defaults (built-in
# consistency check against checkpoints/teacher_{subset}.pt). The rest probe
# smaller/shallower configs (is 128/8/4 more capacity than the data
# supports?) and a GRU-hybrid variant (Sec. 2 of the review: does a small
# recurrent component stabilize a pure-attention teacher the way it helps
# the student?).
SWEEP_GRID = [
    {"name": "teacher_d128_h8_l4_default", "d_model": 128, "n_heads": 8, "n_layers": 4, "d_ff": 512},
    {"name": "teacher_d64_h4_l2", "d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 256},
    {"name": "teacher_d64_h4_l3", "d_model": 64, "n_heads": 4, "n_layers": 3, "d_ff": 256},
    {"name": "teacher_d96_h6_l3", "d_model": 96, "n_heads": 6, "n_layers": 3, "d_ff": 384},
    {"name": "teacher_d128_h8_l2", "d_model": 128, "n_heads": 8, "n_layers": 2, "d_ff": 512},
    {"name": "teacher_gru_hybrid_d128_h8_l4", "d_model": 128, "n_heads": 8, "n_layers": 4,
     "d_ff": 512, "use_gru_head": True, "gru_hidden": 128},
]


def run_sweep(subset: str, device: str, epochs: int = None):
    baseline_ckpt = os.path.join(config.CHECKPOINT_DIR, f"lstm_baseline_{subset}.pt")
    baseline = _get_or_train_baseline(subset, baseline_ckpt, device)
    _, _, test_loader = get_dataloaders(subset, batch_size=256)
    b_preds, b_targets = predict(baseline, test_loader, device)
    baseline_rmse = rmse(b_targets, b_preds)
    print(f"[{subset}][teacher-sweep] LSTM baseline rmse={baseline_rmse:.3f} (bar every teacher must clear)")

    results = []
    for variant in SWEEP_GRID:
        overrides = {k: v for k, v in variant.items() if k != "name"}
        if epochs is not None:
            overrides["epochs"] = epochs
        t_cfg = dataclasses.replace(config.get_teacher_config(subset), **overrides)

        model, test_rmse, test_score = train_teacher_model(
            subset, device, teacher_cfg=t_cfg,
            verbose=True, log_prefix=f"sweep:{variant['name']}",
        )
        prof = profile_model(model, t_cfg.window_size, t_cfg.input_dim, device)
        beats_baseline = test_rmse < baseline_rmse

        print(f"[{subset}][teacher-sweep] {variant['name']}: rmse={test_rmse:.3f} "
              f"(baseline={baseline_rmse:.3f}, {'BEATS' if beats_baseline else 'does NOT beat'} "
              f"baseline) params={prof['params']} flops={prof['flops']:.0f}")

        results.append({
            "variant": variant["name"], "subset": subset,
            "d_model": t_cfg.d_model, "n_heads": t_cfg.n_heads, "n_layers": t_cfg.n_layers,
            "d_ff": t_cfg.d_ff, "use_gru_head": t_cfg.use_gru_head, "epochs_used": t_cfg.epochs,
            "test_rmse": test_rmse, "test_score": test_score,
            "baseline_rmse": baseline_rmse, "beats_baseline": beats_baseline,
            **prof,
        })

    df = pd.DataFrame(results)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"teacher_arch_sweep_{subset}.csv")
    df.to_csv(out_path, index=False)
    save_config_snapshot(out_path, extra={"subset": subset, "sweep_grid": SWEEP_GRID, "epochs_override": epochs})
    print(f"\nTeacher sweep results saved -> {out_path}")

    if not df["beats_baseline"].any():
        print(f"[{subset}][teacher-sweep] WARNING: no swept teacher config beats "
              f"the LSTM baseline -- per the review's bar, none of these are "
              f"currently 'good enough to distil from'.")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", help="FD001 | FD002 | FD003 | FD004 | all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override epochs for every sweep variant. Omit to match "
                              "the main training run's epoch count/early-stopping.")
    args = parser.parse_args()

    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    all_dfs = []
    for s in subsets:
        all_dfs.append(run_sweep(s, args.device, args.epochs))

    if len(subsets) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(config.RESULTS_DIR, "teacher_arch_sweep_all.csv")
        combined.to_csv(combined_path, index=False)
        save_config_snapshot(combined_path, extra={"subsets": subsets, "sweep_grid": SWEEP_GRID})
        print(f"\nCombined teacher sweep results saved -> {combined_path}")


if __name__ == "__main__":
    main()
