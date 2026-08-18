"""
Trade-off analysis (Sec. 7.6): trains several student configurations at
different compression levels (fewer attention heads, fewer encoder layers,
smaller embedding dimension) via the exact same KD training procedure used
by train_student_kd.py, and records RMSE vs. efficiency for each, so the
accuracy/efficiency frontier can be plotted.

This script does NOT duplicate the training loop -- it imports and calls
train_student() from train_student_kd.py directly, so a sweep variant
configured identically to config.STUDENT_CFG reproduces the same result
as `python -m src.train_student_kd --subset X` (same seed, same init, same
epochs/early-stopping, same optimizer). See the docstring on train_student()
in train_student_kd.py for details.

Usage:
    python -m src.trade_off_sweep --subset FD001
    python -m src.trade_off_sweep --subset all
"""

import argparse
import dataclasses
import os

import pandas as pd
import torch

from src import config
from src.config import StudentConfig
from src.train_student_kd import train_student
from src.utils.complexity import profile_model
from src.utils.run_record import save_config_snapshot


# Each entry overrides fields of the default StudentConfig to sweep one or
# more compression axes. The first entry intentionally matches
# config.STUDENT_CFG's current defaults (d_model=16, n_heads=1, n_layers=1,
# use_gru_head=True -- winner of the original 4-candidate sweep) so it
# serves as a built-in consistency check against comparison_{subset}.csv --
# with train_student() shared between both scripts and the seed fixed, this
# row should reproduce the same RMSE as the "Student (Distilled)" row there.
#
# Widened per the project review (Sec. "Student -- Architecture"): the
# original sweep only tried 4 combinations, all with the GRU head on. This
# adds smaller-than-winner options (can we compress further for free?),
# a couple more mid-sized options, and a use_gru_head=False counterpart for
# every size so the GRU's actual accuracy contribution is measurable
# instead of assumed.
SWEEP_GRID = [
    {"name": "student_d16_h1_l1_gru_default", "d_model": 16, "n_heads": 1, "n_layers": 1, "use_gru_head": True},
    {"name": "student_d16_h1_l1_nogru", "d_model": 16, "n_heads": 1, "n_layers": 1, "use_gru_head": False},
    {"name": "student_d8_h1_l1_gru", "d_model": 8, "n_heads": 1, "n_layers": 1, "use_gru_head": True, "gru_hidden": 16},
    {"name": "student_d8_h1_l1_nogru", "d_model": 8, "n_heads": 1, "n_layers": 1, "use_gru_head": False},
    {"name": "student_d24_h1_l1_gru", "d_model": 24, "n_heads": 1, "n_layers": 1, "use_gru_head": True, "gru_hidden": 32},
    {"name": "student_d24_h1_l1_nogru", "d_model": 24, "n_heads": 1, "n_layers": 1, "use_gru_head": False},
    {"name": "student_d32_h2_l1_gru", "d_model": 32, "n_heads": 2, "n_layers": 1, "use_gru_head": True, "gru_hidden": 32},
    {"name": "student_d32_h2_l1_nogru", "d_model": 32, "n_heads": 2, "n_layers": 1, "use_gru_head": False},
    {"name": "student_d32_h2_l2_gru", "d_model": 32, "n_heads": 2, "n_layers": 2, "use_gru_head": True, "gru_hidden": 32},
    {"name": "student_d32_h2_l2_nogru", "d_model": 32, "n_heads": 2, "n_layers": 2, "use_gru_head": False},
    {"name": "student_d64_h4_l2_gru", "d_model": 64, "n_heads": 4, "n_layers": 2, "use_gru_head": True, "gru_hidden": 64},
    {"name": "student_d64_h4_l2_nogru", "d_model": 64, "n_heads": 4, "n_layers": 2, "use_gru_head": False},
]


def run_sweep(subset: str, device: str, epochs: int = None):
    """Run every SWEEP_GRID variant on one subset.

    Args:
        epochs: overrides StudentConfig.epochs for every variant if given
            (useful to shorten the sweep for a quick exploratory pass).
            Leave as None to use the same epoch count / early-stopping
            patience as the main training run -- recommended once you're
            ready to report final numbers, since it keeps every sweep
            variant on equal footing with comparison_{subset}.csv.
    """
    results = []
    for variant in SWEEP_GRID:
        overrides = {k: v for k, v in variant.items() if k != "name"}
        if epochs is not None:
            overrides["epochs"] = epochs
        s_cfg = dataclasses.replace(StudentConfig(), **overrides)

        student, _, test_rmse, test_score = train_student(
            subset, device, student_cfg=s_cfg,
            verbose=True, log_prefix=f"sweep:{variant['name']}",
        )
        prof = profile_model(student, s_cfg.window_size, s_cfg.input_dim, device)

        print(f"[{subset}][sweep] {variant['name']}: "
              f"rmse={test_rmse:.3f} params={prof['params']} flops={prof['flops']:.0f}")

        results.append({
            "variant": variant["name"], "subset": subset,
            "d_model": s_cfg.d_model, "n_heads": s_cfg.n_heads,
            "n_layers": s_cfg.n_layers, "epochs_used": s_cfg.epochs,
            "test_rmse": test_rmse, "test_score": test_score, **prof,
        })

    df = pd.DataFrame(results)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"tradeoff_sweep_{subset}.csv")
    df.to_csv(out_path, index=False)
    save_config_snapshot(out_path, extra={"subset": subset, "sweep_grid": SWEEP_GRID, "epochs_override": epochs})
    print(f"\nSweep results saved -> {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001",
                         help="FD001 | FD002 | FD003 | FD004 | all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override epochs for every sweep variant. "
                              "Omit to match the main training run's epoch "
                              "count/early-stopping exactly (recommended).")
    args = parser.parse_args()

    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    all_dfs = []
    for s in subsets:
        all_dfs.append(run_sweep(s, args.device, args.epochs))

    if len(subsets) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(config.RESULTS_DIR, "tradeoff_sweep_all.csv")
        combined.to_csv(combined_path, index=False)
        save_config_snapshot(combined_path, extra={"subsets": subsets, "sweep_grid": SWEEP_GRID})
        print(f"\nCombined sweep results saved -> {combined_path}")


if __name__ == "__main__":
    main()