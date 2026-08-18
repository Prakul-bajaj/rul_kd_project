"""
KD-loss-term ablation (project review, Sec. "Student -- Architecture" /
"Student training & distillation process"): the student always received
all three distillation terms (task, soft-target, feature-alignment)
bundled together. "All three vs. none" (KD vs. the no-KD control) was
already tested, but never "which one or two of the three actually matter"
-- this sweeps every non-empty combination of the three terms (by zeroing
out the alphas for the excluded ones) using the exact same student
architecture and training loop as train_student_kd.py, so the only thing
that varies between rows is which loss terms are active.

Usage:
    python -m src.kd_ablation --subset FD001
    python -m src.kd_ablation --subset all
"""

import argparse
import dataclasses
import os

import pandas as pd
import torch

from src import config
from src.config import DistillationConfig
from src.train_student_kd import train_student
from src.utils.complexity import profile_model
from src.utils.run_record import save_config_snapshot


# Task is kept on in every variant (a student trained on soft/feature terms
# alone with no ground-truth signal isn't a meaningful comparison point for
# this project); every combination of soft/feature on top of it is swept.
# Weights among active terms are renormalized to sum to 1 so e.g.
# "task_only" isn't handicapped by an alpha_task of 0.5 relative to a
# variant that gets to keep alpha_task + alpha_soft = 0.8.
ABLATION_GRID = [
    {"name": "task_only", "use_soft": False, "use_feature": False},
    {"name": "task_soft", "use_soft": True, "use_feature": False},
    {"name": "task_feature", "use_soft": False, "use_feature": True},
    {"name": "task_soft_feature_all", "use_soft": True, "use_feature": True},
]


def _renormalized_cfg(base: DistillationConfig, use_soft: bool, use_feature: bool) -> DistillationConfig:
    raw = {
        "alpha_task": base.alpha_task,
        "alpha_soft": base.alpha_soft if use_soft else 0.0,
        "alpha_feature": base.alpha_feature if use_feature else 0.0,
    }
    total = sum(raw.values())
    return dataclasses.replace(base, **{k: v / total for k, v in raw.items()})


def run_ablation(subset: str, device: str, epochs: int = None):
    base_kd_cfg = config.KD_CFG
    s_cfg = config.STUDENT_CFG if epochs is None else dataclasses.replace(config.STUDENT_CFG, epochs=epochs)

    results = []
    for variant in ABLATION_GRID:
        kd_cfg = _renormalized_cfg(base_kd_cfg, variant["use_soft"], variant["use_feature"])

        # train_student() reads config.KD_CFG globally (it's constructed
        # inside the function from src.config's module-level KD_CFG), so we
        # swap it for the duration of this call and restore it after --
        # same trick already used when smoke-testing adaptive_weighting.
        original_kd_cfg = config.KD_CFG
        config.KD_CFG = kd_cfg
        try:
            student, _, test_rmse, test_score = train_student(
                subset, device, student_cfg=s_cfg,
                verbose=True, log_prefix=f"kd-ablation:{variant['name']}",
            )
        finally:
            config.KD_CFG = original_kd_cfg

        prof = profile_model(student, s_cfg.window_size, s_cfg.input_dim, device)
        print(f"[{subset}][kd-ablation] {variant['name']}: rmse={test_rmse:.3f} "
              f"(alpha_task={kd_cfg.alpha_task:.2f} alpha_soft={kd_cfg.alpha_soft:.2f} "
              f"alpha_feature={kd_cfg.alpha_feature:.2f})")

        results.append({
            "variant": variant["name"], "subset": subset,
            "alpha_task": kd_cfg.alpha_task, "alpha_soft": kd_cfg.alpha_soft,
            "alpha_feature": kd_cfg.alpha_feature,
            "test_rmse": test_rmse, "test_score": test_score, **prof,
        })

    df = pd.DataFrame(results)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"kd_ablation_{subset}.csv")
    df.to_csv(out_path, index=False)
    save_config_snapshot(out_path, extra={"subset": subset, "ablation_grid": ABLATION_GRID})
    print(f"\nKD ablation results saved -> {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", help="FD001 | FD002 | FD003 | FD004 | all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    all_dfs = [run_ablation(s, args.device, args.epochs) for s in subsets]

    if len(subsets) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(config.RESULTS_DIR, "kd_ablation_all.csv")
        combined.to_csv(combined_path, index=False)
        save_config_snapshot(combined_path, extra={"subsets": subsets, "ablation_grid": ABLATION_GRID})
        print(f"\nCombined KD ablation results saved -> {combined_path}")


if __name__ == "__main__":
    main()
