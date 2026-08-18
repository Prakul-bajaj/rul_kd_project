"""
Teacher and student ensembling (project review, Sec. 2 "Teacher --
Architecture" and Sec. 6 "Student training & distillation process"): a
single model trained from one random initialization is itself a noisy
sample. This trains N members per model type at different seeds
(config.ENSEMBLE_SEEDS) and averages their predictions -- for the teacher,
the ensemble can additionally serve as the distillation target (averaged
soft-target + averaged pooled features) instead of a single teacher.

Checkpoints are saved per-seed as checkpoints/{teacher|student|student_no_kd}_
{subset}_seed{seed}.pt so individual members stay individually inspectable
and cheap to re-average without retraining.

Usage:
    python -m src.ensemble --subset FD001 --model teacher
    python -m src.ensemble --subset FD001 --model student
    python -m src.ensemble --subset all --model teacher --seeds 42 43 44
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from src import config
from src.config import TeacherConfig, StudentConfig
from src.data.dataset import get_dataloaders
from src.models.teacher_transformer import build_teacher
from src.models.student_model import build_student
from src.train_teacher import train_teacher_model
from src.train_student_kd import train_student
from src.utils.metrics import rmse, nasa_score
from src.utils.run_record import save_config_snapshot


class ModelEnsemble(nn.Module):
    """Averages predictions (and, if requested, pooled/sequence features)
    across a list of same-architecture models -- used for both the teacher
    ensemble and the student ensemble."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, return_features: bool = False):
        preds, pooled_list, seq_list = [], [], []
        for m in self.models:
            if return_features:
                p, feats = m(x, return_features=True)
                pooled_list.append(feats["pooled"])
                seq_list.append(feats["sequence"])
            else:
                p = m(x)
            preds.append(p)
        avg_pred = torch.stack(preds).mean(dim=0)
        if return_features:
            avg_feats = {
                "pooled": torch.stack(pooled_list).mean(dim=0),
                "sequence": torch.stack(seq_list).mean(dim=0),
            }
            return avg_pred, avg_feats
        return avg_pred


def _evaluate(model, loader, device):
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


def _member_ckpt_path(kind: str, subset: str, seed: int) -> str:
    return os.path.join(config.CHECKPOINT_DIR, f"{kind}_{subset}_seed{seed}.pt")


def train_teacher_ensemble(subset: str, device: str, seeds=None, teacher_cfg: TeacherConfig = None, force: bool = False):
    seeds = seeds or config.ENSEMBLE_SEEDS
    cfg = teacher_cfg or config.TEACHER_CFG
    models = []
    for seed in seeds:
        ckpt_path = _member_ckpt_path("teacher", subset, seed)
        if not force and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            m = build_teacher(cfg).to(device)
            m.load_state_dict(ckpt["state_dict"])
            print(f"[{subset}][teacher-ensemble] seed={seed}: loaded cached member "
                  f"(test_rmse={ckpt['test_rmse']:.3f})")
        else:
            m, test_rmse, test_score = train_teacher_model(
                subset, device, teacher_cfg=cfg, verbose=False,
                log_prefix=f"ensemble-teacher-seed{seed}", seed=seed,
            )
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({"state_dict": m.state_dict(), "config": cfg,
                        "test_rmse": test_rmse, "test_score": test_score}, ckpt_path)
            print(f"[{subset}][teacher-ensemble] seed={seed}: trained "
                  f"test_rmse={test_rmse:.3f} test_score={test_score:.1f}")
        models.append(m)

    ensemble = ModelEnsemble(models).to(device)
    _, _, test_loader = get_dataloaders(subset, cfg.batch_size)
    ens_rmse, ens_score = _evaluate(ensemble, test_loader, device)
    print(f"[{subset}][teacher-ensemble] ENSEMBLE ({len(seeds)} members) "
          f"test_rmse={ens_rmse:.3f} test_score={ens_score:.1f}")
    return ensemble, ens_rmse, ens_score


def train_student_ensemble(subset: str, device: str, seeds=None, student_cfg: StudentConfig = None,
                            teacher_model=None, force: bool = False):
    seeds = seeds or config.ENSEMBLE_SEEDS
    cfg = student_cfg or config.STUDENT_CFG
    models = []
    for seed in seeds:
        ckpt_path = _member_ckpt_path("student", subset, seed)
        if not force and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            m = build_student(cfg).to(device)
            m.load_state_dict(ckpt["state_dict"])
            print(f"[{subset}][student-ensemble] seed={seed}: loaded cached member "
                  f"(test_rmse={ckpt['test_rmse']:.3f})")
        else:
            m, _, test_rmse, test_score = train_student(
                subset, device, student_cfg=cfg, verbose=False,
                log_prefix=f"ensemble-student-seed{seed}", seed=seed, teacher_model=teacher_model,
            )
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({"state_dict": m.state_dict(), "config": cfg,
                        "test_rmse": test_rmse, "test_score": test_score}, ckpt_path)
            print(f"[{subset}][student-ensemble] seed={seed}: trained "
                  f"test_rmse={test_rmse:.3f} test_score={test_score:.1f}")
        models.append(m)

    ensemble = ModelEnsemble(models).to(device)
    _, _, test_loader = get_dataloaders(subset, cfg.batch_size)
    ens_rmse, ens_score = _evaluate(ensemble, test_loader, device)
    print(f"[{subset}][student-ensemble] ENSEMBLE ({len(seeds)} members) "
          f"test_rmse={ens_rmse:.3f} test_score={ens_score:.1f}")
    return ensemble, ens_rmse, ens_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", help="FD001 | FD002 | FD003 | FD004 | all")
    parser.add_argument("--model", default="teacher", choices=["teacher", "student"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    rows = []
    for s in subsets:
        if args.model == "teacher":
            _, ens_rmse, ens_score = train_teacher_ensemble(s, args.device, seeds=args.seeds, force=args.force)
        else:
            from src.train_student_kd import load_teacher
            teacher = load_teacher(s, args.device)
            _, ens_rmse, ens_score = train_student_ensemble(
                s, args.device, seeds=args.seeds, teacher_model=teacher, force=args.force,
            )
        rows.append({"subset": s, "model": args.model, "ensemble_rmse": ens_rmse, "ensemble_score": ens_score,
                      "seeds": args.seeds or config.ENSEMBLE_SEEDS})

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"ensemble_{args.model}.csv")
    df.to_csv(out_path, index=False)
    save_config_snapshot(out_path, extra={"subsets": subsets, "model": args.model, "seeds": args.seeds or config.ENSEMBLE_SEEDS})
    print(f"\nEnsemble results saved -> {out_path}")


if __name__ == "__main__":
    main()
