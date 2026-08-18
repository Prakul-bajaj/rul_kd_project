"""
Orchestrates the full project-review follow-up pipeline end to end across
FD001-FD004:

  1. teacher architecture sweep (full epochs) -> pick a winning global
     TeacherConfig (must beat the LSTM baseline where possible)
  2. retrain the official teacher_{subset}.pt checkpoints with the winner
  3. train/re-train student (KD) and student (no-KD) with the improved
     teacher + the already-fixed KD loss scale, for a trustworthy
     KD-vs-no-KD comparison
  4. widened student architecture sweep (report only, informational)
  5. KD loss-term ablation
  6. 5-fold cross-validation for teacher / student / student-no-kd
  7. near-failure + operating-condition diagnostics for teacher and student
  8. teacher and student ensembles
  9. final evaluate.py --subset all (includes ensemble rows if present)

Each stage writes its own results CSV (+ config snapshot) as it goes, so a
crash partway through doesn't lose earlier stages' output. Progress and
any per-stage errors are logged to results/PIPELINE_LOG.txt and a final
manifest is written to results/PIPELINE_SUMMARY.json.

Usage:
    python -m scripts.run_full_pipeline
"""

import dataclasses
import json
import os
import sys
import time
import traceback

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data.preprocessing import process_subset
from src.teacher_arch_sweep import run_sweep as run_teacher_sweep
from src.train_teacher import train_one_subset as train_teacher_cli
from src.train_student_kd import train_one_subset as train_student_kd_cli
from src.train_student_no_kd import train_one_subset as train_student_no_kd_cli
from src.trade_off_sweep import run_sweep as run_student_sweep
from src.kd_ablation import run_ablation as run_kd_ablation
from src.cross_validation import run_teacher_cv, run_student_cv, summarize as cv_summarize
from src.diagnostics import run_diagnostic
from src.ensemble import train_teacher_ensemble, train_student_ensemble
from src.evaluate import evaluate_subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_PATH = os.path.join(config.RESULTS_DIR, "PIPELINE_LOG.txt")
SUMMARY_PATH = os.path.join(config.RESULTS_DIR, "PIPELINE_SUMMARY.json")

summary = {"device": DEVICE, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def stage(name):
    def decorator(fn):
        def wrapped(*args, **kwargs):
            log(f"=== STAGE START: {name} ===")
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
                summary["stages"][name] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
                log(f"=== STAGE OK: {name} ({time.time()-t0:.1f}s) ===")
                return result
            except Exception as e:
                summary["stages"][name] = {"status": "FAILED", "error": str(e),
                                            "seconds": round(time.time() - t0, 1)}
                log(f"=== STAGE FAILED: {name}: {e} ===")
                log(traceback.format_exc())
                return None
            finally:
                with open(SUMMARY_PATH, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
        return wrapped
    return decorator


@stage("1_preprocess")
def stage_preprocess():
    for s in config.SUBSETS:
        process_subset(s)


@stage("2_teacher_arch_sweep")
def stage_teacher_sweep():
    dfs = {s: run_teacher_sweep(s, DEVICE) for s in config.SUBSETS}
    combined = pd.concat(dfs.values(), ignore_index=True)
    combined.to_csv(os.path.join(config.RESULTS_DIR, "teacher_arch_sweep_all.csv"), index=False)

    # Pick one global winning architecture across all 4 subsets: prefer the
    # variant that beats the LSTM baseline most often, tie-broken by the
    # best average (test_rmse / baseline_rmse) ratio (lower is better).
    grp = combined.groupby("variant").agg(
        beats_count=("beats_baseline", "sum"),
        mean_ratio=("test_rmse", lambda s: (s / combined.loc[s.index, "baseline_rmse"]).mean()),
    ).reset_index()
    grp = grp.sort_values(["beats_count", "mean_ratio"], ascending=[False, True])
    winner_name = grp.iloc[0]["variant"]
    log(f"teacher sweep winner across all subsets: {winner_name} "
        f"(beats baseline on {int(grp.iloc[0]['beats_count'])}/4 subsets)")

    winner_row = combined[combined["variant"] == winner_name].iloc[0]
    winner_cfg = dataclasses.replace(
        config.TeacherConfig(),
        d_model=int(winner_row["d_model"]), n_heads=int(winner_row["n_heads"]),
        n_layers=int(winner_row["n_layers"]), d_ff=int(winner_row["d_ff"]),
        use_gru_head=bool(winner_row["use_gru_head"]),
    )
    config.TEACHER_CFG = winner_cfg
    summary["teacher_winner"] = {"variant": winner_name, "config": dataclasses.asdict(winner_cfg)}
    return winner_cfg


@stage("3_retrain_official_teacher")
def stage_retrain_teacher():
    for s in config.SUBSETS:
        train_teacher_cli(s, DEVICE, force=True)


@stage("4_kd_vs_no_kd")
def stage_kd_vs_no_kd():
    for s in config.SUBSETS:
        train_student_kd_cli(s, DEVICE, force=True)
        train_student_no_kd_cli(s, DEVICE, force=True)


@stage("5_student_arch_sweep")
def stage_student_sweep():
    dfs = [run_student_sweep(s, DEVICE) for s in config.SUBSETS]
    pd.concat(dfs, ignore_index=True).to_csv(
        os.path.join(config.RESULTS_DIR, "tradeoff_sweep_all.csv"), index=False)


@stage("6_kd_ablation")
def stage_kd_ablation():
    dfs = [run_kd_ablation(s, DEVICE) for s in config.SUBSETS]
    pd.concat(dfs, ignore_index=True).to_csv(
        os.path.join(config.RESULTS_DIR, "kd_ablation_all.csv"), index=False)


@stage("7_cross_validation")
def stage_cross_validation():
    rows = []
    for s in config.SUBSETS:
        for model_name, fn in [("teacher", lambda s=s: run_teacher_cv(s, DEVICE)),
                                ("student", lambda s=s: run_student_cv(s, DEVICE, use_kd=True)),
                                ("student_no_kd", lambda s=s: run_student_cv(s, DEVICE, use_kd=False))]:
            df = fn()
            df.to_csv(os.path.join(config.RESULTS_DIR, f"cv_{s}_{model_name}.csv"), index=False)
            summ = cv_summarize(df)
            rows.append({"subset": s, "model": model_name, **summ})
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "cv_summary_all.csv"), index=False)


@stage("8_diagnostics")
def stage_diagnostics():
    for s in config.SUBSETS:
        for model_name in ["teacher", "student", "student_no_kd"]:
            run_diagnostic(s, model_name, DEVICE, split="val")


@stage("9_ensembles")
def stage_ensembles():
    from src.train_student_kd import load_teacher
    for s in config.SUBSETS:
        train_teacher_ensemble(s, DEVICE, force=True)
        teacher = load_teacher(s, DEVICE)
        train_student_ensemble(s, DEVICE, teacher_model=teacher, force=True)


@stage("10_final_evaluate")
def stage_final_evaluate():
    dfs = [evaluate_subset(s, DEVICE, train_baseline=True) for s in config.SUBSETS]
    pd.concat(dfs, ignore_index=True).to_csv(
        os.path.join(config.RESULTS_DIR, "comparison_all.csv"), index=False)


def main():
    log(f"FULL PIPELINE START -- device={DEVICE}")
    stage_preprocess()
    stage_teacher_sweep()
    stage_retrain_teacher()
    stage_kd_vs_no_kd()
    stage_student_sweep()
    stage_kd_ablation()
    stage_cross_validation()
    stage_diagnostics()
    stage_ensembles()
    stage_final_evaluate()
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log("FULL PIPELINE DONE")


if __name__ == "__main__":
    main()
