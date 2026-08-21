"""
Final FD001/FD002 training pass under the literature-grounded "mastery"
config (see src/config.py's WINDOW_SIZE_BY_SUBSET / SENSOR_COLUMNS_BY_SUBSET
/ TEACHER_ARCH_BY_SUBSET, all finalized by src/fd_mastery_sweep.py):

  1. preprocess FD001/FD002 with the finalized per-subset window/sensors
  2. train official teacher_{subset}.pt (per-subset architecture)
  3. train official student_{subset}.pt (KD) and student_no_kd_{subset}.pt
  4. teacher + student 3-seed ensembles
  5. final evaluate.py comparison table
  6. 10-run best/ensemble robustness protocol (design-decision 9 --
     replaces the earlier k-fold CV plan; see src/multi_seed_eval.py)

Usage:
    python -m scripts.run_fd001_fd002_final
"""

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
from src.train_teacher import train_one_subset as train_teacher_cli
from src.train_student_kd import train_one_subset as train_student_kd_cli
from src.train_student_no_kd import train_one_subset as train_student_no_kd_cli
from src.ensemble import train_teacher_ensemble, train_student_ensemble
from src.train_student_kd import load_teacher
from src.evaluate import evaluate_subset
from src.multi_seed_eval import run_multi_seed_teacher, run_multi_seed_student

SUBSETS = ["FD001", "FD002"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_PATH = os.path.join(config.RESULTS_DIR, "FD_FINAL_LOG.txt")
SUMMARY_PATH = os.path.join(config.RESULTS_DIR, "FD_FINAL_SUMMARY.json")

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
                summary["stages"][name] = {"status": "FAILED", "error": str(e), "seconds": round(time.time() - t0, 1)}
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
    for s in SUBSETS:
        process_subset(s)


@stage("2_official_teacher")
def stage_teacher():
    for s in SUBSETS:
        train_teacher_cli(s, DEVICE, force=True)


@stage("3_official_student_kd_and_nokd")
def stage_student():
    for s in SUBSETS:
        train_student_kd_cli(s, DEVICE, force=True)
        train_student_no_kd_cli(s, DEVICE, force=True)


@stage("4_ensembles")
def stage_ensembles():
    for s in SUBSETS:
        train_teacher_ensemble(s, DEVICE, force=True)
        teacher = load_teacher(s, DEVICE)
        train_student_ensemble(s, DEVICE, teacher_model=teacher, force=True)


@stage("5_final_evaluate")
def stage_evaluate():
    dfs = [evaluate_subset(s, DEVICE, train_baseline=True) for s in SUBSETS]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(config.RESULTS_DIR, "comparison_all.csv"), index=False)


@stage("6_multi_seed_protocol")
def stage_multi_seed():
    rows = []
    for s in SUBSETS:
        for model_name, fn in [("teacher", run_multi_seed_teacher), ("student", run_multi_seed_student)]:
            df, summ = fn(s, DEVICE)
            df.to_csv(os.path.join(config.RESULTS_DIR, f"multi_seed_{s}_{model_name}.csv"), index=False)
            rows.append(summ)
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "multi_seed_summary_final.csv"), index=False)


def main():
    log(f"FD001/FD002 FINAL RUN START -- device={DEVICE}")
    stage_preprocess()
    stage_teacher()
    stage_student()
    stage_ensembles()
    stage_evaluate()
    stage_multi_seed()
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log("FD001/FD002 FINAL RUN DONE")


if __name__ == "__main__":
    main()
