"""
Prints subset-by-model comparison tables for every metric in
results/comparison_all.csv, and auto-flags common sanity-check issues:

  1. FD002/FD004 (6 operating conditions) should have higher RMSE than
     FD001/FD003 (1 condition) for every model -- this is a load-bearing,
     literature-consistent pattern (Noot et al. 2025, A-DDF 2025, TBiGNet
     2025 all report it). A violation usually means a preprocessing or
     train/val/test split bug on the "easier-looking" subset, not a
     genuinely better result.
  2. Teacher should beat the LSTM Baseline on RMSE. If not, the Transformer
     teacher likely needs more training / a learning-rate fix before
     building the student on top of it.
  3. Student RMSE should not be *worse* than the LSTM Baseline -- if KD
     isn't even beating a plain lightweight LSTM, the distillation loss
     weights (alpha_task/alpha_soft/alpha_feature in config.py) probably
     need tuning.
  4. Student should have meaningfully fewer params/FLOPs than Teacher --
     flags if the "compression" is suspiciously small (<50%).

Usage:
    python -m src.compare_results
    python -m src.compare_results --metric score
"""

import argparse
import os

import pandas as pd

from src import config

SIMPLE_SUBSETS = ["FD001", "FD003"]   # 1 operating condition
COMPLEX_SUBSETS = ["FD002", "FD004"]  # 6 operating conditions

METRIC_LABELS = {
    "rmse": "RMSE (lower is better)",
    "score": "NASA Score (lower is better)",
    "params": "Parameter count (lower is better for Student)",
    "size_mb": "Model size, MB (lower is better for Student)",
    "flops": "FLOPs per forward pass (lower is better for Student)",
    "inference_ms": "Inference time, ms/sample (lower is better)",
}

def load(path=None):
    path = path or os.path.join(config.RESULTS_DIR, "comparison_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.evaluate --subset all` first, "
            f"or pass --path to point at a specific comparison_{{subset}}.csv."
        )
    return pd.read_csv(path)


def print_pivot(df, metric):
    label = METRIC_LABELS.get(metric, metric)
    print(f"\n=== {label} ===")
    pivot = df.pivot(index="subset", columns="model", values=metric)
    print(pivot.round(3).to_string())


def run_sanity_checks(df):
    print("\n" + "=" * 60)
    print("SANITY-CHECK FLAGS")
    print("=" * 60)
    issues_found = False

    if "rmse" in df.columns:
        pivot = df.pivot(index="subset", columns="model", values="rmse")

        # 1. FD002/FD004 should be harder (higher RMSE) than FD001/FD003
        for model in pivot.columns:
            simple_avg = pivot.loc[[s for s in SIMPLE_SUBSETS if s in pivot.index], model].mean()
            complex_avg = pivot.loc[[s for s in COMPLEX_SUBSETS if s in pivot.index], model].mean()
            if pd.notna(simple_avg) and pd.notna(complex_avg) and complex_avg < simple_avg:
                issues_found = True
                print(f"[FLAG] {model}: FD002/FD004 avg RMSE ({complex_avg:.2f}) is LOWER "
                      f"than FD001/FD003 avg ({simple_avg:.2f}) -- expected the opposite. "
                      f"Check preprocessing/splitting for FD002/FD004.")

        # 2. Teacher should beat LSTM Baseline
        if "Teacher (Transformer)" in pivot.columns and "LSTM Baseline" in pivot.columns:
            for subset in pivot.index:
                t, b = pivot.loc[subset, "Teacher (Transformer)"], pivot.loc[subset, "LSTM Baseline"]
                if pd.notna(t) and pd.notna(b) and t >= b:
                    issues_found = True
                    print(f"[FLAG] {subset}: Teacher RMSE ({t:.2f}) does not beat "
                          f"LSTM Baseline ({b:.2f}). Consider more teacher training "
                          f"epochs or a learning-rate adjustment.")

        # 3. Student should beat (or at least not badly lose to) LSTM Baseline
        if "Student (Distilled)" in pivot.columns and "LSTM Baseline" in pivot.columns:
            for subset in pivot.index:
                s, b = pivot.loc[subset, "Student (Distilled)"], pivot.loc[subset, "LSTM Baseline"]
                if pd.notna(s) and pd.notna(b) and s > b:
                    issues_found = True
                    print(f"[FLAG] {subset}: Student RMSE ({s:.2f}) is WORSE than "
                          f"LSTM Baseline ({b:.2f}) despite the Student having fewer "
                          f"params/FLOPs. Consider tuning alpha_task/alpha_soft/"
                          f"alpha_feature in config.py.")

    # 4. Compression sanity check
    if {"params", "model", "subset"}.issubset(df.columns):
        for subset in df["subset"].unique():
            sub = df[df["subset"] == subset]
            t_row = sub[sub["model"] == "Teacher (Transformer)"]
            s_row = sub[sub["model"] == "Student (Distilled)"]
            if not t_row.empty and not s_row.empty:
                t_params = t_row.iloc[0]["params"]
                s_params = s_row.iloc[0]["params"]
                if t_params:
                    reduction = 100 * (t_params - s_params) / t_params
                    if reduction < 50:
                        issues_found = True
                        print(f"[FLAG] {subset}: Student only reduces parameters by "
                              f"{reduction:.0f}% vs. Teacher -- expected substantially "
                              f"more (literature reports 83-98%). Check StudentConfig "
                              f"in config.py is actually smaller than TeacherConfig.")

    if not issues_found:
        print("No issues flagged -- results look internally consistent.")
    print("=" * 60)


def run_kd_attribution_check(df):
    """Directly answers: is the KD student's win actually due to distillation,
    or would the same architecture do just as well trained from scratch on
    ground truth alone? Requires student_no_kd checkpoints/rows to exist."""
    if "Student (No KD)" not in df.get("model", pd.Series()).unique():
        print("\n[info] 'Student (No KD)' rows not found -- run "
              "`python -m src.train_student_no_kd --subset all` and re-run "
              "evaluate.py to enable this check, which isolates how much of "
              "the Student's performance is actually attributable to "
              "knowledge distillation rather than the architecture alone.")
        return

    print("\n" + "=" * 60)
    print("KD ATTRIBUTION CHECK (Student with KD vs. same architecture, no KD)")
    print("=" * 60)
    pivot = df.pivot(index="subset", columns="model", values="rmse")
    for subset in pivot.index:
        kd = pivot.loc[subset].get("Student (Distilled)")
        no_kd = pivot.loc[subset].get("Student (No KD)")
        if pd.isna(kd) or pd.isna(no_kd):
            continue
        diff = no_kd - kd
        pct = 100 * diff / no_kd if no_kd else 0
        if diff > 0:
            print(f"  {subset}: KD helps -- distilled RMSE {kd:.2f} beats "
                  f"no-KD RMSE {no_kd:.2f} by {diff:.2f} ({pct:.1f}%)")
        else:
            print(f"  {subset}: KD does NOT help here -- no-KD RMSE {no_kd:.2f} "
                  f"is as good or better than distilled RMSE {kd:.2f}. The "
                  f"architecture alone may be doing the work, not distillation.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="optional path to a specific CSV")
    parser.add_argument("--metric", default="all",
                         help="rmse | score | params | size_mb | flops | inference_ms | all")
    args = parser.parse_args()

    df = load(args.path)

    metrics = list(METRIC_LABELS.keys()) if args.metric == "all" else [args.metric]
    for m in metrics:
        if m in df.columns:
            print_pivot(df, m)

    run_sanity_checks(df)
    run_kd_attribution_check(df)


if __name__ == "__main__":
    main()