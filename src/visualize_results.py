"""
Visualize the teacher / student / baseline comparison results produced by
evaluate.py (Sec. 7.5).

Reads results/comparison_all.csv (or a single results/comparison_{subset}.csv)
and produces a set of PNG figures under results/plots/:

  1. rmse_by_subset.png        - grouped bar chart, RMSE per model per subset
  2. score_by_subset.png       - grouped bar chart, NASA score per model per subset
  3. params_vs_rmse.png        - scatter: parameter count (log) vs RMSE
  4. flops_vs_rmse.png         - scatter: FLOPs (log) vs RMSE
  5. inference_time.png        - bar chart: per-sample inference latency (ms)
  6. compression_summary.png   - bar chart: % reduction in params/FLOPs/size,
                                  student vs teacher, one bar-group per subset
  7. efficiency_frontier.png   - bubble chart: model size (x) vs RMSE (y),
                                  bubble area = FLOPs, one panel per subset

Usage:
    python -m src.visualize_results                 # uses results/comparison_all.csv
    python -m src.visualize_results --subset FD001   # uses results/comparison_FD001.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

MODEL_ORDER = ["Teacher (Transformer)", "Student (Distilled)", "Student (No KD)", "LSTM Baseline"]
MODEL_COLORS = {
    "Teacher (Transformer)": "#1F4E78",
    "Student (Distilled)": "#2E9E5B",
    "Student (No KD)": "#D89B00",
    "LSTM Baseline": "#B0B0B0",
}


def load_results(subset: str) -> pd.DataFrame:
    fname = "comparison_all.csv" if subset == "all" else f"comparison_{subset}.csv"
    path = os.path.join(config.RESULTS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.evaluate --subset {subset}` first."
        )
    df = pd.read_csv(path)
    # keep only known models, in a fixed display order, and only known subsets
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["subset", "model"])
    return df


def _grouped_bar(df, value_col, ylabel, title, out_path, fmt="{:.2f}"):
    subsets = sorted(df["subset"].unique())
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    x = np.arange(len(subsets))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("subset").reindex(subsets)
        vals = sub[value_col].values
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                       label=model, color=MODEL_COLORS.get(model, None))
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_path}")


def plot_rmse_by_subset(df, out_dir):
    _grouped_bar(df, "rmse", "RMSE (cycles)", "RMSE by model and subset",
                 os.path.join(out_dir, "rmse_by_subset.png"))


def plot_score_by_subset(df, out_dir):
    _grouped_bar(df, "score", "NASA Score (lower is better)",
                 "NASA scoring function by model and subset",
                 os.path.join(out_dir, "score_by_subset.png"), fmt="{:.0f}")


def plot_inference_time(df, out_dir):
    _grouped_bar(df, "inference_ms", "Inference time (ms / sample)",
                 "Per-sample inference latency by model and subset",
                 os.path.join(out_dir, "inference_time.png"), fmt="{:.3f}")


def plot_params_vs_rmse(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model in [m for m in MODEL_ORDER if m in df["model"].unique()]:
        sub = df[df["model"] == model]
        ax.scatter(sub["params"], sub["rmse"], s=90,
                   color=MODEL_COLORS.get(model), label=model, edgecolor="white")
        for _, row in sub.iterrows():
            ax.annotate(row["subset"], (row["params"], row["rmse"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Model size vs. accuracy trade-off")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "params_vs_rmse.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_path}")


def plot_flops_vs_rmse(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model in [m for m in MODEL_ORDER if m in df["model"].unique()]:
        sub = df[df["model"] == model]
        ax.scatter(sub["flops"], sub["rmse"], s=90,
                   color=MODEL_COLORS.get(model), label=model, edgecolor="white")
        for _, row in sub.iterrows():
            ax.annotate(row["subset"], (row["flops"], row["rmse"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per forward pass (log scale)")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Computational cost vs. accuracy trade-off")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "flops_vs_rmse.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_path}")


def plot_compression_summary(df, out_dir):
    """% reduction of student vs. teacher for params / FLOPs / size / inference time,
    one bar-group per subset. Requires both a Teacher and Student row per subset."""
    metrics = ["params", "flops", "size_mb", "inference_ms"]
    labels = ["Parameters", "FLOPs", "Model size", "Inference time"]
    subsets = sorted(df["subset"].unique())

    reduction = {m: [] for m in metrics}
    valid_subsets = []
    for s in subsets:
        t = df[(df["subset"] == s) & (df["model"] == "Teacher (Transformer)")]
        st = df[(df["subset"] == s) & (df["model"] == "Student (Distilled)")]
        if t.empty or st.empty:
            continue
        valid_subsets.append(s)
        for m in metrics:
            t_val, s_val = t.iloc[0][m], st.iloc[0][m]
            pct = 100.0 * (t_val - s_val) / t_val if t_val else 0.0
            reduction[m].append(pct)

    if not valid_subsets:
        print("[visualize] skipping compression_summary.png -- need both "
              "Teacher and Student rows per subset")
        return

    x = np.arange(len(valid_subsets))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = reduction[m]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=lab)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_subsets)
    ax.set_ylabel("% reduction vs. teacher (higher = more compressed)")
    ax.set_title("Student compression relative to teacher")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "compression_summary.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_path}")


def plot_efficiency_frontier(df, out_dir):
    """Bubble chart: model size (MB) vs RMSE, bubble area ~ FLOPs, one panel/subset."""
    subsets = sorted(df["subset"].unique())
    n = len(subsets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    flops_all = df["flops"].replace(0, np.nan).dropna()
    flops_min, flops_max = (flops_all.min(), flops_all.max()) if not flops_all.empty else (1, 1)

    def scale_area(flops):
        if flops_max == flops_min:
            return 300
        norm = (flops - flops_min) / (flops_max - flops_min)
        return 150 + norm * 1500

    for ax, s in zip(axes[0], subsets):
        sub = df[df["subset"] == s]
        for model in [m for m in MODEL_ORDER if m in sub["model"].unique()]:
            row = sub[sub["model"] == model].iloc[0]
            ax.scatter(row["size_mb"], row["rmse"], s=scale_area(row["flops"]),
                       color=MODEL_COLORS.get(model), alpha=0.75,
                       edgecolor="white", label=model)
        ax.set_title(s)
        ax.set_xlabel("Model size (MB)")
        ax.set_ylabel("RMSE (cycles)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle("Efficiency frontier (bubble area \u221d FLOPs)", y=1.14, fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "efficiency_frontier.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="all",
                         help="all | FD001 | FD002 | FD003 | FD004")
    args = parser.parse_args()

    df = load_results(args.subset)
    if df.empty:
        raise SystemExit("No matching rows found -- check results/comparison_*.csv content.")

    out_dir = os.path.join(config.RESULTS_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_rmse_by_subset(df, out_dir)
    plot_score_by_subset(df, out_dir)
    plot_inference_time(df, out_dir)
    plot_params_vs_rmse(df, out_dir)
    plot_flops_vs_rmse(df, out_dir)
    plot_compression_summary(df, out_dir)
    plot_efficiency_frontier(df, out_dir)

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()