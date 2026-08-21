"""
Preprocessing pipeline for the NASA C-MAPSS turbofan degradation dataset.

Expected raw files (download from the NASA Prognostics Data Repository and
place under data/raw/):
    train_FD00{1..4}.txt
    test_FD00{1..4}.txt
    RUL_FD00{1..4}.txt

Each *_train.txt / *_test.txt row is space-separated:
    unit_id, cycle, op_setting_1..3, sensor_1..21

Run as a script to build data/processed/{subset}.npz for every subset
defined in config.SUBSETS:
    python -m src.data.preprocessing --subset FD001
    python -m src.data.preprocessing --subset all

NORMALIZATION STRATEGY
-----------------------
FD001/FD003 run under a single operating condition, so a single global
z-score normalization is sufficient.

FD002/FD004 run under 6 distinct operating conditions, and raw sensor
readings shift substantially with the operating regime itself -- independent
of actual degradation. Normalizing globally leaves that regime-shift mixed
in with the real degradation signal, which is a well-documented cause of
poor convergence on these two subsets (see e.g. Cheng et al. 2025, "An
adaptive dual distillation framework for efficient remaining useful life
prediction", which explicitly discretizes operating conditions and
normalizes independently per condition for FD002/FD004).

This module reproduces that fix: for subsets in config.MULTI_CONDITION_SUBSETS,
operating-condition rows are clustered (KMeans, config.N_OPERATING_CONDITIONS
clusters) and sensor/op-setting columns are z-score normalized *within each
cluster* rather than globally.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src import config


COLUMN_NAMES = (
    ["unit_id", "cycle", "op1", "op2", "op3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _raw_path(subset: str, split: str) -> str:
    return os.path.join(config.RAW_DATA_DIR, f"{split}_{subset}.txt")


def load_raw(subset: str, split: str) -> pd.DataFrame:
    """Load a raw C-MAPSS text file into a DataFrame with named columns."""
    path = _raw_path(subset, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the C-MAPSS dataset from the "
            f"NASA Prognostics Data Repository and place the raw .txt files "
            f"under {config.RAW_DATA_DIR}/"
        )
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_rul_labels(subset: str) -> np.ndarray:
    """Load the ground-truth RUL values provided for the test split."""
    path = os.path.join(config.RAW_DATA_DIR, f"RUL_{subset}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    return pd.read_csv(path, sep=r"\s+", header=None).values.flatten()


def compute_piecewise_rul(df: pd.DataFrame, rul_max: int = config.RUL_MAX_CAP) -> pd.DataFrame:
    """Attach a piecewise-linear RUL label to every row of the training data.

    For each unit, RUL(t) = min(max_cycle - t, rul_max). Capping the early
    (healthy) portion of life to a constant avoids biasing the regressor
    towards over-predicting RUL for engines that are still far from failure
    (Heimes 2008; used identically in Noot et al. 2025 and A-DDF 2025).
    """
    df = df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    rul = max_cycle - df["cycle"]
    df["RUL"] = rul.clip(upper=rul_max)
    return df


def select_features(df: pd.DataFrame, sensor_columns=None) -> pd.DataFrame:
    """Keep only the operating-setting + informative-sensor columns.

    sensor_columns overrides config.SELECTED_SENSOR_COLUMNS (e.g. to test
    config.ALL_21_SENSOR_COLUMNS for a specific subset -- see
    config.SENSOR_COLUMNS_BY_SUBSET / config.get_sensor_columns)."""
    sensor_columns = sensor_columns if sensor_columns is not None else config.SELECTED_SENSOR_COLUMNS
    sensor_cols = [f"sensor_{i}" for i in sensor_columns]
    op_cols = ["op1", "op2", "op3"]
    keep = ["unit_id", "cycle"] + op_cols + sensor_cols
    if "RUL" in df.columns:
        keep.append("RUL")
    return df[keep]


# ---------------------------------------------------------------------------
# Normalization: global (single-condition subsets)
# ---------------------------------------------------------------------------
def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a single global z-score scaler on the training feature columns."""
    feature_cols = [c for c in train_df.columns if c not in ("unit_id", "cycle", "RUL")]
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ("unit_id", "cycle", "RUL")]
    scaled = scaler.transform(df[feature_cols].values)
    scaled = np.clip(scaled, -config.NORMALIZATION_CLIP, config.NORMALIZATION_CLIP)
    df[feature_cols] = scaled
    return df


# ---------------------------------------------------------------------------
# Normalization: condition-wise (multi-condition subsets FD002 / FD004)
# ---------------------------------------------------------------------------
def fit_condition_clusters(train_df: pd.DataFrame, n_clusters: int) -> KMeans:
    """Cluster rows by operating condition (op1, op2, op3) on the training set."""
    op_cols = ["op1", "op2", "op3"]
    km = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10)
    km.fit(train_df[op_cols].values)
    return km


def assign_condition_clusters(df: pd.DataFrame, kmeans: KMeans) -> np.ndarray:
    op_cols = ["op1", "op2", "op3"]
    return kmeans.predict(df[op_cols].values)


def merge_small_clusters(cluster_labels: np.ndarray, kmeans: KMeans,
                          min_fraction: float = config.MIN_CLUSTER_FRACTION) -> np.ndarray:
    """Reassign points from clusters smaller than `min_fraction` of the data
    to their nearest *large* cluster.

    A cluster with very few training points produces an unstable
    StandardScaler (std computed from a handful of samples can be near
    zero), which then divides ordinary test-time deviations into extreme
    z-scores -- and a Transformer's attention (QK^T dot products) amplifies
    that instability far more than an LSTM's gated cell state does. This is
    a likely cause of a Teacher/Student RMSE blow-up that doesn't affect the
    LSTM baseline on the same preprocessed data.
    """
    n = len(cluster_labels)
    counts = pd.Series(cluster_labels).value_counts()
    small_clusters = counts[counts < min_fraction * n].index.tolist()
    if not small_clusters:
        return cluster_labels

    large_clusters = [c for c in counts.index if c not in small_clusters]
    if not large_clusters:
        # degenerate case: everything is "small" -- leave as-is rather than
        # merging into nothing.
        return cluster_labels

    print(f"  merging {len(small_clusters)} small cluster(s) "
          f"{small_clusters} (< {min_fraction*100:.0f}% of data) into "
          f"nearest larger cluster")

    centers = kmeans.cluster_centers_
    remap = {}
    for c in small_clusters:
        dists = np.linalg.norm(centers[large_clusters] - centers[c], axis=1)
        remap[c] = large_clusters[int(np.argmin(dists))]

    merged = np.array([remap.get(c, c) for c in cluster_labels])
    return merged


def fit_condition_scalers(train_df: pd.DataFrame, cluster_labels: np.ndarray) -> dict:
    """Fit one StandardScaler per operating-condition cluster on the training set."""
    feature_cols = [c for c in train_df.columns if c not in ("unit_id", "cycle", "RUL")]
    scalers = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        print(f"  cluster {cluster_id}: {mask.sum()} rows ({100*mask.sum()/len(train_df):.1f}%)")
        scaler = StandardScaler()
        scaler.fit(train_df.loc[mask, feature_cols].values)
        scalers[cluster_id] = scaler
    return scalers


def apply_condition_scalers(df: pd.DataFrame, cluster_labels: np.ndarray, scalers: dict) -> pd.DataFrame:
    """Apply the per-cluster scaler to each row according to its cluster label,
    then clip to [-NORMALIZATION_CLIP, +NORMALIZATION_CLIP] as a safety net
    against any remaining small-sample scaler instability."""
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ("unit_id", "cycle", "RUL")]
    out = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for cluster_id, scaler in scalers.items():
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            continue
        out[mask] = scaler.transform(df.loc[mask, feature_cols].values)

    n_clipped = int(np.sum(np.abs(out) > config.NORMALIZATION_CLIP))
    if n_clipped > 0:
        print(f"  clipping {n_clipped} extreme value(s) "
              f"({100*n_clipped/out.size:.3f}% of all feature values) to "
              f"+/-{config.NORMALIZATION_CLIP}")
    out = np.clip(out, -config.NORMALIZATION_CLIP, config.NORMALIZATION_CLIP)
    df[feature_cols] = out
    return df


# ---------------------------------------------------------------------------
# Operating-condition one-hot feature (optional explicit model input)
# ---------------------------------------------------------------------------
def attach_condition_column(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Attach the per-row condition cluster id as a plain integer column
    ('cond_id'), kept separate from the z-scored feature columns so it can
    be used for boundary-crossing checks / diagnostics regardless of
    whether it's also exposed to the model as an explicit input."""
    df = df.copy()
    df["cond_id"] = cluster_labels.astype(int)
    return df


def add_condition_onehot(df: pd.DataFrame, n_conditions: int) -> pd.DataFrame:
    """Expand 'cond_id' into n_conditions one-hot columns ('cond_oh_0' ..),
    appended as regular (unscaled 0/1) feature columns. Uses a fixed width
    of N_OPERATING_CONDITIONS for every subset -- including single-condition
    ones, where it's a constant [1,0,0,...] tag -- so TeacherConfig/
    StudentConfig.input_dim stays identical across subsets."""
    df = df.copy()
    for k in range(n_conditions):
        df[f"cond_oh_{k}"] = (df["cond_id"] == k).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------
def make_windows(df: pd.DataFrame, window_size: int, stride: int, is_train: bool,
                  enforce_no_crossing: bool = False):
    """Slide a fixed-length window over every unit's time series.

    Training: one window per valid (unit, end-cycle) position, label = RUL
    at the last time step of the window. If `enforce_no_crossing` is True
    and the df carries a 'cond_id' column, any candidate window whose
    'cond_id' is not constant across the whole window is dropped (Sec.
    "Data" of the project review -- a window spanning an operating-condition
    change mixes two regimes).
    Test: only the *last* window per unit is used (matches the official
    C-MAPSS evaluation protocol), left-padded with zeros if the unit's
    recorded life is shorter than window_size. The last window is always
    kept even if it crosses a condition boundary (the protocol requires
    exactly one prediction per test unit); `n_test_crossing` counts how
    often that happens so it's visible as a diagnostic, not silently lost.

    Returns X, y, unit_ids, end_cycles, end_cond_ids, n_dropped_crossing,
    n_windows_crossing (kept windows -- always 0 for train when
    enforce_no_crossing=True, since those are dropped instead; for test
    this counts how many final windows straddle a condition change).
    """
    has_cond = "cond_id" in df.columns
    feature_cols = [c for c in df.columns if c not in ("unit_id", "cycle", "RUL", "cond_id")]
    X, y, unit_ids, end_cycles, end_conds = [], [], [], [], []
    n_dropped = 0
    n_crossing_kept = 0

    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        values = g[feature_cols].values
        cycles = g["cycle"].values
        conds = g["cond_id"].values if has_cond else None
        n = len(values)

        if is_train:
            if n < window_size:
                pad = np.zeros((window_size - n, values.shape[1]))
                padded = np.vstack([pad, values])
                X.append(padded)
                y.append(g["RUL"].values[-1])
                unit_ids.append(uid)
                end_cycles.append(cycles[-1])
                end_conds.append(conds[-1] if has_cond else 0)
                continue
            for start in range(0, n - window_size + 1, stride):
                end = start + window_size
                crosses = has_cond and len(np.unique(conds[start:end])) > 1
                if enforce_no_crossing and crosses:
                    n_dropped += 1
                    continue
                if crosses:
                    n_crossing_kept += 1
                X.append(values[start:end])
                y.append(g["RUL"].values[end - 1])
                unit_ids.append(uid)
                end_cycles.append(cycles[end - 1])
                end_conds.append(conds[end - 1] if has_cond else 0)
        else:
            if n < window_size:
                pad = np.zeros((window_size - n, values.shape[1]))
                window = np.vstack([pad, values])
                win_conds = np.concatenate([np.zeros(window_size - n), conds]) if has_cond else None
            else:
                window = values[-window_size:]
                win_conds = conds[-window_size:] if has_cond else None
            if has_cond and len(np.unique(win_conds)) > 1:
                n_crossing_kept += 1
            X.append(window)
            unit_ids.append(uid)
            end_cycles.append(cycles[-1])
            end_conds.append(conds[-1] if has_cond else 0)
            y.append(None)

    if len(X) == 0:
        total_candidates = n_dropped if is_train else 0
        raise ValueError(
            f"make_windows produced 0 windows (enforce_no_crossing={enforce_no_crossing}, "
            f"{n_dropped} candidate window(s) were dropped for crossing a condition "
            f"boundary). This typically means the operating condition in this subset "
            f"changes faster than the window size, making a hard no-crossing rule "
            f"infeasible -- see the NO_WINDOW_CONDITION_CROSSING comment in "
            f"src/config.py. Set it back to False, or shrink WINDOW_SIZE, or check "
            f"the condition-run-length distribution before re-enabling it."
        )
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32) if is_train else None
    return (X, y, np.array(unit_ids), np.array(end_cycles),
            np.array(end_conds), n_dropped, n_crossing_kept)


def train_val_split_by_unit(unit_ids: np.ndarray, val_ratio: float, seed: int):
    """Split by *unit*, not by window, so no engine leaks across train/val."""
    rng = np.random.default_rng(seed)
    unique_units = np.unique(unit_ids)
    rng.shuffle(unique_units)
    n_val = max(1, int(len(unique_units) * val_ratio))
    val_units = set(unique_units[:n_val])
    val_mask = np.array([u in val_units for u in unit_ids])
    return ~val_mask, val_mask


# ---------------------------------------------------------------------------
# Main per-subset pipeline
# ---------------------------------------------------------------------------
def process_subset(subset: str, stride: int = None, suffix: str = "",
                    window_size: int = None, sensor_columns=None):
    """Build data/processed/{subset}{suffix}.npz.

    Args:
        stride: overrides config.WINDOW_STRIDE (for a stride sweep) without
            touching the module-level default.
        suffix: appended to the output filename, e.g. "_stride5", so a
            stride sweep doesn't overwrite the main processed file.
        window_size: overrides config.get_window_size(subset) (for a
            window-size sweep).
        sensor_columns: overrides config.get_sensor_columns(subset) (for a
            sensor-count A/B test, e.g. config.ALL_21_SENSOR_COLUMNS).
    """
    stride = stride if stride is not None else config.WINDOW_STRIDE
    window_size = window_size if window_size is not None else config.get_window_size(subset)
    sensor_columns = sensor_columns if sensor_columns is not None else config.get_sensor_columns(subset)

    print(f"[{subset}] loading raw data ...")
    train_raw = load_raw(subset, "train")
    test_raw = load_raw(subset, "test")
    rul_test = load_rul_labels(subset)

    train_raw = compute_piecewise_rul(train_raw)
    train_sel = select_features(train_raw, sensor_columns=sensor_columns)
    test_sel = select_features(test_raw, sensor_columns=sensor_columns)

    is_multi_condition = subset in config.MULTI_CONDITION_SUBSETS
    if is_multi_condition:
        print(f"[{subset}] multi-condition subset -> using condition-wise "
              f"normalization ({config.N_OPERATING_CONDITIONS} clusters)")
        kmeans = fit_condition_clusters(train_sel, config.N_OPERATING_CONDITIONS)
        train_clusters = assign_condition_clusters(train_sel, kmeans)
        train_clusters = merge_small_clusters(train_clusters, kmeans)
        test_clusters = assign_condition_clusters(test_sel, kmeans)
        # apply the same small-cluster remap learned on train to test labels
        # (merge_small_clusters is deterministic given the same kmeans fit,
        # so recomputing on test cluster ids with the same threshold logic
        # would double-count; instead, remap any test label that no longer
        # exists among the (already-merged) train cluster ids to its nearest
        # surviving cluster center)
        surviving = np.unique(train_clusters)
        if not set(np.unique(test_clusters)).issubset(set(surviving)):
            centers = kmeans.cluster_centers_
            remap = {}
            for c in np.unique(test_clusters):
                if c not in surviving:
                    dists = np.linalg.norm(centers[surviving] - centers[c], axis=1)
                    remap[c] = surviving[int(np.argmin(dists))]
            test_clusters = np.array([remap.get(c, c) for c in test_clusters])

        scalers = fit_condition_scalers(train_sel, train_clusters)
        train_scaled = apply_condition_scalers(train_sel, train_clusters, scalers)
        test_scaled = apply_condition_scalers(test_sel, test_clusters, scalers)
    else:
        print(f"[{subset}] single-condition subset -> using global normalization")
        scaler = fit_scaler(train_sel)
        train_scaled = apply_scaler(train_sel, scaler)
        test_scaled = apply_scaler(test_sel, scaler)
        # single condition -> everything is cluster 0, so the boundary-
        # crossing check and the one-hot condition feature are both no-ops,
        # but stay well-defined and shape-consistent across all subsets.
        train_clusters = np.zeros(len(train_scaled), dtype=int)
        test_clusters = np.zeros(len(test_scaled), dtype=int)

    train_scaled = attach_condition_column(train_scaled, train_clusters)
    test_scaled = attach_condition_column(test_scaled, test_clusters)
    if config.USE_CONDITION_FEATURE:
        train_scaled = add_condition_onehot(train_scaled, config.N_OPERATING_CONDITIONS)
        test_scaled = add_condition_onehot(test_scaled, config.N_OPERATING_CONDITIONS)

    (X_train_all, y_train_all, units_train_all, cycles_train_all,
     conds_train_all, n_dropped, _) = make_windows(
        train_scaled, window_size, stride, is_train=True,
        enforce_no_crossing=config.NO_WINDOW_CONDITION_CROSSING,
    )
    if is_multi_condition and config.NO_WINDOW_CONDITION_CROSSING:
        print(f"[{subset}] dropped {n_dropped} training window(s) that "
              f"crossed an operating-condition boundary")

    train_mask, val_mask = train_val_split_by_unit(
        units_train_all, config.VAL_SPLIT_RATIO, config.RANDOM_SEED
    )
    X_train, y_train = X_train_all[train_mask], y_train_all[train_mask]
    X_val, y_val = X_train_all[val_mask], y_train_all[val_mask]
    units_train, units_val = units_train_all[train_mask], units_train_all[val_mask]

    (X_test, _, test_unit_order, cycles_test, conds_test,
     _, n_test_crossing) = make_windows(
        test_scaled, window_size, stride, is_train=False,
        enforce_no_crossing=False,  # protocol requires exactly one window per test unit
    )
    if is_multi_condition and n_test_crossing:
        print(f"[{subset}] note: {n_test_crossing}/{len(test_unit_order)} test "
              f"unit(s) have a final window that crosses a condition boundary "
              f"(kept -- exactly one prediction per test unit is required)")
    # RUL_FD00x.txt is ordered by unit_id 1..N_test in the original files
    y_test = np.clip(rul_test.astype(np.float32), 0, config.RUL_MAX_CAP)

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DATA_DIR, f"{subset}{suffix}.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        # Full pre-split training pool + per-window metadata, kept so a
        # cross-validation harness can re-fold by unit_id without leakage,
        # and so a near-failure diagnostic / condition diagnostic can bucket
        # predictions without recomputing preprocessing.
        X_trainval=X_train_all, y_trainval=y_train_all,
        units_trainval=units_train_all, cycles_trainval=cycles_train_all,
        conds_trainval=conds_train_all,
        units_train=units_train, units_val=units_val,
        units_test=test_unit_order, cycles_test=cycles_test, conds_test=conds_test,
    )
    print(
        f"[{subset}] saved -> {out_path} | "
        f"train={X_train.shape} val={X_val.shape} test={X_test.shape} "
        f"(stride={stride}, window_size={window_size}, input_dim={X_train.shape[-1]})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset", default="all",
        help="FD001 | FD002 | FD003 | FD004 | all",
    )
    parser.add_argument("--stride", type=int, default=None,
                         help="Override WINDOW_STRIDE (for a stride sweep).")
    parser.add_argument("--window-size", type=int, default=None,
                         help="Override config.get_window_size(subset) (for a window-size sweep).")
    parser.add_argument("--suffix", default="",
                         help="Suffix for the output .npz filename, e.g. "
                              "_stride5, so it doesn't overwrite the main file.")
    args = parser.parse_args()
    subsets = config.SUBSETS if args.subset == "all" else [args.subset]
    for s in subsets:
        process_subset(s, stride=args.stride, suffix=args.suffix, window_size=args.window_size)


if __name__ == "__main__":
    main()