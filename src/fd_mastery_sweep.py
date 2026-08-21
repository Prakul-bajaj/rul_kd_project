"""
FD001/FD002 "mastery" sweep set -- implements decisions 1, 3, 4, and 6 of
the literature-grounded design-decision pass (Noot et al. 2025 "P01" DAST,
Cheng et al. 2025 "P03" A-DDF, Wang et al. 2025 "P04" TBiGNet). FD003/FD004
are out of scope for this pass.

Four independent phases, run in sequence (each phase's winner feeds the
next, since window size and sensor count change input_dim/window_size for
everything downstream):

  1. Window-size sweep for FD002 only ({30,40,50,60} -- FD001 stays fixed
     at 30, per P01's own subset-dependent choice).
  2. Sensor-count A/B test for FD002 only (14, the current fixed list, vs.
     all 21 -- P01 keeps all 21 specifically for FD002/FD004).
  3. Teacher architecture sweep for BOTH FD001 and FD002, using each
     subset's now-finalized window size / sensor count: d_model in
     {32,64,96,128}, n_heads in {2,4} (only where d_model % n_heads == 0),
     n_layers in {1,2,3}, d_ff = 2*d_model (P01's expansion ratio, not the
     project's old 4x).
  4. Batch-size sweep for both subsets ({32,64,128,256}).

Usage:
    python -m src.fd_mastery_sweep
    python -m src.fd_mastery_sweep --epochs 30          # quick pass
    python -m src.fd_mastery_sweep --skip-window --skip-sensors  # arch+batch only
"""

import argparse
import dataclasses
import os

import pandas as pd
import torch

from src import config
from src.data.preprocessing import process_subset
from src.data.dataset import get_dataloaders
from src.train_teacher import train_teacher_model
from src.utils.complexity import profile_model
from src.utils.run_record import save_config_snapshot

WINDOW_SWEEP_VALUES = [30, 40, 50, 60]
BATCH_SIZE_SWEEP = [32, 64, 128, 256]

TEACHER_ARCH_GRID = [
    {"name": f"d{d}_h{h}_l{l}", "d_model": d, "n_heads": h, "n_layers": l, "d_ff": d * 2}
    for d in [32, 64, 96, 128]
    for h in [2, 4]
    if d % h == 0
    for l in [1, 2, 3]
]


def run_window_sweep(subset: str, device: str, epochs: int = None):
    rows = []
    for ws in WINDOW_SWEEP_VALUES:
        suffix = f"_wsweep{ws}"
        process_subset(subset, window_size=ws, suffix=suffix)
        t_cfg = config.get_teacher_config(subset, window_size=ws)
        if epochs is not None:
            t_cfg = dataclasses.replace(t_cfg, epochs=epochs)
        train_loader, val_loader, test_loader = get_dataloaders(subset, t_cfg.batch_size, suffix=suffix)
        _, test_rmse, test_score = train_teacher_model(
            subset, device, teacher_cfg=t_cfg, verbose=False, log_prefix=f"window-sweep:{ws}",
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        )
        print(f"[{subset}][window-sweep] window={ws}: rmse={test_rmse:.3f} score={test_score:.1f}")
        rows.append({"subset": subset, "window_size": ws, "test_rmse": test_rmse, "test_score": test_score})

    df = pd.DataFrame(rows)
    out_path = os.path.join(config.RESULTS_DIR, f"fd_mastery_window_sweep_{subset}.csv")
    df.to_csv(out_path, index=False)
    save_config_snapshot(out_path, extra={"subset": subset, "window_values": WINDOW_SWEEP_VALUES})
    winner = int(df.loc[df["test_rmse"].idxmin(), "window_size"])
    print(f"[{subset}][window-sweep] WINNER: window_size={winner}\nsaved -> {out_path}")
    return df, winner


def run_sensor_sweep(subset: str, device: str, window_size: int, epochs: int = None):
    variants = [("14_sensors", config.SELECTED_SENSOR_COLUMNS), ("21_sensors", config.ALL_21_SENSOR_COLUMNS)]
    rows = []
    for name, cols in variants:
        suffix = f"_sensors_{name}"
        process_subset(subset, window_size=window_size, sensor_columns=cols, suffix=suffix)
        input_dim = config.compute_input_dim(sensor_columns=cols)
        t_cfg = config.get_teacher_config(subset, window_size=window_size, input_dim=input_dim)
        if epochs is not None:
            t_cfg = dataclasses.replace(t_cfg, epochs=epochs)
        train_loader, val_loader, test_loader = get_dataloaders(subset, t_cfg.batch_size, suffix=suffix)
        _, test_rmse, test_score = train_teacher_model(
            subset, device, teacher_cfg=t_cfg, verbose=False, log_prefix=f"sensor-sweep:{name}",
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        )
        print(f"[{subset}][sensor-sweep] {name} (n={len(cols)}): rmse={test_rmse:.3f} score={test_score:.1f}")
        rows.append({"subset": subset, "variant": name, "n_sensors": len(cols),
                      "test_rmse": test_rmse, "test_score": test_score})

    df = pd.DataFrame(rows)
    out_path = os.path.join(config.RESULTS_DIR, f"fd_mastery_sensor_test_{subset}.csv")
    df.to_csv(out_path, index=False)
    save_config_snapshot(out_path, extra={"subset": subset, "window_size": window_size})
    winner = df.loc[df["test_rmse"].idxmin(), "variant"]
    print(f"[{subset}][sensor-sweep] WINNER: {winner}\nsaved -> {out_path}")
    return df, winner


def _sensor_cols_for(variant_name: str):
    return config.ALL_21_SENSOR_COLUMNS if variant_name == "21_sensors" else config.SELECTED_SENSOR_COLUMNS


def run_teacher_arch_sweep(subsets, device: str, window_size_by_subset: dict,
                            sensor_columns_by_subset: dict, epochs: int = None):
    all_rows = []
    for subset in subsets:
        ws = window_size_by_subset[subset]
        cols = sensor_columns_by_subset[subset]
        input_dim = config.compute_input_dim(sensor_columns=cols)
        suffix = "_archsweep"
        process_subset(subset, window_size=ws, sensor_columns=cols, suffix=suffix)
        train_loader, val_loader, test_loader = get_dataloaders(subset, config.TEACHER_CFG.batch_size, suffix=suffix)

        for variant in TEACHER_ARCH_GRID:
            overrides = {k: v for k, v in variant.items() if k != "name"}
            if epochs is not None:
                overrides["epochs"] = epochs
            t_cfg = config.get_teacher_config(subset, window_size=ws, input_dim=input_dim, **overrides)
            model, test_rmse, test_score = train_teacher_model(
                subset, device, teacher_cfg=t_cfg, verbose=False, log_prefix=f"arch-sweep:{variant['name']}",
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            )
            prof = profile_model(model, ws, input_dim, device)
            print(f"[{subset}][arch-sweep] {variant['name']}: rmse={test_rmse:.3f} params={prof['params']}")
            all_rows.append({
                "subset": subset, "variant": variant["name"],
                "d_model": t_cfg.d_model, "n_heads": t_cfg.n_heads, "n_layers": t_cfg.n_layers, "d_ff": t_cfg.d_ff,
                "test_rmse": test_rmse, "test_score": test_score, **prof,
            })

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(config.RESULTS_DIR, "fd_mastery_teacher_arch_sweep.csv")
    df.to_csv(out_path, index=False)
    save_config_snapshot(out_path, extra={"subsets": subsets, "grid": TEACHER_ARCH_GRID})
    for subset in subsets:
        sub_df = df[df["subset"] == subset]
        winner = sub_df.loc[sub_df["test_rmse"].idxmin()]
        print(f"[{subset}][arch-sweep] WINNER: {winner['variant']} (rmse={winner['test_rmse']:.3f})")
    print(f"saved -> {out_path}")
    return df


def run_batch_size_sweep(subsets, device: str, window_size_by_subset: dict, sensor_columns_by_subset: dict,
                          arch_overrides: dict = None, epochs: int = None):
    arch_overrides = arch_overrides or {}
    rows = []
    for subset in subsets:
        ws = window_size_by_subset[subset]
        cols = sensor_columns_by_subset[subset]
        input_dim = config.compute_input_dim(sensor_columns=cols)
        suffix = "_bssweep"
        process_subset(subset, window_size=ws, sensor_columns=cols, suffix=suffix)
        arch = arch_overrides.get(subset, {})

        for bs in BATCH_SIZE_SWEEP:
            t_cfg = config.get_teacher_config(subset, window_size=ws, input_dim=input_dim, batch_size=bs, **arch)
            if epochs is not None:
                t_cfg = dataclasses.replace(t_cfg, epochs=epochs)
            train_loader, val_loader, test_loader = get_dataloaders(subset, bs, suffix=suffix)
            _, test_rmse, test_score = train_teacher_model(
                subset, device, teacher_cfg=t_cfg, verbose=False, log_prefix=f"bs-sweep:{bs}",
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            )
            print(f"[{subset}][bs-sweep] batch_size={bs}: rmse={test_rmse:.3f} score={test_score:.1f}")
            rows.append({"subset": subset, "batch_size": bs, "test_rmse": test_rmse, "test_score": test_score})

    df = pd.DataFrame(rows)
    out_path = os.path.join(config.RESULTS_DIR, "fd_mastery_batch_size_sweep.csv")
    df.to_csv(out_path, index=False)
    save_config_snapshot(out_path, extra={"subsets": subsets, "batch_sizes": BATCH_SIZE_SWEEP, "arch_overrides": arch_overrides})
    for subset in subsets:
        sub_df = df[df["subset"] == subset]
        winner = sub_df.loc[sub_df["test_rmse"].idxmin()]
        print(f"[{subset}][bs-sweep] WINNER: batch_size={int(winner['batch_size'])} (rmse={winner['test_rmse']:.3f})")
    print(f"saved -> {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for every run (quick pass)")
    parser.add_argument("--skip-window", action="store_true")
    parser.add_argument("--skip-sensors", action="store_true")
    parser.add_argument("--skip-arch", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    args = parser.parse_args()

    window_size_by_subset = {"FD001": config.get_window_size("FD001")}
    sensor_columns_by_subset = {"FD001": config.get_sensor_columns("FD001")}

    if not args.skip_window:
        _, winner_ws = run_window_sweep("FD002", args.device, epochs=args.epochs)
        window_size_by_subset["FD002"] = winner_ws
    else:
        window_size_by_subset["FD002"] = config.get_window_size("FD002")

    if not args.skip_sensors:
        _, winner_sensor = run_sensor_sweep("FD002", args.device, window_size_by_subset["FD002"], epochs=args.epochs)
        sensor_columns_by_subset["FD002"] = _sensor_cols_for(winner_sensor)
    else:
        sensor_columns_by_subset["FD002"] = config.get_sensor_columns("FD002")

    arch_overrides = {}
    if not args.skip_arch:
        arch_df = run_teacher_arch_sweep(["FD001", "FD002"], args.device,
                                          window_size_by_subset, sensor_columns_by_subset, epochs=args.epochs)
        for subset in ["FD001", "FD002"]:
            sub_df = arch_df[arch_df["subset"] == subset]
            winner = sub_df.loc[sub_df["test_rmse"].idxmin()]
            arch_overrides[subset] = {"d_model": int(winner["d_model"]), "n_heads": int(winner["n_heads"]),
                                       "n_layers": int(winner["n_layers"]), "d_ff": int(winner["d_ff"])}

    if not args.skip_batch:
        run_batch_size_sweep(["FD001", "FD002"], args.device, window_size_by_subset,
                              sensor_columns_by_subset, arch_overrides=arch_overrides, epochs=args.epochs)

    print("\n=== FD001/FD002 mastery sweep -- final settings determined ===")
    for subset in ["FD001", "FD002"]:
        print(f"  {subset}: window={window_size_by_subset[subset]}, "
              f"n_sensors={len(sensor_columns_by_subset[subset])}, "
              f"arch={arch_overrides.get(subset, 'unchanged (skipped)')}")


if __name__ == "__main__":
    main()
