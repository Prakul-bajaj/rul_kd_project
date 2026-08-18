"""
Config-snapshot utility (Engineering hygiene fix, Sec. 8 of the project
review): every time a results CSV is written, a sibling
`<same-name>.config.json` is written alongside it capturing every
hyperparameter that could plausibly explain "why does this number look
different from last time" -- teacher/student/KD config, preprocessing
config, and a timestamp. Without this, a re-run with changed config
silently overwrites the previous CSV with no record of what produced
either version.
"""

import dataclasses
import json
import os
import time

from src import config


def _current_config_dict() -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teacher_cfg": dataclasses.asdict(config.TEACHER_CFG),
        "student_cfg": dataclasses.asdict(config.STUDENT_CFG),
        "kd_cfg": dataclasses.asdict(config.KD_CFG),
        "preprocessing": {
            "window_size": config.WINDOW_SIZE,
            "window_stride": config.WINDOW_STRIDE,
            "rul_max_cap": config.RUL_MAX_CAP,
            "val_split_ratio": config.VAL_SPLIT_RATIO,
            "random_seed": config.RANDOM_SEED,
            "n_operating_conditions": config.N_OPERATING_CONDITIONS,
            "min_cluster_fraction": config.MIN_CLUSTER_FRACTION,
            "normalization_clip": config.NORMALIZATION_CLIP,
            "no_window_condition_crossing": getattr(config, "NO_WINDOW_CONDITION_CROSSING", None),
            "use_condition_feature": getattr(config, "USE_CONDITION_FEATURE", None),
        },
        "grad_clip_norm": config.GRAD_CLIP_NORM,
    }


def save_config_snapshot(csv_path: str, extra: dict = None) -> str:
    """Write a JSON snapshot of the current config next to `csv_path`.

    e.g. results/comparison_FD001.csv -> results/comparison_FD001.config.json

    `extra` can carry anything specific to the run that isn't already part
    of src.config (e.g. sweep-variant overrides, CV fold count).
    """
    base, _ = os.path.splitext(csv_path)
    out_path = base + ".config.json"
    payload = _current_config_dict()
    if extra:
        payload["extra"] = extra
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path
