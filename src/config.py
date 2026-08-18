"""
Central configuration for the RUL Knowledge Distillation project.
Edit values here rather than hard-coding them inside scripts.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# ---------------------------------------------------------------------------
# Dataset (NASA C-MAPSS)
# ---------------------------------------------------------------------------
# The 14 informative sensors commonly used in RUL literature
# (Zhang et al. 2016 / Li et al. 2018 selection — 21 raw sensors minus the
# 7 that are constant / near-constant and carry no degradation signal).
SELECTED_SENSOR_COLUMNS: List[int] = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
# Column layout of the raw *_train.txt / *_test.txt files (space separated):
# 0: unit_id, 1: cycle, 2-4: operating settings, 5-25: sensors 1-21
OP_SETTING_COLUMNS: List[int] = [2, 3, 4]

RUL_MAX_CAP = 125          # piecewise-linear RUL cap (Heimes 2008 convention)
WINDOW_SIZE = 30           # sliding-window length (time steps)
WINDOW_STRIDE = 1
VAL_SPLIT_RATIO = 0.2      # fraction of training units held out for validation
RANDOM_SEED = 42

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MULTI_CONDITION_SUBSETS = ["FD002", "FD004"]   # 6 operating conditions each
N_OPERATING_CONDITIONS = 6                      # cluster count for condition-wise normalization
MIN_CLUSTER_FRACTION = 0.02                     # merge clusters smaller than 2% of the training data
NORMALIZATION_CLIP = 10.0                       # clip normalized features to [-clip, +clip] as a safety net
GRAD_CLIP_NORM = 1.0                            # gradient-norm clipping during training (Transformer stability)

# Don't let a training window span a change in operating-condition cluster
# (only affects FD002/FD004). An external reference notebook found windows
# mixing two condition regimes to be a real contributor to its worst
# results -- but empirically checking this project's own FD002/FD004 data
# (see near-failure/condition diagnostic) shows the operating condition
# there changes almost every single cycle (mean run length ~1.2 cycles,
# max ~7, vs. a 30-cycle window), so a hard "never cross a boundary" rule
# drops essentially 100% of training windows and is not viable as a direct
# port of that notebook's fix. Left here (and enforced correctly in
# make_windows) so it can still be A/B tested, but default OFF for that
# reason -- condition-wise normalization is doing the real work instead.
NO_WINDOW_CONDITION_CROSSING = False

# Feed the operating-condition cluster id to the model as an explicit
# one-hot input (appended to every time step), instead of only using it to
# pick a normalization scaler and then discarding it. Off by default so it
# can be A/B tested against the current behaviour; flip to True to enable.
USE_CONDITION_FEATURE = False

# Cross-validation fold count (Sec. "Data" / "Teacher overfitting" of the
# project review -- a single fixed 80/20 split makes "best epoch" and
# reported RMSE partly a function of which units landed in that one split).
CV_N_FOLDS = 5


def compute_input_dim(use_condition_feature: bool = USE_CONDITION_FEATURE) -> int:
    """Number of per-time-step input features, including the optional
    operating-condition one-hot tag (see USE_CONDITION_FEATURE)."""
    base = len(SELECTED_SENSOR_COLUMNS) + len(OP_SETTING_COLUMNS)
    if use_condition_feature:
        base += N_OPERATING_CONDITIONS
    return base


@dataclass
class TeacherConfig:
    """Hyperparameters for the high-capacity Transformer teacher."""
    input_dim: int = field(default_factory=compute_input_dim)
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.2
    window_size: int = WINDOW_SIZE
    use_gru_head: bool = False   # hybrid Transformer+GRU teacher variant (Sec. 2)
    gru_hidden: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 150
    patience: int = 15         # early-stopping patience (epochs)
    warmup_epochs: int = 5     # linear LR warmup before cosine decay kicks in
    use_plateau_scheduler: bool = False  # ReduceLROnPlateau instead of cosine
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    augment_noise_std: float = 0.0   # Gaussian noise std added to training windows (0 = off)
    augment_mask_prob: float = 0.0   # per-timestep random masking probability (0 = off)


@dataclass
class StudentConfig:
    """Hyperparameters for the lightweight student.

    Deliberately much smaller than the teacher: fewer layers, fewer heads,
    a lower-dimensional embedding, and (optionally) a GRU in place of the
    feed-forward block to keep temporal modelling cheap. Tune these values
    during the Sec. 7.6 trade-off / compression sweep.
    """
    input_dim: int = field(default_factory=compute_input_dim)
    d_model: int = 16
    n_heads: int = 1
    n_layers: int = 1
    d_ff: int = 64
    dropout: float = 0.1
    window_size: int = WINDOW_SIZE
    use_gru_head: bool = True   # hybrid Transformer+GRU student (Sec. 7.3)
    gru_hidden: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 150
    patience: int = 15
    warmup_epochs: int = 5
    use_plateau_scheduler: bool = False
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    augment_noise_std: float = 0.0
    augment_mask_prob: float = 0.0


@dataclass
class DistillationConfig:
    """Weights for the combined KD loss (Sec. 7.4)."""
    alpha_task: float = 0.5      # weight on hard-label RMSE loss
    alpha_soft: float = 0.3      # weight on soft-target (teacher-output) loss
    alpha_feature: float = 0.2   # weight on intermediate feature-alignment loss
    temperature: float = 2.0     # softens the gap between teacher/student outputs
    adaptive_weighting: bool = False  # compute alpha_* dynamically from loss magnitudes instead of using the fixed values above


TEACHER_CFG = TeacherConfig()
STUDENT_CFG = StudentConfig()
KD_CFG = DistillationConfig()

# Seeds used for ensembling (Sec. 2 teacher ensemble, Sec. 6 student ensemble).
# RANDOM_SEED itself is always included as the first/primary member so a
# single-model run and ensemble member #0 stay reproducible against each other.
ENSEMBLE_SEEDS = [RANDOM_SEED, 43, 44]