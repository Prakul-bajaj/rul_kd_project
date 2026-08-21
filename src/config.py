"""
Central configuration for the RUL Knowledge Distillation project.
Edit values here rather than hard-coding them inside scripts.
"""

import os
import dataclasses
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
# All 21 raw sensors, unfiltered -- Noot et al. 2025 (DAST, "P01") keep the
# full 21 for FD002/FD004 rather than the fixed 14-sensor list, on the
# grounds that a sensor constant under FD001's single operating condition
# may vary meaningfully once FD002's 6 conditions are mixed in. Used by
# SENSOR_COLUMNS_BY_SUBSET below to A/B against the fixed 14-sensor list.
ALL_21_SENSOR_COLUMNS: List[int] = list(range(1, 22))
# Column layout of the raw *_train.txt / *_test.txt files (space separated):
# 0: unit_id, 1: cycle, 2-4: operating settings, 5-25: sensors 1-21
OP_SETTING_COLUMNS: List[int] = [2, 3, 4]

RUL_MAX_CAP = 125          # piecewise-linear RUL cap (Heimes 2008 convention)
WINDOW_SIZE = 30           # sliding-window length (time steps) -- default/fallback for subsets not in WINDOW_SIZE_BY_SUBSET
WINDOW_STRIDE = 1

# ---------------------------------------------------------------------------
# Per-subset window size / sensor selection (FD001 vs. FD002 "mastery" pass)
# ---------------------------------------------------------------------------
# FINALIZED by src/fd_mastery_sweep.py (results/fd_mastery_window_sweep_FD002.csv,
# results/fd_mastery_sensor_test_FD002.csv): window=60 beat {30,40,50} by a
# clear margin (13.84 vs. 14.39/15.18/15.26 test RMSE), and 21 sensors beat
# 14 (13.25 vs. 13.84) -- both landing exactly on Noot et al. 2025 (DAST,
# "P01")'s own published choices for FD002/FD004, independently rediscovered
# here rather than assumed. FD001 stays at its literature-matched default
# (window=30, 14 sensors) per the same sweep (see module docstring history
# in fd_mastery_sweep.py for the sweep grid and full results).
WINDOW_SIZE_BY_SUBSET = {"FD001": 30, "FD002": 60}
SENSOR_COLUMNS_BY_SUBSET = {"FD001": SELECTED_SENSOR_COLUMNS, "FD002": ALL_21_SENSOR_COLUMNS}

# Teacher architecture, also finalized by src/fd_mastery_sweep.py
# (results/fd_mastery_teacher_arch_sweep.csv), swept independently per
# subset using each subset's own window/sensor settings above. The winners
# are genuinely different architectures -- FD001 (single condition, more
# temporal signal per window at 14 sensors) wants more capacity; FD002
# (6 conditions, but now much richer per-window input at window=60/21
# sensors) wants less. d_ff = 2x d_model throughout, per P01's ratio.
TEACHER_ARCH_BY_SUBSET = {
    "FD001": {"d_model": 96, "n_heads": 2, "n_layers": 3, "d_ff": 192},
    "FD002": {"d_model": 32, "n_heads": 2, "n_layers": 3, "d_ff": 64},
}
# Batch size: both subsets independently picked 256 (see
# results/fd_mastery_batch_size_sweep.csv) -- the smaller batch sizes P04
# reported success with did NOT transfer here; 256 (the project's original
# default, and P01's own choice) won clearly on both FD001 and FD002. No
# override dict needed -- TeacherConfig/StudentConfig.batch_size default
# (256) already matches the swept winner for both subsets.


def get_window_size(subset: str) -> int:
    return WINDOW_SIZE_BY_SUBSET.get(subset, WINDOW_SIZE)


def get_sensor_columns(subset: str) -> List[int]:
    return SENSOR_COLUMNS_BY_SUBSET.get(subset, SELECTED_SENSOR_COLUMNS)
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


def compute_input_dim(use_condition_feature: bool = USE_CONDITION_FEATURE,
                       sensor_columns: List[int] = None) -> int:
    """Number of per-time-step input features, including the optional
    operating-condition one-hot tag (see USE_CONDITION_FEATURE)."""
    sensor_columns = sensor_columns if sensor_columns is not None else SELECTED_SENSOR_COLUMNS
    base = len(sensor_columns) + len(OP_SETTING_COLUMNS)
    if use_condition_feature:
        base += N_OPERATING_CONDITIONS
    return base


def get_input_dim(subset: str, use_condition_feature: bool = USE_CONDITION_FEATURE) -> int:
    return compute_input_dim(use_condition_feature, sensor_columns=get_sensor_columns(subset))


@dataclass
class TeacherConfig:
    """Hyperparameters for the high-capacity Transformer teacher.

    d_model/n_heads/n_layers/d_ff/dropout/lr_schedule/patience below reflect
    the FD001/FD002 "mastery" literature pass (Noot et al. 2025 "P01" DAST,
    Cheng et al. 2025 "P03" A-DDF, Wang et al. 2025 "P04" TBiGNet):
      - d_model=64/n_heads=4/n_layers=2 matches DAST's own published Teacher
        capacity (the closest architectural match among the three); d_ff=128
        is a 2x expansion (not the previous 4x) to match every paper's much
        smaller reported parameter counts.
      - dropout=0.4 matches TBiGNet (P04, the closest-goal paper: efficient/
        lightweight RUL); A-DDF (P03) uses 0.5. Both are well above this
        project's previous 0.2, motivated by confirmed teacher overfitting
        (train loss falling while val RMSE rises/oscillates).
      - lr_schedule="warm_restarts" matches TBiGNet (P04) exactly
        (CosineAnnealingWarmRestarts, restart every 5 epochs, min lr 1e-5) --
        the only paper of the three both architecturally close AND using a
        restart-based schedule, specifically to escape the sharp overfit
        minima this project has directly observed.
      - patience=15 / epochs=150 matches TBiGNet (P04) exactly; reverted
        from an earlier ad-hoc patience=30 + plateau-scheduler experiment
        that was tested and found not to fix the underlying overfitting
        curve (see results/teacher_fix_check_FD001_FD003.txt).
    d_model/n_heads/n_layers themselves are further swept in
    src/fd_mastery_sweep.py around this literature-grounded starting point.
    """
    input_dim: int = field(default_factory=compute_input_dim)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.4
    window_size: int = WINDOW_SIZE
    use_gru_head: bool = False   # hybrid Transformer+GRU teacher variant (Sec. 2)
    gru_hidden: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 150
    patience: int = 15         # early-stopping patience (epochs) -- reverted to literature/original value
    warmup_epochs: int = 5     # linear LR warmup before the main schedule kicks in
    lr_schedule: str = "warm_restarts"  # "cosine" | "plateau" | "warm_restarts"
    warm_restart_t0: int = 5            # restart period (epochs), matches TBiGNet (P04)
    warm_restart_eta_min: float = 1e-5  # matches TBiGNet (P04)
    plateau_patience: int = 8           # used only if lr_schedule == "plateau"
    plateau_factor: float = 0.5
    augment_noise_std: float = 0.0   # Gaussian noise std added to training windows (0 = off)
    augment_mask_prob: float = 0.0   # per-timestep random masking probability (0 = off)
    output_activation: str = "none"  # "none" | "softplus" | "relu" -- see RULRegressionHead


@dataclass
class StudentConfig:
    """Hyperparameters for the lightweight student.

    Deliberately much smaller than the teacher: fewer layers, fewer heads,
    a lower-dimensional embedding, and (optionally) a GRU in place of the
    feed-forward block to keep temporal modelling cheap. Tune these values
    during the Sec. 7.6 trade-off / compression sweep.

    dropout=0.3 and lr_schedule="warm_restarts" mirror the same FD001/FD002
    literature pass as TeacherConfig (see its docstring for citations).
    """
    input_dim: int = field(default_factory=compute_input_dim)
    d_model: int = 16
    n_heads: int = 1
    n_layers: int = 1
    d_ff: int = 64
    dropout: float = 0.3
    window_size: int = WINDOW_SIZE
    use_gru_head: bool = True   # hybrid Transformer+GRU student (Sec. 7.3)
    gru_hidden: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 150
    patience: int = 15
    warmup_epochs: int = 5
    lr_schedule: str = "warm_restarts"
    warm_restart_t0: int = 5
    warm_restart_eta_min: float = 1e-5
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    augment_noise_std: float = 0.0
    augment_mask_prob: float = 0.0
    output_activation: str = "none"


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

# 10-run best/ensemble robustness protocol (Noot et al. 2025 "P01" and Wang
# et al. 2025 "P04" both use this exact protocol -- 10 independent training
# runs, keep the single best and/or ensemble the best 6-of-10 -- rather than
# formal k-fold, which none of the three FD001/FD002-relevant papers use).
MULTI_SEED_N_RUNS = 10
MULTI_SEED_ENSEMBLE_TOP_K = 6
MULTI_SEED_SEEDS = list(range(RANDOM_SEED, RANDOM_SEED + MULTI_SEED_N_RUNS))


def get_teacher_config(subset: str, **overrides) -> TeacherConfig:
    """A TeacherConfig with window_size/input_dim/architecture baked in for
    `subset` (see WINDOW_SIZE_BY_SUBSET / SENSOR_COLUMNS_BY_SUBSET /
    TEACHER_ARCH_BY_SUBSET), otherwise identical to the shared global
    TeacherConfig() defaults (dropout, lr_schedule, patience, etc. --
    those weren't found to need per-subset variation). Pass further
    dataclasses.replace-style overrides as kwargs (e.g. dropout=0.5) to
    go beyond even the finalized per-subset settings, e.g. for a sweep."""
    base = {
        "window_size": get_window_size(subset),
        "input_dim": get_input_dim(subset),
        **TEACHER_ARCH_BY_SUBSET.get(subset, {}),
    }
    base.update(overrides)
    return dataclasses.replace(TeacherConfig(), **base)


def get_student_config(subset: str, **overrides) -> StudentConfig:
    base = {
        "window_size": get_window_size(subset),
        "input_dim": get_input_dim(subset),
    }
    base.update(overrides)
    return dataclasses.replace(StudentConfig(), **base)