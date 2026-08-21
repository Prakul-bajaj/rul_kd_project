"""
Shared training-process helpers (project review, Sec. "Teacher overfitting
& training process"): LR warmup, an optional validation-plateau-driven LR
schedule as an alternative to pure cosine decay, and light data
augmentation on training windows. Used by train_teacher.py,
train_student_kd.py (train_student), and train_student_no_kd.py so all
three share identical warmup/schedule/augmentation behaviour driven off the
same config fields (TeacherConfig / StudentConfig).
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau


class WarmupScheduler:
    """Linearly ramps the LR from 0 -> base_lr over `warmup_epochs`, then
    hands control to `post_scheduler` for every subsequent epoch.

    `post_scheduler` may be either a plain epoch-driven scheduler (call
    .step() with no args, e.g. CosineAnnealingLR) or a metric-driven one
    (ReduceLROnPlateau, call .step(val_metric)) -- `is_plateau` picks which
    calling convention to use once warmup is over.
    """

    def __init__(self, optimizer, base_lr: float, warmup_epochs: int, post_scheduler, is_plateau: bool):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = max(0, warmup_epochs)
        self.post_scheduler = post_scheduler
        self.is_plateau = is_plateau
        self._epoch = 0
        if self.warmup_epochs > 0:
            self._set_lr(self.base_lr / self.warmup_epochs)

    def _set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def step(self, val_metric=None):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            self._set_lr(self.base_lr * self._epoch / self.warmup_epochs)
        else:
            if self.is_plateau:
                self.post_scheduler.step(val_metric)
            else:
                self.post_scheduler.step()


def build_scheduler(optimizer, cfg):
    """cfg is a TeacherConfig or StudentConfig. `cfg.lr_schedule` selects the
    post-warmup schedule:
      "warm_restarts" (default, FD001/FD002 literature pass -- matches
        TBiGNet/P04's CosineAnnealingWarmRestarts exactly): periodically
        restarts the LR every warm_restart_t0 epochs instead of decaying
        once over the whole budget, to kick training out of the sharp
        overfit minima this project has directly observed.
      "plateau": ReduceLROnPlateau, decays only when val RMSE stalls.
      "cosine": single smooth decay over the remaining epoch budget
        (the project's original default).
    """
    schedule = getattr(cfg, "lr_schedule", "cosine")
    if schedule == "plateau":
        post = ReduceLROnPlateau(optimizer, mode="min",
                                  patience=cfg.plateau_patience, factor=cfg.plateau_factor)
        is_plateau = True
    elif schedule == "warm_restarts":
        post = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.warm_restart_t0, eta_min=cfg.warm_restart_eta_min,
        )
        is_plateau = False
    elif schedule == "cosine":
        remaining_epochs = max(1, cfg.epochs - cfg.warmup_epochs)
        post = CosineAnnealingLR(optimizer, T_max=remaining_epochs)
        is_plateau = False
    else:
        raise ValueError(f"unknown lr_schedule: {schedule!r}")
    return WarmupScheduler(optimizer, cfg.lr, cfg.warmup_epochs, post, is_plateau)


def augment_batch(X: torch.Tensor, cfg) -> torch.Tensor:
    """Light augmentation on a (B, window, features) training batch:
    additive Gaussian noise (augment_noise_std) and/or random per-timestep
    masking (augment_mask_prob). Both are no-ops (returns X unchanged) at
    their default 0.0 config values -- opt in via TeacherConfig/StudentConfig."""
    if cfg.augment_noise_std > 0:
        X = X + torch.randn_like(X) * cfg.augment_noise_std
    if cfg.augment_mask_prob > 0:
        keep = (torch.rand(X.shape[0], X.shape[1], 1, device=X.device) >= cfg.augment_mask_prob).float()
        X = X * keep
    return X
