"""
Combined Knowledge Distillation loss for RUL regression (Sec. 7.4).

Three terms, matching the methodology description:
  1. Task loss      : RMSE between student prediction and ground-truth RUL
  2. Soft-target loss: RMSE between student prediction and a
                       temperature-softened teacher target -- since RUL is a
                       regression target there is no softmax distribution
                       to soften as in classification KD (Hinton et al.,
                       2015); instead the teacher's own continuous output
                       is blended toward the ground truth by `temperature`
                       to form an auxiliary "soft label" (see
                       `soft_target()` below).
  3. Feature loss   : RMSE between a linearly-projected student pooled
                       embedding and the teacher's pooled embedding,
                       aligning intermediate representations
                       (Patch-Embedding-Alignment idea, Liu et al. 2024).

L_total = alpha_task * L_task + alpha_soft * L_soft + alpha_feature * L_feature
(or, if cfg.adaptive_weighting: uncertainty-weighted combination, see
AdaptiveWeighting below -- Kendall et al. 2018, "Multi-Task Learning Using
Uncertainty to Weigh Losses").

FIX (project review, Sec. "Student training/distillation"): the original
implementation used RMSE for L_task but plain MSE for L_soft/L_feature,
so the nominal alpha weights (0.5/0.3/0.2) didn't reflect the actual
gradient contribution of each term -- MSE is quadratic in the residual
while RMSE is linear, so whichever term had the larger residual magnitude
dominated regardless of its alpha. All three terms now use the same RMSE
functional form so the alphas are directly comparable.

Also fixed: the original temperature scaling was
`mse(pred/t, teacher/t) * t**2`, which is algebraically identical to
`mse(pred, teacher)` for ANY t (the /t and *t**2 exactly cancel for a
plain squared-error loss -- this is not the classification-KD case where
temperature acts on a softmax nonlinearity first). Temperature therefore
had zero effect. It's now used to blend the teacher's prediction toward
the ground truth (`soft_target`), which is what the code's own comment
("softens how strongly the student chases it") actually described.
"""

import torch
import torch.nn as nn

from src.config import DistillationConfig


class FeatureProjector(nn.Module):
    """Learnable linear map bridging the student's embedding dim to the
    teacher's, so pooled features of different width can be compared."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(student_pooled)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(nn.functional.mse_loss(pred, target) + 1e-8)


def soft_target(teacher_pred: torch.Tensor, y_true: torch.Tensor, temperature: float) -> torch.Tensor:
    """Blend the teacher's prediction toward the ground truth as
    `temperature` increases: T=1 -> soft target is exactly the teacher's
    prediction; T->inf -> soft target approaches the ground truth (the
    teacher's signal is "softened" away). This is the regression analogue
    of classification-KD temperature (which has no direct equivalent for a
    plain squared-error regression loss, see module docstring)."""
    return y_true + (teacher_pred - y_true) / temperature


class AdaptiveWeighting(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall, Gal & Cipolla, 2018):
    each term gets a learnable log-variance `s_i`; the combined loss is
    sum_i [ exp(-s_i) * L_i + s_i ]. A term whose loss the model can't
    easily reduce gets down-weighted automatically (large s_i), instead of
    the relative weighting being frozen at whatever alpha_* was guessed
    before training started."""

    def __init__(self, n_terms: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_terms))

    def forward(self, losses: list) -> torch.Tensor:
        total = 0.0
        for i, l in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * l + self.log_vars[i]
        return total


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, cfg: DistillationConfig, student_dim: int, teacher_dim: int):
        super().__init__()
        self.cfg = cfg
        self.projector = FeatureProjector(student_dim, teacher_dim)
        self.adaptive = AdaptiveWeighting(n_terms=3) if cfg.adaptive_weighting else None

    def forward(self, student_pred, teacher_pred, y_true,
                student_feats: dict, teacher_feats: dict):
        # 1. hard-label task loss
        l_task = rmse_loss(student_pred, y_true)

        # 2. soft-target loss (temperature-blended teacher target, see
        #    soft_target() -- both terms now RMSE, same scale as l_task)
        target = soft_target(teacher_pred.detach(), y_true, self.cfg.temperature)
        l_soft = rmse_loss(student_pred, target)

        # 3. feature-alignment loss on the pooled embedding
        proj_student = self.projector(student_feats["pooled"])
        l_feature = rmse_loss(proj_student, teacher_feats["pooled"].detach())

        if self.adaptive is not None:
            total = self.adaptive([l_task, l_soft, l_feature])
        else:
            total = (
                self.cfg.alpha_task * l_task
                + self.cfg.alpha_soft * l_soft
                + self.cfg.alpha_feature * l_feature
            )
        return total, {
            "loss_task": l_task.item(),
            "loss_soft": l_soft.item(),
            "loss_feature": l_feature.item(),
            "loss_total": total.item(),
        }
