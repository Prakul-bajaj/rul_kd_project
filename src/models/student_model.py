"""
Student model: a compressed Transformer (fewer layers/heads, smaller
embedding) with an optional GRU fusion head, per Sec. 7.3.

Designed to expose a feature representation of the *same dimensionality
convention* as the teacher (a "sequence" and a "pooled" tensor) so that
distillation.kd_loss can compare them directly, projecting through a
learnable linear layer where dimensions differ (mirrors the Patch
Embedding Alignment idea in Liu et al., TransKD, 2024).
"""

import torch
import torch.nn as nn

from src.config import StudentConfig
from src.models.layers import PositionalEncoding, RULRegressionHead


class StudentTransformer(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_encoding = PositionalEncoding(cfg.d_model, max_len=cfg.window_size + 8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        self.use_gru = cfg.use_gru_head
        if self.use_gru:
            # Hybrid Transformer+GRU: GRU compensates for the reduced
            # attention capacity by cheaply reinforcing temporal order.
            self.gru = nn.GRU(
                input_size=cfg.d_model,
                hidden_size=cfg.gru_hidden,
                batch_first=True,
                bidirectional=False,
            )
            head_in = cfg.gru_hidden
        else:
            head_in = cfg.d_model

        self.head = RULRegressionHead(head_in, max(head_in // 2, 8), cfg.dropout)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)                     # (B, W, d_model) -- feature map
        h = self.norm(h)

        if self.use_gru:
            gru_out, _ = self.gru(h)
            pooled = gru_out[:, -1, :]           # last time-step hidden state
        else:
            pooled = h.mean(dim=1)

        rul_pred = self.head(pooled)

        if return_features:
            return rul_pred, {"sequence": h, "pooled": pooled}
        return rul_pred


def build_student(cfg: StudentConfig) -> StudentTransformer:
    return StudentTransformer(cfg)
