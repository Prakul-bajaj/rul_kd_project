"""
Teacher model: a standard multi-head self-attention Transformer encoder
for Remaining Useful Life prediction (Sec. 7.2).

Input:  (batch, window, input_dim) raw sensor/operating-setting window
Output: (batch, 1) predicted RUL
Also exposes the pooled encoder representation ("features") for use as a
distillation target by the student (Sec. 7.4).
"""

import torch
import torch.nn as nn

from src.config import TeacherConfig
from src.models.layers import PositionalEncoding, RULRegressionHead


class TeacherTransformer(nn.Module):
    def __init__(self, cfg: TeacherConfig):
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
            norm_first=True,   # Pre-LN: LayerNorm before each sublayer, not after --
                                # removes the need for careful warmup to avoid early
                                # training instability (Xiong et al. 2020)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        self.use_gru = getattr(cfg, "use_gru_head", False)
        if self.use_gru:
            # Hybrid attention+GRU variant (Sec. 2 of the project review):
            # the student already benefits from a small recurrent component
            # on top of attention; this tests whether the same helps a
            # pure-attention teacher train more reliably.
            self.gru = nn.GRU(
                input_size=cfg.d_model,
                hidden_size=cfg.gru_hidden,
                batch_first=True,
                bidirectional=False,
            )
            head_in = cfg.gru_hidden
        else:
            head_in = cfg.d_model

        self.head = RULRegressionHead(head_in, head_in // 2, cfg.dropout,
                                       output_activation=getattr(cfg, "output_activation", "none"))
        self._init_weights()

    def _init_weights(self):
        """Explicit Xavier-uniform init for every >=2D parameter, instead of
        relying on PyTorch's per-module defaults (which vary in scale
        across Linear/GRU/MultiheadAttention and aren't tuned as a set for
        this architecture)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        x: (batch, window, input_dim)
        """
        h = self.input_proj(x)                 # (B, W, d_model)
        h = self.pos_encoding(h)
        h = self.encoder(h)                    # (B, W, d_model)  -- feature map
        h = self.norm(h)

        if self.use_gru:
            gru_out, _ = self.gru(h)
            pooled = gru_out[:, -1, :]           # last time-step hidden state
        else:
            pooled = h.mean(dim=1)               # global average pool over time
        rul_pred = self.head(pooled)            # (B, 1)

        if return_features:
            # Both the full sequence feature map and the pooled embedding are
            # returned so the distillation loss can choose either level.
            return rul_pred, {"sequence": h, "pooled": pooled}
        return rul_pred


def build_teacher(cfg: TeacherConfig) -> TeacherTransformer:
    return TeacherTransformer(cfg)
