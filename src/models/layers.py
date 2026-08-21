"""
Shared building blocks used by both the teacher and student models.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class RULRegressionHead(nn.Module):
    """Two-layer MLP head mapping pooled features -> a scalar RUL prediction.

    output_activation lets the final layer be constrained to strictly
    positive output ("softplus" or "relu") instead of an unconstrained
    linear output -- RUL is a cycles-remaining quantity that's never
    negative by construction (clipped at 0 in preprocessing), so nothing
    stops the current linear head from predicting a negative RUL. Kept
    toggleable ("none" by default) rather than switched on unconditionally,
    so it can be A/B'd against the existing unconstrained head instead of
    silently changing every existing checkpoint's architecture.
    """

    def __init__(self, d_in: int, d_hidden: int, dropout: float = 0.1,
                 output_activation: str = "none"):
        super().__init__()
        layers = [
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        ]
        if output_activation == "softplus":
            layers.append(nn.Softplus())
        elif output_activation == "relu":
            layers.append(nn.ReLU())
        elif output_activation != "none":
            raise ValueError(f"unknown output_activation: {output_activation!r}")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
