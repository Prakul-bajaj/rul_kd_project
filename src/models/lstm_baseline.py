"""
Plain LSTM baseline (no attention, no distillation) used purely as a
reference point in the final comparison table -- shows what a
conventional recurrent model achieves on the same data/split so the
teacher's and student's gains can be quantified against it.
"""

import torch
import torch.nn as nn

from src.models.layers import RULRegressionHead


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = RULRegressionHead(hidden_dim * 2, hidden_dim, dropout)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        out, _ = self.lstm(x)
        pooled = out[:, -1, :]
        pred = self.head(pooled)
        if return_features:
            return pred, {"sequence": out, "pooled": pooled}
        return pred
