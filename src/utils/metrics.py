"""
Evaluation metrics for RUL prediction (Sec. 7.5 / Table 1).
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official C-MAPSS asymmetric scoring function (Saxena et al., 2008).

    Penalises late (over-estimated RUL / under-predicted degradation)
    predictions more heavily than early ones -- a late maintenance call is
    operationally worse than an early one.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))
