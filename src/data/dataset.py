"""
PyTorch Dataset wrapping the .npz arrays produced by preprocessing.py.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src import config


class CMAPSSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()          # (N, window, features)
        self.y = torch.from_numpy(y).float().view(-1, 1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_split(subset: str):
    """Load the pre-processed arrays for one C-MAPSS subset."""
    path = os.path.join(config.PROCESSED_DATA_DIR, f"{subset}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.preprocessing "
            f"--subset {subset}` first."
        )
    data = np.load(path)
    return data


def get_dataloaders(subset: str, batch_size: int, num_workers: int = 0):
    """num_workers defaults to 0: CMAPSSDataset just indexes an
    already-in-memory tensor (no per-item disk I/O), so worker processes
    add pure overhead here -- especially on Windows, where the 'spawn'
    start method re-imports the whole process and respawns workers every
    epoch by default (no persistent_workers), which was dominating wall
    time on these small subsets far more than the actual GPU compute."""
    data = load_split(subset)
    train_ds = CMAPSSDataset(data["X_train"], data["y_train"])
    val_ds = CMAPSSDataset(data["X_val"], data["y_val"])
    test_ds = CMAPSSDataset(data["X_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader
