
"""Time series cross-validation utilities (expanding walk-forward)."""
from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple

class ExpandingWindowSplit:
    """Generates train/test indices for walk-forward validation.
    Parameters
    ----------
    n_splits : int
        Number of splits.
    test_size : int
        Number of samples in each test fold.
    initial_train_size : int
        Size of the initial training window.
    step_size : int | None
        Step to move the window; if None uses test_size.
    """
    def __init__(self, n_splits: int, test_size: int, initial_train_size: int, step_size: int | None = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.initial_train_size = initial_train_size
        self.step_size = step_size or test_size

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        start = self.initial_train_size
        for i in range(self.n_splits):
            train_end = start + i*self.step_size
            test_end = train_end + self.test_size
            if test_end > n_samples:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            yield train_idx, test_idx
