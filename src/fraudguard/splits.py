"""Time-based train / validation / test split (train on the past, test on the future)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudguard.config import TARGET_COLUMN


@dataclass(frozen=True)
class TimeSplit:
    train: np.ndarray  # boolean masks aligned with the input frame's row order
    valid: np.ndarray
    test: np.ndarray
    valid_start: pd.Timestamp
    test_start: pd.Timestamp


def time_split(df: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2) -> TimeSplit:
    """Split by time. Every train row is strictly earlier than every valid row, which is
    strictly earlier than every test row. Each part must contain both classes."""
    if not (0 < train_frac < 1 and 0 < valid_frac < 1 and train_frac + valid_frac < 1):
        raise ValueError("need 0 < train_frac, valid_frac and train_frac + valid_frac < 1")
    n = len(df)
    if n < 30:
        raise ValueError(f"need at least 30 rows to split, got {n}")
    ts = df["timestamp"]
    sorted_ts = ts.sort_values(kind="stable").reset_index(drop=True)
    valid_start = sorted_ts.iloc[int(n * train_frac)]
    test_start = sorted_ts.iloc[int(n * (train_frac + valid_frac))]
    train = (ts < valid_start).to_numpy()
    valid = ((ts >= valid_start) & (ts < test_start)).to_numpy()
    test = (ts >= test_start).to_numpy()
    for name, mask in (("train", train), ("valid", valid), ("test", test)):
        if not mask.any():
            raise ValueError(f"{name} split is empty (timestamps too concentrated?)")
        if TARGET_COLUMN in df.columns and df.loc[mask, TARGET_COLUMN].nunique() < 2:
            raise ValueError(f"{name} split does not contain both classes; use more data")
    return TimeSplit(train, valid, test, valid_start, test_start)
