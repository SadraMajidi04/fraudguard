"""Drift monitoring with the Population Stability Index (PSI).

Fraudsters adapt and customers change, so a model decays. PSI compares the distribution of
a feature at training time with its distribution in live traffic.
Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, >= 0.25 major shift (investigate/retrain).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.features import FEATURE_COLUMNS


def _status(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "major"


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI between a reference sample and a new sample. NaN gets its own bin."""
    e = np.asarray(expected, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    if e.size == 0 or a.size == 0:
        raise ValueError("both samples must be non-empty")
    if np.isinf(e).any() or np.isinf(a).any():
        raise ValueError("samples must not contain infinite values")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    ef, af = e[~np.isnan(e)], a[~np.isnan(a)]
    if ef.size:
        edges = np.unique(np.quantile(ef, np.linspace(0, 1, bins + 1)[1:-1]))
    else:
        edges = np.array([])
    # bins are (-inf, e1], (e1, e2], ..., (e_k, inf): a constant reference still has two sides
    n_bins = len(edges) + 1
    e_cnt = np.bincount(np.searchsorted(edges, ef, side="left"), minlength=n_bins).astype(float)
    a_cnt = np.bincount(np.searchsorted(edges, af, side="left"), minlength=n_bins).astype(float)
    e_cnt = np.r_[e_cnt, e.size - ef.size]  # NaN bin
    a_cnt = np.r_[a_cnt, a.size - af.size]
    pe = np.clip(e_cnt / e.size, eps, None)
    pa = np.clip(a_cnt / a.size, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features=None) -> pd.DataFrame:
    features = list(features) if features is not None else list(FEATURE_COLUMNS)
    for name, frame in (("reference", reference), ("current", current)):
        missing = [f for f in features if f not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing features: {missing}")
    rows = []
    for f in features:
        value = psi(reference[f], current[f])
        rows.append((f, value, _status(value), float(reference[f].isna().mean()),
                     float(current[f].isna().mean())))
    out = pd.DataFrame(rows, columns=["feature", "psi", "status", "nan_rate_ref", "nan_rate_cur"])
    return out.sort_values("psi", ascending=False, kind="stable").reset_index(drop=True)
