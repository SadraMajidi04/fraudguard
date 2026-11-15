"""Explainability without extra dependencies.

* permutation_importance - global: how much does PR-AUC drop when a feature is shuffled?
* reason_codes           - local: for ONE transaction, which features pushed the score up?
  Each feature is replaced by a typical (median) value and the score change is measured
  (occlusion). Works for any model; every blocked transaction gets human-readable reasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.evaluation import pr_auc
from fraudguard.features import CATEGORY_NAMES, FEATURE_COLUMNS, FEATURE_DESCRIPTIONS


def compute_reference(X: pd.DataFrame) -> pd.Series:
    """Typical value per feature (median; 0 when a column is entirely missing)."""
    if len(X) == 0:
        raise ValueError("cannot compute a reference from an empty frame")
    ref = X[FEATURE_COLUMNS].median(skipna=True)
    return ref.fillna(0.0)


def permutation_importance(scorer, X: pd.DataFrame, y, n_repeats: int = 3, seed: int = 0,
                           features=None) -> pd.DataFrame:
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    y = np.asarray(y)
    features = list(features) if features is not None else list(scorer.feature_names_)
    unknown = [f for f in features if f not in X.columns]
    if unknown:
        raise ValueError(f"unknown features: {unknown}")
    rng = np.random.default_rng(seed)
    base = pr_auc(y, scorer.score(X))
    rows = []
    for feat in features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[feat] = rng.permutation(Xp[feat].to_numpy())
            drops.append(base - pr_auc(y, scorer.score(Xp)))
        rows.append((feat, float(np.mean(drops)), float(np.std(drops))))
    out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_std"])
    return out.sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)


_MONEY = {"amount", "cust_prior_mean_amount", "amount_sum_24h"}
_RATIO = {"amount_to_prior_mean", "amount_to_prior_max"}


def describe_reason(feature: str, value) -> str:
    """One human-readable sentence fragment for a feature value."""
    desc = FEATURE_DESCRIPTIONS.get(feature, feature)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{desc}: no history"
    if feature.startswith("is_"):
        return desc if value >= 0.5 else f"NOT {desc}"
    if feature == "category_code":
        code = int(value)
        return f"{desc}: {CATEGORY_NAMES[code] if 0 <= code < len(CATEGORY_NAMES) else code}"
    if feature in _MONEY:
        return f"{desc}: {value:,.2f}"
    if feature in _RATIO:
        return f"{desc}: {value:.1f}x"
    return f"{desc}: {value:g}"


def reason_codes(scorer, x, reference: pd.Series, top_k: int = 3) -> list[dict]:
    """Top features that RAISED the risk score of one transaction (largest first)."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    row = x.iloc[0] if isinstance(x, pd.DataFrame) else x
    feats = list(scorer.feature_names_)
    missing = [f for f in feats if f not in row.index or f not in reference.index]
    if missing:
        raise ValueError(f"missing features: {missing}")
    base = row[feats].to_numpy(dtype=float)
    ref = reference[feats].to_numpy(dtype=float)
    M = np.tile(base, (len(feats) + 1, 1))
    for j in range(len(feats)):
        M[j + 1, j] = ref[j]
    s = scorer.score(pd.DataFrame(M, columns=feats))
    deltas = s[0] - s[1:]
    out = []
    for j in np.argsort(-deltas, kind="stable")[:top_k]:
        if deltas[j] <= 1e-12:
            break
        value = None if np.isnan(base[j]) else float(base[j])
        out.append({
            "feature": feats[j],
            "value": value,
            "score_increase": float(deltas[j]),
            "text": describe_reason(feats[j], value),
        })
    return out
