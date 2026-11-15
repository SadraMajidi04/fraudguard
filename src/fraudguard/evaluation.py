"""Metrics that matter for fraud: PR-AUC, recall at an alert budget, and money.

Accuracy is useless here (99% of traffic is legitimate). Decisions are judged by
expected cost: a missed fraud costs the money lost; every alert costs a review fee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class CostModel:
    review_cost: float = 5.0  # price of one alert (analyst time + customer friction)
    fraud_loss_rate: float = 1.0  # share of a missed fraud's amount that is lost

    def __post_init__(self):
        if not (np.isfinite(self.review_cost) and self.review_cost >= 0):
            raise ValueError("review_cost must be finite and >= 0")
        if not (np.isfinite(self.fraud_loss_rate) and self.fraud_loss_rate > 0):
            raise ValueError("fraud_loss_rate must be finite and > 0")


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float  # flag a transaction when score >= threshold (inf = flag nothing)
    n_flagged: int
    cost: float


def _prepare(y, score, amount=None):
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if y.ndim != 1 or score.shape != y.shape:
        raise ValueError("y and score must be 1-D arrays of equal length")
    if y.size == 0:
        raise ValueError("empty input")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only 0/1")
    if not np.isfinite(score).all():
        raise ValueError("score contains NaN or inf")
    y = y.astype(int)
    if amount is not None:
        amount = np.asarray(amount, dtype=float)
        if amount.shape != y.shape or not np.isfinite(amount).all() or (amount < 0).any():
            raise ValueError("amount must match y in length and be finite and >= 0")
    return y, score, amount


def _need_both_classes(y):
    if y.min() == y.max():
        raise ValueError("y contains a single class; metric is undefined")


def pr_auc(y, score) -> float:
    y, score, _ = _prepare(y, score)
    _need_both_classes(y)
    return float(average_precision_score(y, score))


def roc_auc(y, score) -> float:
    y, score, _ = _prepare(y, score)
    _need_both_classes(y)
    return float(roc_auc_score(y, score))


def recall_at_alert_budget(y, score, budget: float = 0.005) -> dict:
    """Flag the top ``budget`` share of transactions by score (at least 1) and report
    recall/precision: how fraud teams with limited analysts actually work."""
    if not (0 < budget <= 1):
        raise ValueError("budget must be in (0, 1]")
    y, score, _ = _prepare(y, score)
    k = max(1, int(np.ceil(budget * y.size)))
    top = np.argsort(-score, kind="stable")[:k]
    caught = int(y[top].sum())
    total = int(y.sum())
    return {
        "n_flagged": k,
        "recall": caught / total if total else float("nan"),
        "precision": caught / k,
    }


def total_cost(y, flagged, amount, cost_model: CostModel = CostModel()) -> float:
    y, _, amount = _prepare(y, np.zeros(len(y)), amount)
    flagged = np.asarray(flagged, dtype=bool)
    if flagged.shape != y.shape:
        raise ValueError("flagged must match y in length")
    missed = (y == 1) & ~flagged
    return float(
        cost_model.review_cost * flagged.sum()
        + cost_model.fraud_loss_rate * amount[missed].sum()
    )


def optimal_threshold(y, score, amount, cost_model: CostModel = CostModel()) -> OperatingPoint:
    """Threshold with the lowest total cost on the given data (tied scores stay together)."""
    y, score, amount = _prepare(y, score, amount)
    order = np.argsort(-score, kind="stable")
    s = score[order]
    loss = (cost_model.fraud_loss_rate * amount * y)[order]
    k = np.arange(y.size + 1)
    cost = cost_model.review_cost * k + loss.sum() - np.r_[0.0, np.cumsum(loss)]
    allowed = np.zeros(y.size + 1, dtype=bool)
    allowed[0] = allowed[-1] = True
    allowed[1:-1] = s[:-1] > s[1:]  # only cut between different scores
    best = int(np.flatnonzero(allowed)[np.argmin(cost[allowed])])
    thr = float("inf") if best == 0 else float(s[best - 1])
    return OperatingPoint(threshold=thr, n_flagged=best, cost=float(cost[best]))


def evaluate_at_threshold(
    y, score, amount, threshold: float, cost_model: CostModel = CostModel(), budget: float = 0.005
) -> dict:
    """Full report on (typically held-out) data for a threshold chosen elsewhere."""
    y, score, amount = _prepare(y, score, amount)
    _need_both_classes(y)
    flagged = score >= threshold
    tp = int((flagged & (y == 1)).sum())
    n_fraud = int(y.sum())
    cost = total_cost(y, flagged, amount, cost_model)
    approve_all = total_cost(y, np.zeros_like(flagged), amount, cost_model)
    fraud_amt = float(amount[y == 1].sum())
    return {
        "n": int(y.size),
        "n_fraud": n_fraud,
        "pr_auc": pr_auc(y, score),
        "roc_auc": roc_auc(y, score),
        "alert_budget": budget,
        "recall_at_budget": recall_at_alert_budget(y, score, budget)["recall"],
        "threshold": float(threshold),
        "flag_rate": float(flagged.mean()),
        "recall": tp / n_fraud,
        "precision": tp / int(flagged.sum()) if flagged.any() else float("nan"),
        "caught_fraud_amount_share": float(amount[flagged & (y == 1)].sum() / fraud_amt)
        if fraud_amt else float("nan"),
        "cost": cost,
        "cost_approve_all": approve_all,
        "savings_vs_approve_all": (approve_all - cost) / approve_all if approve_all else 0.0,
    }


def bootstrap_ci(metric, y, score, n_boot: int = 200, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for ``metric(y, score)``. Resamples with both classes present."""
    y, score, _ = _prepare(y, score)
    _need_both_classes(y)
    if not (0 < alpha < 1) or n_boot < 10:
        raise ValueError("need 0 < alpha < 1 and n_boot >= 10")
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n_boot:
        idx = rng.integers(0, y.size, y.size)
        if y[idx].min() != y[idx].max():
            vals.append(metric(y[idx], score[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, score)), float(lo), float(hi)
