"""End-to-end experiment: features -> time split -> 4 models -> cost-based evaluation -> bundle.

Methodology (each point prevents a classic mistake):
  * features use only the past                       (no time leakage)
  * train < validation < test in TIME                (no random-split optimism)
  * hyper-parameters and thresholds use VALIDATION   (the test set is touched once)
  * the winner is chosen by validation cost, not test
  * decisions are judged in money, not accuracy
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from fraudguard import __version__
from fraudguard.config import PATTERN_COLUMN, RANDOM_SEED, TARGET_COLUMN
from fraudguard.data.validation import validate_transactions
from fraudguard.evaluation import (
    CostModel,
    bootstrap_ci,
    evaluate_at_threshold,
    optimal_threshold,
    pr_auc,
)
from fraudguard.explain import compute_reference
from fraudguard.features import FEATURE_COLUMNS, build_features
from fraudguard.models import (
    DEFAULT_GBM_GRID,
    AutoencoderScorer,
    LogisticScorer,
    RuleScorer,
    select_gradient_boosting,
)
from fraudguard.splits import time_split

BUNDLE_VERSION = 1
_REQUIRED_BUNDLE_KEYS = (
    "bundle_version", "model_name", "model", "threshold", "feature_columns",
    "reference", "reference_features", "cost_model", "metrics",
)


@dataclass
class ExperimentConfig:
    train_frac: float = 0.6
    valid_frac: float = 0.2
    alert_budget: float = 0.01
    cost_model: CostModel = field(default_factory=CostModel)
    seed: int = RANDOM_SEED
    gbm_grid: tuple = DEFAULT_GBM_GRID
    autoencoder_epochs: int = 60
    n_bootstrap: int = 200
    reference_sample: int = 5000


@dataclass
class ExperimentResult:
    metrics: dict
    bundle: dict
    scorers: dict


def to_jsonable(obj):
    """Make nested results JSON-safe (numpy types, NaN/inf -> None, timestamps -> ISO)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, dt.datetime)):
        return obj.isoformat()
    return obj


def run_experiment(df: pd.DataFrame, config: ExperimentConfig | None = None) -> ExperimentResult:
    cfg = config or ExperimentConfig()
    validate_transactions(df)
    X = build_features(df)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)
    amount = df["amount"].to_numpy(dtype=float)
    sp = time_split(df, cfg.train_frac, cfg.valid_frac)
    tr, va, te = sp.train, sp.valid, sp.test

    scorers = {
        "rules": RuleScorer().fit(X[tr], y[tr]),
        "logistic_regression": LogisticScorer(random_state=cfg.seed).fit(X[tr], y[tr]),
        "autoencoder": AutoencoderScorer(max_iter=cfg.autoencoder_epochs,
                                         random_state=cfg.seed).fit(X[tr], y[tr]),
    }
    gbm, gbm_params, gbm_valid_pr = select_gradient_boosting(
        X[tr], y[tr], X[va], y[va], grid=cfg.gbm_grid, random_state=cfg.seed)
    scorers["gradient_boosting"] = gbm

    models = {}
    for name, scorer in scorers.items():
        s_va, s_te = scorer.score(X[va]), scorer.score(X[te])
        op = optimal_threshold(y[va], s_va, amount[va], cfg.cost_model)
        test = evaluate_at_threshold(y[te], s_te, amount[te], op.threshold, cfg.cost_model,
                                     cfg.alert_budget)
        _, lo, hi = bootstrap_ci(pr_auc, y[te], s_te, n_boot=cfg.n_bootstrap, seed=cfg.seed)
        models[name] = {
            "valid": {"pr_auc": pr_auc(y[va], s_va), "cost": op.cost, "threshold": op.threshold,
                      "n_flagged": op.n_flagged},
            "test": test,
            "test_pr_auc_ci95": [lo, hi],
        }
    models["gradient_boosting"]["selected_params"] = gbm_params
    best = min(models, key=lambda m: models[m]["valid"]["cost"])  # chosen WITHOUT the test set

    metrics = {
        "data": {
            "n_rows": int(len(df)), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean()),
            "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum()),
            "valid_start": sp.valid_start, "test_start": sp.test_start,
        },
        "cost_model": {"review_cost": cfg.cost_model.review_cost,
                       "fraud_loss_rate": cfg.cost_model.fraud_loss_rate},
        "alert_budget": cfg.alert_budget,
        "best_model": best,
        "models": models,
    }
    metrics = to_jsonable(metrics)

    sample = X[tr].sample(n=min(cfg.reference_sample, int(tr.sum())), random_state=cfg.seed)
    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "model_name": best,
        "model": scorers[best],
        "threshold": float(models[best]["valid"]["threshold"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "reference": compute_reference(X[tr]),
        "reference_features": sample.reset_index(drop=True),
        "cost_model": metrics["cost_model"],
        "metrics": metrics,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": {"fraudguard": __version__, "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    return ExperimentResult(metrics=metrics, bundle=bundle, scorers=scorers)


def save_bundle(bundle: dict, path: str | Path) -> Path:
    missing = [k for k in _REQUIRED_BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"bundle is missing keys: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> dict:
    """Load a bundle. SECURITY: joblib uses pickle, so only load files you created/trust."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such model file: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("not a FraudGuard model bundle")
    missing = [k for k in _REQUIRED_BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"bundle is missing keys: {missing}")
    if bundle["bundle_version"] != BUNDLE_VERSION:
        raise ValueError(
            f"unsupported bundle version {bundle['bundle_version']} (expected {BUNDLE_VERSION})")
    if list(bundle["feature_columns"]) != list(FEATURE_COLUMNS):
        raise ValueError("bundle was trained with a different feature set; retrain the model")
    return bundle


def format_report(metrics: dict) -> str:
    """Markdown results table (test set, threshold chosen on validation)."""
    d, cm = metrics["data"], metrics["cost_model"]
    lines = [
        f"Rows: {d['n_rows']:,} ({d['n_fraud']:,} fraud, {d['fraud_rate']:.2%}); "
        f"train/valid/test = {d['n_train']:,}/{d['n_valid']:,}/{d['n_test']:,} (by time).",
        f"Cost model: alert = ${cm['review_cost']:g}, missed fraud = "
        f"{cm['fraud_loss_rate']:g} x amount. Alert budget for recall: "
        f"{metrics['alert_budget']:.2%} of traffic.",
        "",
        "| Model | PR-AUC [95% CI] | ROC-AUC | Recall @ budget | Recall | Precision "
        "| Cost | Savings vs approve-all |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics["models"].items():
        t, ci = m["test"], m["test_pr_auc_ci95"]
        star = " **(selected)**" if name == metrics["best_model"] else ""
        prec = "n/a" if t["precision"] is None else f"{t['precision']:.2f}"
        lines.append(
            f"| {name}{star} | {t['pr_auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {t['roc_auc']:.3f} "
            f"| {t['recall_at_budget']:.2f} | {t['recall']:.2f} | {prec} "
            f"| ${t['cost']:,.0f} | {t['savings_vs_approve_all']:.1%} |")
    first = next(iter(metrics["models"].values()))["test"]
    lines += ["", f"Approve-everything baseline cost on test: ${first['cost_approve_all']:,.0f}."]
    return "\n".join(lines)


def recall_by_pattern(df: pd.DataFrame, scorer, threshold: float, train_frac: float = 0.6,
                      valid_frac: float = 0.2) -> pd.DataFrame:
    """On the TEST period: which kinds of fraud does the deployed threshold catch / miss?

    ``df`` must contain the analysis-only ``fraud_pattern`` column (generate with
    ``include_pattern=True``). Uses the same time split as :func:`run_experiment`.
    """
    if PATTERN_COLUMN not in df.columns:
        raise ValueError(f"'{PATTERN_COLUMN}' column required (generate with include_pattern=True)")
    X = build_features(df)
    te = time_split(df, train_frac, valid_frac).test
    flagged = scorer.score(X[te]) >= threshold
    sub = df[te].assign(flagged=flagged)
    fraud = sub[sub[TARGET_COLUMN] == 1].copy()
    fraud["missed_amount"] = np.where(fraud["flagged"], 0.0, fraud["amount"])
    g = fraud.groupby(PATTERN_COLUMN)
    out = pd.DataFrame({
        "n_fraud": g.size(),
        "caught": g["flagged"].sum().astype(int),
        "missed_amount": g["missed_amount"].sum().astype(float),
    })
    out["recall"] = out["caught"] / out["n_fraud"]
    return out.sort_values("recall").reset_index().rename(columns={PATTERN_COLUMN: "pattern"})
