"""Shared test utilities. Datasets are cached: generation is deterministic."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from fraudguard.data import generate_transactions


@lru_cache(maxsize=4)
def _cached(n: int, rate: float, with_pattern: bool) -> pd.DataFrame:
    return generate_transactions(n, rate, include_pattern=with_pattern)


def make_df(n: int = 20_000, rate: float = 0.02, with_pattern: bool = False) -> pd.DataFrame:
    """Return a fresh copy so a test can never corrupt another test's data."""
    return _cached(n, rate, with_pattern).copy()


@lru_cache(maxsize=2)
def prepared(n: int = 12_000, rate: float = 0.04):
    """(df, X, y, split) with features built once on the full frame (leak-free by design)."""
    from fraudguard.features import build_features
    from fraudguard.splits import time_split

    df = _cached(n, rate, False)
    X = build_features(df)
    return df, X, df["is_fraud"].to_numpy(), time_split(df)


@lru_cache(maxsize=8)
def fitted(name: str):
    """A model fitted on the train part of ``prepared()``; models are never mutated by tests."""
    from fraudguard.models import (
        AutoencoderScorer,
        GradientBoostingScorer,
        LogisticScorer,
        RuleScorer,
    )

    _, X, y, sp = prepared()
    model = {"rules": RuleScorer, "logreg": LogisticScorer, "gbm": GradientBoostingScorer,
             "ae": AutoencoderScorer}[name]()
    return model.fit(X[sp.train], y[sp.train])


class FeatureEchoScorer:
    """Stub model whose score is simply one feature's value (makes serving tests exact)."""

    def __init__(self, feature: str = "amount"):
        from fraudguard.features import FEATURE_COLUMNS

        self.feature = feature
        self.feature_names_ = list(FEATURE_COLUMNS)

    def score(self, X):
        return X[self.feature].fillna(0.0).to_numpy(dtype=float)


def stub_bundle(feature: str = "amount", threshold: float = 100.0) -> dict:
    import pandas as pd

    from fraudguard.features import FEATURE_COLUMNS

    return {
        "bundle_version": 1, "model_name": f"echo:{feature}", "model": FeatureEchoScorer(feature),
        "threshold": threshold, "feature_columns": list(FEATURE_COLUMNS),
        "reference": pd.Series(0.0, index=FEATURE_COLUMNS), "reference_features": None,
        "cost_model": {}, "metrics": {},
    }


def small_config():
    from fraudguard.pipeline import ExperimentConfig

    return ExperimentConfig(
        gbm_grid=(dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=15),),
        autoencoder_epochs=5, n_bootstrap=20)
