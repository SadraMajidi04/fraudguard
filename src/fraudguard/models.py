"""Models behind one tiny interface: ``fit(X, y)`` and ``score(X)`` (higher = more suspicious).

* RuleScorer          - hand-written rules, what many banks run today (baseline)
* LogisticScorer      - linear baseline
* GradientBoostingScorer - strong tabular model (scikit-learn HistGradientBoosting)
* AutoencoderScorer   - neural-network anomaly detector trained on legitimate traffic only;
                        the score is the reconstruction error. Built with scikit-learn's MLP.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudguard.evaluation import pr_auc
from fraudguard.features import FEATURE_COLUMNS


class _BaseScorer:
    name = "base"
    needs_both_classes = True

    def _check_fit_inputs(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty frame")
        y = np.asarray(y)
        if y.shape != (len(X),):
            raise ValueError("y must be 1-D with one label per row of X")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain only 0/1")
        if self.needs_both_classes and len(np.unique(y)) < 2:
            raise ValueError("training labels contain a single class")
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing feature columns: {missing}")
        return y.astype(int)

    def _matrix(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing feature columns: {missing}")
        arr = X[self.feature_names_].to_numpy(dtype=float)
        if np.isinf(arr).any():
            raise ValueError("X contains infinite values")
        return arr

    def _remember_features(self):
        self.feature_names_ = list(FEATURE_COLUMNS)

    def score(self, X) -> np.ndarray:
        raise NotImplementedError


class RuleScorer(_BaseScorer):
    """Additive risk rules written by hand. Needs no training."""

    name = "rules"
    needs_both_classes = False

    def fit(self, X, y):
        self._check_fit_inputs(X, y)
        self._remember_features()
        return self

    def score(self, X) -> np.ndarray:
        d = pd.DataFrame(self._matrix(X), columns=self.feature_names_)
        s = (
            0.35 * (d["amount"] >= 200)
            + 0.25 * (d["is_foreign"] == 1)
            + 0.15 * (d["is_night"] == 1)
            + 0.15 * (d["txn_count_1h"] >= 3)
            + 0.10 * ((d["is_new_device"] == 1) & (d["is_online"] == 1))
        )
        return s.to_numpy(dtype=float)


class LogisticScorer(_BaseScorer):
    name = "logistic_regression"

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C, self.random_state = C, random_state

    def fit(self, X, y):
        y = self._check_fit_inputs(X, y)
        self._remember_features()
        self.pipe_ = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000,
                               random_state=self.random_state),
        )
        self.pipe_.fit(np.clip(self._matrix(X), -1e12, 1e12), y)
        return self

    def score(self, X) -> np.ndarray:
        return self.pipe_.predict_proba(np.clip(self._matrix(X), -1e12, 1e12))[:, 1]


class GradientBoostingScorer(_BaseScorer):
    name = "gradient_boosting"

    def __init__(self, learning_rate=0.1, max_iter=150, max_leaf_nodes=31,
                 l2_regularization=1.0, random_state: int = 42):
        self.learning_rate, self.max_iter = learning_rate, max_iter
        self.max_leaf_nodes, self.l2_regularization = max_leaf_nodes, l2_regularization
        self.random_state = random_state

    def get_params(self) -> dict:
        return dict(learning_rate=self.learning_rate, max_iter=self.max_iter,
                    max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization)

    def fit(self, X, y):
        y = self._check_fit_inputs(X, y)
        self._remember_features()
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization,
            class_weight="balanced", early_stopping=False, random_state=self.random_state,
        )
        self.model_.fit(self._matrix(X), y)  # NaN is handled natively
        return self

    def score(self, X) -> np.ndarray:
        return self.model_.predict_proba(self._matrix(X))[:, 1]


class AutoencoderScorer(_BaseScorer):
    """Learns to reconstruct NORMAL transactions; fraud reconstructs badly (high error)."""

    name = "autoencoder"
    needs_both_classes = False
    MIN_LEGIT_ROWS = 50

    def __init__(self, hidden=(16, 6, 16), max_iter=60, clip=8.0, random_state: int = 42):
        self.hidden, self.max_iter, self.clip = tuple(hidden), max_iter, clip
        self.random_state = random_state

    def _prep(self, X) -> np.ndarray:
        return np.clip(self.scaler_.transform(self.imputer_.transform(self._matrix(X))),
                       -self.clip, self.clip)

    def fit(self, X, y):
        y = self._check_fit_inputs(X, y)
        self._remember_features()
        legit = self._matrix(X)[y == 0]
        if len(legit) < self.MIN_LEGIT_ROWS:
            raise ValueError(f"need at least {self.MIN_LEGIT_ROWS} legitimate rows to train")
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(legit)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(legit))
        Z = np.clip(self.scaler_.transform(self.imputer_.transform(legit)), -self.clip, self.clip)
        self.net_ = MLPRegressor(
            hidden_layer_sizes=self.hidden, max_iter=self.max_iter, batch_size=256,
            random_state=self.random_state, n_iter_no_change=10, tol=1e-4,
        )
        with warnings.catch_warnings():  # a fixed epoch budget is intentional
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.net_.fit(Z, Z)
        return self

    def score(self, X) -> np.ndarray:
        Z = self._prep(X)
        return ((Z - self.net_.predict(Z)) ** 2).mean(axis=1)


DEFAULT_GBM_GRID = (
    dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=15),
    dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31),
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31),
)


def select_gradient_boosting(X_train, y_train, X_valid, y_valid, grid=DEFAULT_GBM_GRID,
                             random_state: int = 42):
    """Pick the configuration with the best VALIDATION PR-AUC (never the test set)."""
    if not grid:
        raise ValueError("grid must not be empty")
    best = None
    for params in grid:
        model = GradientBoostingScorer(random_state=random_state, **params).fit(X_train, y_train)
        value = pr_auc(y_valid, model.score(X_valid))
        if best is None or value > best[0]:
            best = (value, model, dict(params))
    return best[1], best[2], best[0]
