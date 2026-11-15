import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraudguard.evaluation import pr_auc, roc_auc
from fraudguard.features import FEATURE_COLUMNS
from fraudguard.models import (
    AutoencoderScorer,
    GradientBoostingScorer,
    LogisticScorer,
    RuleScorer,
    select_gradient_boosting,
)
from tests.helpers import fitted, prepared


class TestModelQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.X, cls.y, cls.sp = prepared()
        cls.Xv, cls.yv = cls.X[cls.sp.valid], cls.y[cls.sp.valid]
        cls.prev = cls.yv.mean()

    def metric(self, name):
        s = fitted(name).score(self.Xv)
        return pr_auc(self.yv, s), roc_auc(self.yv, s)

    def test_every_model_beats_random_by_a_wide_margin(self):
        for name in ("rules", "logreg", "gbm", "ae"):
            with self.subTest(model=name):
                pr, roc = self.metric(name)
                self.assertGreater(pr, 2 * self.prev)
                self.assertGreater(roc, 0.75)

    def test_learned_supervised_models_beat_hand_written_rules(self):
        rules = self.metric("rules")[0]
        self.assertGreater(self.metric("logreg")[0], rules)
        self.assertGreater(self.metric("gbm")[0], rules)

    def test_gradient_boosting_is_strong(self):
        self.assertGreater(self.metric("gbm")[0], 0.6)

    def test_scores_are_finite_and_one_per_row(self):
        for name in ("rules", "logreg", "gbm", "ae"):
            with self.subTest(model=name):
                s = fitted(name).score(self.Xv)
                self.assertEqual(s.shape, (len(self.Xv),))
                self.assertTrue(np.isfinite(s).all())

    def test_gbm_probabilities_are_in_unit_interval(self):
        s = fitted("gbm").score(self.Xv)
        self.assertTrue(((s >= 0) & (s <= 1)).all())


class TestModelRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.X, cls.y, cls.sp = prepared()
        cls.Xt = cls.X[cls.sp.test].head(300)

    def test_deterministic_training(self):
        _, X, y, sp = prepared()
        for cls in (LogisticScorer, GradientBoostingScorer, AutoencoderScorer):
            with self.subTest(model=cls.__name__):
                a = cls().fit(X[sp.train].head(2000), y[sp.train][:2000]).score(self.Xt)
                b = cls().fit(X[sp.train].head(2000), y[sp.train][:2000]).score(self.Xt)
                np.testing.assert_array_equal(a, b)

    def test_column_order_and_extra_columns_do_not_matter(self):
        for name in ("rules", "logreg", "gbm", "ae"):
            with self.subTest(model=name):
                m = fitted(name)
                shuffled = self.Xt[list(reversed(FEATURE_COLUMNS))].assign(junk=1.0)
                np.testing.assert_allclose(m.score(shuffled), m.score(self.Xt))

    def test_scoring_does_not_mutate_input(self):
        before = self.Xt.copy(deep=True)
        for name in ("rules", "logreg", "gbm", "ae"):
            fitted(name).score(self.Xt)
        pd.testing.assert_frame_equal(self.Xt, before)

    def test_missing_values_are_handled(self):
        Xn = self.Xt.copy()
        Xn.iloc[:, :] = np.nan  # worst case: nothing known
        for name in ("logreg", "gbm", "ae"):
            with self.subTest(model=name):
                self.assertTrue(np.isfinite(fitted(name).score(Xn)).all())
        Xn2 = self.Xt.copy()
        Xn2["secs_since_prev_txn"] = np.nan
        for name in ("logreg", "gbm", "ae"):
            self.assertEqual(len(fitted(name).score(Xn2)), len(Xn2))

    def test_infinite_values_are_rejected(self):
        Xi = self.Xt.copy()
        Xi.loc[Xi.index[0], "amount"] = np.inf
        for name in ("logreg", "gbm", "ae", "rules"):
            with self.subTest(model=name), self.assertRaises(ValueError):
                fitted(name).score(Xi)

    def test_missing_feature_column_is_rejected(self):
        for name in ("logreg", "gbm", "ae", "rules"):
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, "missing"):
                fitted(name).score(self.Xt.drop(columns="amount"))

    def test_non_dataframe_input_is_rejected(self):
        with self.assertRaises(TypeError):
            fitted("gbm").score(self.Xt.to_numpy())

    def test_extreme_values_do_not_crash(self):
        Xe = self.Xt.copy()
        Xe["amount"] = 1e12
        Xe["txn_count_1h"] = 1e6
        for name in ("rules", "logreg", "gbm", "ae"):
            with self.subTest(model=name):
                self.assertTrue(np.isfinite(fitted(name).score(Xe)).all())

    def test_joblib_round_trip_gives_identical_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("logreg", "gbm", "ae"):
                with self.subTest(model=name):
                    path = Path(tmp) / f"{name}.joblib"
                    joblib.dump(fitted(name), path)
                    back = joblib.load(path)
                    np.testing.assert_array_equal(back.score(self.Xt), fitted(name).score(self.Xt))


class TestFitValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.X, cls.y, cls.sp = prepared()

    def test_bad_training_inputs(self):
        X, y = self.X.head(500), self.y[:500]
        for cls in (LogisticScorer, GradientBoostingScorer, AutoencoderScorer):
            with self.subTest(model=cls.__name__):
                with self.assertRaises(TypeError):
                    cls().fit(X.to_numpy(), y)
                with self.assertRaises(ValueError):
                    cls().fit(X.iloc[0:0], y[:0])
                with self.assertRaises(ValueError):
                    cls().fit(X, y[:-1])  # length mismatch
                with self.assertRaises(ValueError):
                    cls().fit(X, np.where(y == 1, 2, 0))  # not 0/1
                with self.assertRaises(ValueError):
                    cls().fit(X.drop(columns="amount"), y)

    def test_supervised_models_reject_single_class(self):
        X = self.X.head(500)
        for cls in (LogisticScorer, GradientBoostingScorer):
            with self.subTest(model=cls.__name__):
                with self.assertRaisesRegex(ValueError, "single class"):
                    cls().fit(X, np.zeros(500, dtype=int))

    def test_autoencoder_needs_enough_legit_rows(self):
        with self.assertRaisesRegex(ValueError, "legitimate rows"):
            AutoencoderScorer().fit(self.X.head(30), np.zeros(30, dtype=int))
        with self.assertRaisesRegex(ValueError, "legitimate rows"):
            AutoencoderScorer().fit(self.X.head(200), np.ones(200, dtype=int))

    def test_autoencoder_trains_on_legit_rows_only(self):
        # Scores must not depend on the fraud rows' feature values at all.
        X, y = self.X.head(3000).copy(), self.y[:3000]
        a = AutoencoderScorer().fit(X, y).score(X.head(100))
        X2 = X.copy()
        X2.loc[X2.index[y == 1], "amount"] = 12345.0
        b = AutoencoderScorer().fit(X2, y).score(X.head(100))
        np.testing.assert_array_equal(a, b)


class TestRulesAndSelection(unittest.TestCase):
    def test_rule_scorer_values(self):
        row = {c: 0.0 for c in FEATURE_COLUMNS}
        X = pd.DataFrame([row] * 3)
        X.loc[1, ["amount", "is_foreign"]] = [500.0, 1.0]
        X.loc[2, ["amount", "is_foreign", "is_night", "txn_count_1h", "is_new_device",
                  "is_online"]] = [500.0, 1, 1, 5, 1, 1]
        s = RuleScorer().fit(X, [0, 1, 1]).score(X)
        np.testing.assert_allclose(s, [0.0, 0.60, 1.0])

    def test_selection_picks_best_validation_configuration(self):
        _, X, y, sp = prepared()
        grid = (dict(learning_rate=0.1, max_iter=2, max_leaf_nodes=4),
                dict(learning_rate=0.1, max_iter=80, max_leaf_nodes=15))
        model, params, value = select_gradient_boosting(
            X[sp.train], y[sp.train], X[sp.valid], y[sp.valid], grid=grid)
        self.assertEqual(params, grid[1])
        self.assertAlmostEqual(value, pr_auc(y[sp.valid], model.score(X[sp.valid])))

    def test_selection_rejects_empty_grid(self):
        _, X, y, sp = prepared()
        with self.assertRaises(ValueError):
            select_gradient_boosting(X[sp.train], y[sp.train], X[sp.valid], y[sp.valid], grid=())


if __name__ == "__main__":
    unittest.main()
