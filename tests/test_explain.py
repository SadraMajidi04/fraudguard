import unittest

import numpy as np
import pandas as pd

from fraudguard.explain import (
    compute_reference,
    describe_reason,
    permutation_importance,
    reason_codes,
)
from fraudguard.features import FEATURE_COLUMNS
from tests.helpers import fitted, prepared


class AmountOnlyScorer:
    """Toy model whose score is exactly the amount: makes the right answers obvious."""

    feature_names_ = list(FEATURE_COLUMNS)

    def score(self, X):
        return X["amount"].to_numpy(dtype=float)


def toy_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X["amount"] > 0.8).astype(int).to_numpy()
    return X, y


class TestPermutationImportance(unittest.TestCase):
    def test_only_the_used_feature_matters(self):
        X, y = toy_frame()
        imp = permutation_importance(AmountOnlyScorer(), X, y, n_repeats=3)
        self.assertEqual(imp.iloc[0]["feature"], "amount")
        self.assertGreater(imp.iloc[0]["importance_mean"], 0.3)
        others = imp[imp["feature"] != "amount"]["importance_mean"]
        self.assertTrue((others == 0).all())

    def test_sorted_descending_and_complete(self):
        X, y = toy_frame()
        imp = permutation_importance(AmountOnlyScorer(), X, y)
        self.assertEqual(set(imp["feature"]), set(FEATURE_COLUMNS))
        self.assertTrue(imp["importance_mean"].is_monotonic_decreasing)

    def test_deterministic_and_non_mutating(self):
        X, y = toy_frame()
        before = X.copy(deep=True)
        a = permutation_importance(AmountOnlyScorer(), X, y, seed=5)
        b = permutation_importance(AmountOnlyScorer(), X, y, seed=5)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(X, before)

    def test_feature_subset_and_errors(self):
        X, y = toy_frame()
        imp = permutation_importance(AmountOnlyScorer(), X, y, features=["amount", "hour"])
        self.assertEqual(len(imp), 2)
        with self.assertRaises(ValueError):
            permutation_importance(AmountOnlyScorer(), X, y, features=["nope"])
        with self.assertRaises(ValueError):
            permutation_importance(AmountOnlyScorer(), X, y, n_repeats=0)

    def test_real_model_relies_on_meaningful_features(self):
        _, X, y, sp = prepared()
        imp = permutation_importance(fitted("gbm"), X[sp.valid], y[sp.valid], n_repeats=2)
        meaningful = {
            "amount", "cust_prior_mean_amount", "cust_prior_txn_count", "log_amount",
            "amount_to_prior_mean", "amount_to_prior_max", "is_foreign", "is_night", "hour",
            "txn_count_1h", "is_online", "category_code", "is_high_risk_category",
            "is_new_country", "is_new_device", "txn_count_24h", "secs_since_prev_txn", "is_atm",
        }
        self.assertIn("amount", set(imp["feature"].head(2)))
        self.assertTrue(set(imp["feature"].head(5)) <= meaningful)
        self.assertGreater(imp.iloc[0]["importance_mean"], 0.01)


class TestReasonCodes(unittest.TestCase):
    def setUp(self):
        X, _ = toy_frame()
        self.ref = compute_reference(X)
        self.row = X.iloc[0].copy()

    def test_single_driver_is_reported_with_exact_effect(self):
        self.row["amount"] = 500.0
        reasons = reason_codes(AmountOnlyScorer(), self.row, self.ref)
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0]["feature"], "amount")
        self.assertAlmostEqual(reasons[0]["score_increase"], 500.0 - self.ref["amount"])
        self.assertIn("transaction amount", reasons[0]["text"])

    def test_typical_transaction_has_no_reasons(self):
        self.assertEqual(reason_codes(AmountOnlyScorer(), self.ref.copy(), self.ref), [])

    def test_accepts_one_row_frame_and_respects_top_k(self):
        self.row["amount"] = 500.0
        frame = self.row.to_frame().T
        self.assertEqual(len(reason_codes(AmountOnlyScorer(), frame, self.ref, top_k=5)), 1)
        with self.assertRaises(ValueError):
            reason_codes(AmountOnlyScorer(), frame, self.ref, top_k=0)

    def test_missing_feature_is_reported(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            reason_codes(AmountOnlyScorer(), self.row.drop("amount"), self.ref)
        with self.assertRaisesRegex(ValueError, "missing"):
            reason_codes(AmountOnlyScorer(), self.row, self.ref.drop("amount"))

    def test_real_model_explains_a_fraud_with_sorted_positive_reasons(self):
        df, X, y, sp = prepared()
        model, ref = fitted("gbm"), compute_reference(X[sp.train])
        scores = model.score(X[sp.test])
        idx = np.flatnonzero(y[sp.test] == 1)
        best = idx[np.argmax(scores[idx])]
        row = X[sp.test].iloc[best]
        reasons = reason_codes(model, row, ref, top_k=3)
        self.assertGreaterEqual(len(reasons), 1)
        deltas = [r["score_increase"] for r in reasons]
        self.assertTrue(all(d > 0 for d in deltas))
        self.assertEqual(deltas, sorted(deltas, reverse=True))
        self.assertTrue(all(r["text"] for r in reasons))

    def test_missing_value_is_described_as_no_history(self):
        class NeedsHistory(AmountOnlyScorer):
            def score(self, X):
                return X["secs_since_prev_txn"].fillna(0).to_numpy(dtype=float)

        row = self.ref.copy()
        row["secs_since_prev_txn"] = 1000.0
        ref = self.ref.copy()
        reasons = reason_codes(NeedsHistory(), row, ref)
        self.assertEqual(reasons[0]["feature"], "secs_since_prev_txn")
        row["secs_since_prev_txn"] = np.nan
        self.assertEqual(reason_codes(NeedsHistory(), row, ref), [])


NO_HIST = "seconds since the customer's previous transaction: no history"


class TestDescribeReason(unittest.TestCase):
    def test_formats(self):
        cases = [
            (("is_foreign", 1.0), "made outside the customer's home country"),
            (("is_foreign", 0.0), "NOT made outside the customer's home country"),
            (("category_code", 6.0), "merchant category: electronics"),
            (("category_code", 99.0), "merchant category: 99"),
            (("amount", 1450.0), "transaction amount: 1,450.00"),
            (("amount_to_prior_mean", 7.25),
             "amount compared with the customer's usual amount: 7.2x"),
            (("txn_count_1h", 5.0), "number of transactions by this customer in the last hour: 5"),
            (("hour", 3.0), "hour of day: 3"),
            (("secs_since_prev_txn", None), NO_HIST),
            (("secs_since_prev_txn", float("nan")), NO_HIST),
            (("unknown_feature", 2.5), "unknown_feature: 2.5"),
        ]
        for (feat, val), expected in cases:
            with self.subTest(feature=feat, value=val):
                self.assertEqual(describe_reason(feat, val), expected)

    def test_every_feature_has_a_description(self):
        for f in FEATURE_COLUMNS:
            self.assertNotEqual(describe_reason(f, 1.0).split(":")[0], f)


class TestReference(unittest.TestCase):
    def test_median_and_all_nan_column(self):
        X, _ = toy_frame(101)
        X["hour"] = np.nan
        ref = compute_reference(X)
        self.assertEqual(ref["hour"], 0.0)
        self.assertAlmostEqual(ref["amount"], X["amount"].median())

    def test_empty_frame(self):
        with self.assertRaises(ValueError):
            compute_reference(pd.DataFrame(columns=FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
