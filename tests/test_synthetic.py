import unittest

import numpy as np
import pandas as pd

from fraudguard.config import CASH_CATEGORY, CHANNELS, HIGH_RISK_CATEGORIES
from fraudguard.data import FRAUD_PATTERNS, generate_transactions, validate_transactions
from fraudguard.data import synthetic as syn
from tests.helpers import make_df


class TestProbabilityTables(unittest.TestCase):
    def test_all_probability_vectors_sum_to_one(self):
        for name in ("_HOME_COUNTRY_P", "_CATEGORY_P", "_LEGIT_CHANNEL_P", "_LEGIT_ATM_P"):
            with self.subTest(name=name):
                self.assertAlmostEqual(float(getattr(syn, name).sum()), 1.0)

    def test_table_lengths_match_config(self):
        self.assertEqual(len(syn._HOME_COUNTRY_P), len(syn._COUNTRIES))
        self.assertEqual(len(syn._CATEGORY_P), len(syn._CATEGORIES))
        self.assertEqual(len(syn._LEGIT_ATM_AMOUNTS), len(syn._LEGIT_ATM_P))
        self.assertAlmostEqual(sum(syn._PATTERN_WEIGHTS), 1.0)


class TestShapeAndCounts(unittest.TestCase):
    def test_exact_row_count_and_fraud_count(self):
        for n, rate in [(1000, 0.01), (5000, 0.05), (20_000, 0.02), (101, 0.1), (10, 0.2)]:
            with self.subTest(n=n, rate=rate):
                df = generate_transactions(n, rate)
                self.assertEqual(len(df), n)
                self.assertEqual(int(df["is_fraud"].sum()), round(n * rate))

    def test_tiny_fraud_counts_still_work(self):
        # 1, 2, 3 ... frauds: exercises pattern-splitting rounding edge cases
        for k in range(1, 12):
            with self.subTest(frauds=k):
                df = generate_transactions(100, k / 100, include_pattern=True)
                self.assertEqual(int(df["is_fraud"].sum()), k)
                self.assertEqual(len(df), 100)

    def test_output_passes_validation(self):
        report = validate_transactions(make_df(), raise_on_error=False)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.warnings, [])

    def test_transaction_ids_unique_and_sequential(self):
        df = make_df()
        self.assertTrue(df["transaction_id"].is_unique)
        self.assertEqual(df["transaction_id"].tolist(), list(range(1, len(df) + 1)))

    def test_sorted_by_time(self):
        self.assertTrue(make_df()["timestamp"].is_monotonic_increasing)

    def test_timestamps_inside_requested_window(self):
        df = generate_transactions(5000, 0.05, n_days=10, start_date="2024-03-01")
        self.assertGreaterEqual(df["timestamp"].min(), pd.Timestamp("2024-03-01"))
        self.assertLess(df["timestamp"].max(), pd.Timestamp("2024-03-11"))

    def test_pattern_column_hidden_by_default(self):
        self.assertNotIn("fraud_pattern", make_df().columns)
        self.assertIn("fraud_pattern", make_df(with_pattern=True).columns)

    def test_target_is_last_feature_column_order(self):
        self.assertEqual(make_df().columns[-1], "is_fraud")


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_data(self):
        a = generate_transactions(3000, 0.03, seed=7)
        b = generate_transactions(3000, 0.03, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_different_data(self):
        a = generate_transactions(3000, 0.03, seed=7)
        b = generate_transactions(3000, 0.03, seed=8)
        self.assertFalse(a["amount"].equals(b["amount"]))


class TestRealism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_df(with_pattern=True)
        cls.legit = cls.df[cls.df["fraud_pattern"] == "legit"]

    def by(self, pattern):
        return self.df[self.df["fraud_pattern"] == pattern]

    def test_all_patterns_present(self):
        self.assertEqual(set(self.df["fraud_pattern"]) - {"legit"}, set(FRAUD_PATTERNS))

    def test_pattern_matches_label(self):
        self.assertTrue((self.df["is_fraud"] == (self.df["fraud_pattern"] != "legit")).all())

    def test_amounts_positive_finite_two_decimals(self):
        amt = self.df["amount"]
        self.assertTrue(np.isfinite(amt).all())
        self.assertTrue((amt > 0).all())
        self.assertTrue(((amt * 100).round(6) % 1 < 1e-6).all())
        self.assertLessEqual(amt.max(), 20_000)

    def test_card_testing_amounts_tiny_and_online(self):
        ct = self.by("card_testing")
        self.assertLessEqual(ct["amount"].max(), 5.0)
        self.assertTrue((ct["channel"] == "online").all())

    def test_card_testing_comes_in_bursts(self):
        ct = self.by("card_testing").sort_values("timestamp")
        best = 0
        for _, g in ct.groupby("customer_id"):
            t = g["timestamp"]
            in_window = (t.shift(-2) - t).dt.total_seconds() <= 15 * 60  # 3 txns in 15 min
            best = max(best, int(in_window.sum()))
        self.assertGreater(best, 0, "no customer shows a 3-transactions-in-15-minutes burst")

    def test_takeover_large_online_risky_merchants_and_mostly_foreign(self):
        at = self.by("account_takeover")
        self.assertGreater(at["amount"].median(), 200)
        self.assertGreater(at["amount"].median(), 5 * self.legit["amount"].median())
        self.assertTrue((at["channel"] == "online").all())
        self.assertTrue(at["merchant_category"].isin(HIGH_RISK_CATEGORIES).all())
        self.assertGreater((at["country"] != at["customer_home_country"]).mean(), 0.6)

    def test_takeover_happens_at_night_more_than_legit(self):
        night = lambda d: d["timestamp"].dt.hour.lt(6).mean()  # noqa: E731
        self.assertGreater(night(self.by("account_takeover")), night(self.legit) + 0.3)

    def test_stealth_takeover_mimics_normal_behaviour(self):
        st = self.by("stealth_takeover")
        self.assertGreater(len(st), 20)
        self.assertGreater((st["country"] == st["customer_home_country"]).mean(), 0.9)
        self.assertLess(st["timestamp"].dt.hour.lt(6).mean(), 0.25)
        self.assertTrue((st["channel"] == "online").all())

    def test_pattern_shares_roughly_follow_weights(self):
        fraud = self.df[self.df["is_fraud"] == 1]["fraud_pattern"].value_counts(normalize=True)
        for pat, w in zip(FRAUD_PATTERNS, syn._PATTERN_WEIGHTS, strict=True):
            self.assertAlmostEqual(fraud[pat], w, delta=0.03)

    def test_cashout_is_atm_cash_and_large(self):
        co = self.by("atm_cashout")
        self.assertTrue((co["channel"] == "atm").all())
        self.assertTrue((co["merchant_category"] == CASH_CATEGORY).all())
        self.assertTrue(co["amount"].between(200, 1000).all())

    def test_legit_mostly_home_country(self):
        home = self.legit["country"] == self.legit["customer_home_country"]
        self.assertGreater(home.mean(), 0.9)

    def test_legit_atm_rows_use_cash_category_and_atm_channel_consistently(self):
        atm = self.legit["channel"] == "atm"
        self.assertTrue((self.legit.loc[atm, "merchant_category"] == CASH_CATEGORY).all())
        self.assertTrue((self.legit.loc[~atm, "merchant_category"] != CASH_CATEGORY).all())

    def test_legit_has_hard_overlap_with_fraud(self):
        # A fair benchmark must not be trivially separable: some legit rows are big,
        # foreign, online purchases too.
        hard = self.legit[
            (self.legit["amount"] > 150)
            & (self.legit["channel"] == "online")
        ]
        self.assertGreater(len(hard), 20)

    def test_channels_and_countries_in_allowed_sets(self):
        self.assertTrue(self.df["channel"].isin(CHANNELS).all())
        self.assertTrue(self.df["country"].isin(syn.COUNTRIES).all())


class TestArgumentValidation(unittest.TestCase):
    def test_bad_arguments_raise_value_error(self):
        cases = [
            dict(n_transactions=0), dict(n_transactions=1), dict(n_transactions=-5),
            dict(n_transactions=1000.5), dict(fraud_rate=0.0), dict(fraud_rate=1.0),
            dict(fraud_rate=-0.1), dict(fraud_rate=1.5), dict(n_customers=0),
            dict(n_merchants=0), dict(n_days=0), dict(start_date="not-a-date"),
            dict(start_date=None),
            dict(n_transactions=100, fraud_rate=0.001),  # rounds to 0 frauds
            dict(n_transactions=10, fraud_rate=0.99),  # no legit rows left
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    generate_transactions(**{"n_transactions": 1000, **kwargs})

    def test_extreme_but_legal_sizes(self):
        for kwargs in (dict(n_customers=1), dict(n_merchants=1), dict(n_days=1),
                       dict(n_merchants=3, n_customers=2)):
            with self.subTest(**kwargs):
                df = generate_transactions(2000, 0.05, **kwargs)
                self.assertTrue(validate_transactions(df, raise_on_error=False).ok)
                self.assertEqual(len(df), 2000)


if __name__ == "__main__":
    unittest.main()
