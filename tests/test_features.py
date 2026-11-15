import unittest

import numpy as np
import pandas as pd

from fraudguard.features import FEATURE_COLUMNS, build_features
from tests.helpers import make_df


def tx(tid, ts, cust="A", amount=10.0, channel="pos", country="US", home="US", device="d1",
       cat="grocery", merchant="M1"):
    return dict(transaction_id=tid, timestamp=pd.Timestamp(ts), customer_id=cust,
                merchant_id=merchant, merchant_category=cat, channel=channel, country=country,
                customer_home_country=home, device_id=device, amount=amount)


def frame(rows):
    return pd.DataFrame(rows)


class TestHandComputedExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [
            tx(1, "2025-01-01 10:00:00", amount=10),
            tx(2, "2025-01-01 10:30:00", amount=30, channel="online"),
            tx(3, "2025-01-01 10:30:00", amount=20, channel="online"),  # same second as #2
            tx(4, "2025-01-02 10:30:00", amount=50, channel="online", country="DE", device="d2"),
            tx(5, "2025-01-01 10:15:00", cust="B", amount=999, country="NG", device="b1"),
        ]
        cls.f = build_features(frame(rows)).set_index(pd.Index([1, 2, 3, 4, 5]))

    def test_first_transaction_has_no_history(self):
        r = self.f.loc[1]
        self.assertEqual(r["cust_prior_txn_count"], 0)
        self.assertTrue(np.isnan(r["cust_prior_mean_amount"]))
        self.assertTrue(np.isnan(r["amount_to_prior_mean"]))
        self.assertTrue(np.isnan(r["amount_to_prior_max"]))
        self.assertTrue(np.isnan(r["secs_since_prev_txn"]))
        self.assertEqual(r["txn_count_1h"], 0)
        self.assertEqual(r["amount_sum_24h"], 0)
        self.assertEqual(r["is_new_device"], 1)

    def test_second_transaction(self):
        r = self.f.loc[2]
        self.assertEqual(r["txn_count_1h"], 1)
        self.assertEqual(r["cust_prior_mean_amount"], 10)
        self.assertEqual(r["amount_to_prior_mean"], 3)
        self.assertEqual(r["secs_since_prev_txn"], 1800)
        self.assertEqual(r["is_new_device"], 0)
        self.assertEqual(r["is_online"], 1)

    def test_same_second_transactions_do_not_count_each_other_in_windows(self):
        r = self.f.loc[3]
        self.assertEqual(r["txn_count_1h"], 1)  # only #1; #2 is the same second
        self.assertEqual(r["cust_prior_txn_count"], 2)  # but order-based history sees #1, #2
        self.assertEqual(r["cust_prior_mean_amount"], 20)
        self.assertEqual(r["secs_since_prev_txn"], 0)

    def test_window_edges_and_foreign_flags(self):
        r = self.f.loc[4]
        self.assertEqual(r["txn_count_24h"], 2)  # #2 and #3 are exactly 24h old (included)
        self.assertEqual(r["amount_sum_24h"], 50)  # #1 is 24h30m old (excluded)
        self.assertEqual(r["txn_count_1h"], 0)
        self.assertEqual(r["cust_prior_txn_count"], 3)
        self.assertEqual(r["cust_prior_mean_amount"], 20)
        self.assertEqual(r["amount_to_prior_mean"], 2.5)
        self.assertAlmostEqual(r["amount_to_prior_max"], 50 / 30)
        self.assertEqual(r["secs_since_prev_txn"], 86400)
        self.assertEqual((r["is_foreign"], r["is_new_country"], r["is_new_device"]), (1, 1, 1))
        self.assertEqual((r["hour"], r["day_of_week"], r["is_night"]), (10, 3, 0))

    def test_customers_are_independent(self):
        r = self.f.loc[5]  # customer B: foreign, first time, nothing from customer A leaks in
        self.assertEqual(r["cust_prior_txn_count"], 0)
        self.assertEqual(r["txn_count_24h"], 0)
        self.assertEqual(r["amount_sum_24h"], 0)
        # NG differs from B's home (US) and it is B's first transaction there
        self.assertEqual((r["is_foreign"], r["is_new_country"]), (1, 1))


class TestNoLeakageAndInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_df(4000, 0.05)
        cls.full = build_features(cls.df)

    def test_columns_index_dtypes_and_finiteness(self):
        self.assertEqual(list(self.full.columns), FEATURE_COLUMNS)
        self.assertTrue(self.full.index.equals(self.df.index))
        self.assertTrue(all(dt == np.float64 for dt in self.full.dtypes))
        self.assertFalse(np.isinf(self.full.to_numpy()).any())

    def test_deleting_the_future_never_changes_a_row(self):
        for cut in (1, 2, 50, 777, 2500, 3999):
            with self.subTest(cut=cut):
                part = build_features(self.df.head(cut))
                pd.testing.assert_frame_equal(part, self.full.head(cut))

    def test_row_order_does_not_matter(self):
        shuffled = self.df.sample(frac=1.0, random_state=1)
        out = build_features(shuffled)
        pd.testing.assert_frame_equal(out.loc[self.df.index], self.full)

    def test_labels_are_never_used(self):
        flipped = self.df.copy()
        flipped["is_fraud"] = 1 - flipped["is_fraud"]
        pd.testing.assert_frame_equal(build_features(flipped), self.full)
        pd.testing.assert_frame_equal(build_features(self.df.drop(columns="is_fraud")), self.full)

    def test_works_with_non_default_and_duplicate_free_index(self):
        df = self.df.head(300).copy()
        df.index = np.arange(1000, 1300)
        out = build_features(df)
        pd.testing.assert_frame_equal(out, self.full.head(300).set_axis(df.index))

    def test_known_signal_exists(self):
        y = self.df["is_fraud"]
        # card-testing bursts => many transactions in the last hour on fraud rows
        self.assertGreater(self.full.loc[y == 1, "txn_count_1h"].mean(),
                           3 * self.full.loc[y == 0, "txn_count_1h"].mean())
        self.assertGreater(self.full.loc[y == 1, "is_foreign"].mean(), 0.4)


class TestInputValidation(unittest.TestCase):
    def base(self):
        return frame([tx(1, "2025-01-01 10:00:00"), tx(2, "2025-01-01 11:00:00")])

    def test_empty_frame_returns_empty_features(self):
        out = build_features(self.base().iloc[0:0])
        self.assertEqual(list(out.columns), FEATURE_COLUMNS)
        self.assertEqual(len(out), 0)

    def test_not_a_dataframe(self):
        with self.assertRaises(TypeError):
            build_features([1, 2])

    def test_missing_column(self):
        for col in ("amount", "timestamp", "customer_id", "transaction_id", "device_id"):
            with self.subTest(col=col), self.assertRaisesRegex(ValueError, "missing columns"):
                build_features(self.base().drop(columns=col))

    def test_nulls(self):
        df = self.base()
        df["device_id"] = df["device_id"].astype(object)
        df.loc[0, "device_id"] = None
        with self.assertRaisesRegex(ValueError, "null"):
            build_features(df)

    def test_bad_amounts(self):
        for bad in (0, -3.0, np.inf, np.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                df = self.base()
                df.loc[0, "amount"] = bad
                build_features(df)

    def test_text_amount(self):
        df = self.base()
        df["amount"] = df["amount"].astype(str)
        with self.assertRaises(ValueError):
            build_features(df)

    def test_timezone_aware_rejected(self):
        df = self.base()
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        with self.assertRaisesRegex(ValueError, "timezone"):
            build_features(df)

    def test_duplicate_transaction_ids(self):
        df = self.base()
        df.loc[1, "transaction_id"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate transaction_id"):
            build_features(df)

    def test_unknown_category_becomes_nan_not_a_crash(self):
        df = frame([tx(1, "2025-01-01 10:00:00", cat="space_tourism")])
        out = build_features(df)
        self.assertTrue(np.isnan(out.loc[0, "category_code"]))
        self.assertEqual(out.loc[0, "is_high_risk_category"], 0)

    def test_single_row(self):
        out = build_features(frame([tx(1, "2025-01-01 10:00:00")]))
        self.assertEqual(out.shape, (1, len(FEATURE_COLUMNS)))

    def test_many_rows_same_timestamp_are_deterministic(self):
        rows = [tx(i, "2025-01-01 10:00:00", amount=float(i)) for i in range(1, 8)]
        a = build_features(frame(rows))
        b = build_features(frame(rows[::-1])).loc[::-1].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a["cust_prior_txn_count"].tolist(), list(range(7)))

    def test_far_future_row_does_not_change_the_past(self):
        base = self.base()
        extra = pd.concat([base, frame([tx(3, "2030-01-01 00:00:00", amount=1e6)])],
                          ignore_index=True)
        pd.testing.assert_frame_equal(build_features(extra).head(2), build_features(base))


if __name__ == "__main__":
    unittest.main()
