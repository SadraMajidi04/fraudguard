import unittest

import numpy as np
import pandas as pd

from fraudguard.features import FEATURE_COLUMNS
from fraudguard.monitoring import drift_report, psi
from tests.helpers import prepared


class TestPsi(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_identical_samples_give_zero(self):
        x = self.rng.normal(size=5000)
        self.assertAlmostEqual(psi(x, x), 0.0, places=9)

    def test_same_distribution_is_stable(self):
        a, b = self.rng.normal(size=20_000), self.rng.normal(size=20_000)
        self.assertLess(psi(a, b), 0.05)

    def test_shifted_distribution_is_major(self):
        a, b = self.rng.normal(size=5000), self.rng.normal(1.0, 1.0, 5000)
        self.assertGreater(psi(a, b), 0.25)

    def test_larger_shift_gives_larger_psi_and_never_negative(self):
        a = self.rng.normal(size=5000)
        values = [psi(a, self.rng.normal(s, 1, 5000)) for s in (0.0, 0.3, 0.8, 2.0)]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(v >= 0 for v in values))

    def test_scale_invariance(self):
        a, b = self.rng.normal(size=3000), self.rng.normal(0.5, 1, 3000)
        self.assertAlmostEqual(psi(a, b), psi(a * 1000, b * 1000), places=9)

    def test_nan_is_its_own_bin(self):
        a = self.rng.normal(size=4000)
        b = a.copy()
        b[:1200] = np.nan
        self.assertGreater(psi(a, b), 0.25)
        self.assertAlmostEqual(psi(b, b), 0.0, places=9)

    def test_all_nan_both_sides(self):
        x = np.full(100, np.nan)
        self.assertAlmostEqual(psi(x, x), 0.0, places=9)

    def test_constant_features(self):
        c = np.full(500, 3.0)
        self.assertAlmostEqual(psi(c, c), 0.0, places=9)
        self.assertGreater(psi(c, np.full(500, 9.0)), 0.25)

    def test_unseen_values_do_not_produce_nan_or_inf(self):
        v = psi(self.rng.normal(size=1000), self.rng.normal(100, 1, 1000))
        self.assertTrue(np.isfinite(v))

    def test_invalid_arguments(self):
        good = np.arange(10.0)
        for args in ((np.array([]), good), (good, np.array([])), (good, np.r_[good, np.inf])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                psi(*args)
        with self.assertRaises(ValueError):
            psi(good, good, bins=1)

    def test_accepts_pandas_series_and_lists(self):
        self.assertAlmostEqual(psi(pd.Series([1.0, 2, 3, 4]), [1.0, 2, 3, 4], bins=2), 0.0)


class TestDriftReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, X, _, sp = prepared()
        cls.ref, cls.cur = X[sp.train], X[sp.test]

    def test_structure_and_sorting(self):
        rep = drift_report(self.ref, self.cur)
        self.assertEqual(set(rep["feature"]), set(FEATURE_COLUMNS))
        self.assertTrue(rep["psi"].is_monotonic_decreasing)
        self.assertTrue(rep["status"].isin(["stable", "moderate", "major"]).all())

    def test_same_data_is_all_stable(self):
        rep = drift_report(self.ref, self.ref)
        self.assertTrue((rep["status"] == "stable").all())
        self.assertTrue((rep["psi"] < 1e-9).all())

    def test_injected_drift_is_detected_only_where_injected(self):
        drifted = self.cur.copy()
        drifted["amount"] = drifted["amount"] * 4
        rep = drift_report(self.ref, drifted).set_index("feature")
        self.assertEqual(rep.loc["amount", "status"], "major")
        self.assertEqual(rep.loc["is_atm", "status"], "stable")

    def test_status_thresholds(self):
        rng = np.random.default_rng(1)
        base = pd.DataFrame({"a": rng.normal(size=20_000)})
        for shift, expected in ((0.0, "stable"), (1.5, "major")):
            cur = pd.DataFrame({"a": rng.normal(shift, 1, 20_000)})
            self.assertEqual(drift_report(base, cur, features=["a"]).loc[0, "status"], expected)

    def test_missing_features_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference is missing"):
            drift_report(self.ref.drop(columns="amount"), self.cur)
        with self.assertRaisesRegex(ValueError, "current is missing"):
            drift_report(self.ref, self.cur.drop(columns="amount"))

    def test_reports_missing_rate_change(self):
        cur = self.cur.copy()
        cur.loc[cur.index[:100], "amount"] = np.nan
        rep = drift_report(self.ref, cur).set_index("feature")
        self.assertGreater(rep.loc["amount", "nan_rate_cur"], rep.loc["amount", "nan_rate_ref"])


if __name__ == "__main__":
    unittest.main()
