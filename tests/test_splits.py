import unittest

import numpy as np
import pandas as pd

from fraudguard.splits import time_split
from tests.helpers import make_df


class TestTimeSplit(unittest.TestCase):
    def test_masks_partition_rows_and_respect_time(self):
        df = make_df(6000, 0.05)
        sp = time_split(df)
        total = sp.train.astype(int) + sp.valid.astype(int) + sp.test.astype(int)
        self.assertTrue((total == 1).all())
        ts = df["timestamp"]
        self.assertLess(ts[sp.train].max(), ts[sp.valid].min())
        self.assertLess(ts[sp.valid].max(), ts[sp.test].min())
        self.assertAlmostEqual(sp.train.mean(), 0.6, delta=0.02)
        self.assertAlmostEqual(sp.valid.mean(), 0.2, delta=0.02)

    def test_shuffled_input_still_splits_by_time(self):
        df = make_df(6000, 0.05).sample(frac=1.0, random_state=3)
        sp = time_split(df)
        ts = df["timestamp"].to_numpy()
        self.assertLess(ts[sp.train].max(), ts[sp.valid].min())
        self.assertLess(ts[sp.valid].max(), ts[sp.test].min())

    def test_each_part_has_both_classes(self):
        df = make_df(6000, 0.05)
        sp = time_split(df)
        for m in (sp.train, sp.valid, sp.test):
            self.assertEqual(df.loc[m, "is_fraud"].nunique(), 2)

    def test_custom_fractions(self):
        sp = time_split(make_df(6000, 0.05), 0.5, 0.25)
        self.assertAlmostEqual(sp.test.mean(), 0.25, delta=0.02)

    def test_bad_fractions(self):
        df = make_df(6000, 0.05)
        for a, b in [(0, 0.2), (0.6, 0), (0.9, 0.2), (1.0, 0.1), (-0.1, 0.3), (0.5, 0.5)]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                time_split(df, a, b)

    def test_too_few_rows(self):
        with self.assertRaises(ValueError):
            time_split(make_df(6000, 0.05).head(10))

    def test_single_class_part_is_rejected(self):
        df = make_df(6000, 0.05).copy()
        df.loc[df.index[-1500:], "is_fraud"] = 0  # test split now has no fraud
        with self.assertRaisesRegex(ValueError, "both classes"):
            time_split(df)

    def test_all_same_timestamp_is_rejected(self):
        df = make_df(6000, 0.05).copy()
        df["timestamp"] = pd.Timestamp("2025-01-01")
        with self.assertRaises(ValueError):
            time_split(df)

    def test_frame_without_target_still_splits(self):
        df = make_df(6000, 0.05).drop(columns="is_fraud")
        self.assertTrue(time_split(df).test.any())

    def test_masks_are_boolean_numpy_arrays(self):
        sp = time_split(make_df(6000, 0.05))
        for m in (sp.train, sp.valid, sp.test):
            self.assertIsInstance(m, np.ndarray)
            self.assertEqual(m.dtype, bool)


if __name__ == "__main__":
    unittest.main()
