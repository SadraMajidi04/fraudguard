import unittest

import numpy as np
import pandas as pd

from fraudguard.data import DataValidationError, validate_transactions
from tests.helpers import make_df


def small() -> pd.DataFrame:
    return make_df().head(200).reset_index(drop=True)


class TestValidData(unittest.TestCase):
    def test_valid_table_passes(self):
        rep = validate_transactions(small())
        self.assertTrue(rep.ok)

    def test_extra_columns_allowed(self):
        df = small()
        df["extra"] = 1
        self.assertTrue(validate_transactions(df).ok)

    def test_does_not_mutate_input(self):
        df = small()
        before = df.copy(deep=True)
        validate_transactions(df)
        pd.testing.assert_frame_equal(df, before)

    def test_bool_target_is_accepted(self):
        df = small()
        df["is_fraud"] = df["is_fraud"].astype(bool)
        self.assertTrue(validate_transactions(df, raise_on_error=False).ok)

    def test_object_dtype_strings_are_accepted(self):
        df = small()
        df["customer_id"] = df["customer_id"].astype(object)
        self.assertTrue(validate_transactions(df, raise_on_error=False).ok)


class TestBrokenData(unittest.TestCase):
    def assert_fails(self, df, fragment):
        with self.assertRaises(DataValidationError) as ctx:
            validate_transactions(df)
        self.assertIn(fragment, str(ctx.exception))
        rep = validate_transactions(df, raise_on_error=False)
        self.assertFalse(rep.ok)

    def test_not_a_dataframe(self):
        for bad in (None, [1, 2], {"a": 1}, "file.csv"):
            with self.subTest(type=type(bad).__name__), self.assertRaises(TypeError):
                validate_transactions(bad)

    def test_empty_table(self):
        self.assert_fails(small().iloc[0:0], "0 rows")

    def test_missing_columns(self):
        for col in ("amount", "timestamp", "is_fraud", "customer_id", "device_id"):
            with self.subTest(col=col):
                self.assert_fails(small().drop(columns=col), "missing required columns")

    def test_duplicate_column_names(self):
        df = small()
        df = pd.concat([df, df[["amount"]]], axis=1)
        self.assert_fails(df, "duplicate column names")

    def test_null_values(self):
        for col in ("amount", "customer_id", "timestamp", "is_fraud", "channel"):
            with self.subTest(col=col):
                df = small()
                df[col] = df[col].astype(object)
                df.loc[3, col] = None
                self.assert_fails(df, f"column '{col}' has 1 null")

    def test_nan_amount(self):
        df = small()
        df.loc[0, "amount"] = np.nan
        self.assert_fails(df, "'amount' has 1 null")

    def test_non_positive_amounts(self):
        for bad in (0, -1.5):
            with self.subTest(amount=bad):
                df = small()
                df.loc[0, "amount"] = bad
                self.assert_fails(df, "<= 0")

    def test_infinite_amount(self):
        for bad in (np.inf, -np.inf):
            with self.subTest(amount=bad):
                df = small()
                df.loc[0, "amount"] = bad
                self.assert_fails(df, "non-finite")

    def test_amount_as_text(self):
        df = small()
        df["amount"] = df["amount"].astype(str)
        self.assert_fails(df, "'amount' must be numeric")

    def test_target_values(self):
        for bad in (2, -1, 0.5):
            with self.subTest(label=bad):
                df = small()
                df["is_fraud"] = df["is_fraud"].astype(float)
                df.loc[0, "is_fraud"] = bad
                self.assert_fails(df, "must contain only 0/1")

    def test_target_as_text(self):
        df = small()
        df["is_fraud"] = df["is_fraud"].astype(str)
        self.assert_fails(df, "must be numeric 0/1")

    def test_duplicate_transaction_ids(self):
        df = small()
        df.loc[1, "transaction_id"] = df.loc[0, "transaction_id"]
        self.assert_fails(df, "duplicate value")

    def test_transaction_id_must_be_integer(self):
        df = small()
        df["transaction_id"] = df["transaction_id"].astype(float)
        self.assert_fails(df, "'transaction_id' must be an integer")

    def test_timestamp_as_text(self):
        df = small()
        df["timestamp"] = df["timestamp"].astype(str)
        self.assert_fails(df, "'timestamp' must be a datetime")

    def test_unknown_channel(self):
        df = small()
        df["channel"] = df["channel"].astype(object)
        df.loc[0, "channel"] = "carrier-pigeon"
        self.assert_fails(df, "unknown values")

    def test_blank_strings(self):
        for col in ("customer_id", "device_id", "country"):
            for blank in ("", "   "):
                with self.subTest(col=col, blank=blank):
                    df = small()
                    df[col] = df[col].astype(object)
                    df.loc[0, col] = blank
                    self.assert_fails(df, "blank strings")

    def test_numeric_id_columns_rejected(self):
        df = small()
        df["customer_id"] = np.arange(len(df))
        self.assert_fails(df, "'customer_id' must be a string")

    def test_all_problems_reported_together(self):
        df = small().drop(columns="device_id")
        df.loc[0, "amount"] = -1
        df["channel"] = df["channel"].astype(object)
        df.loc[1, "channel"] = "x"
        rep = validate_transactions(df, raise_on_error=False)
        self.assertGreaterEqual(len(rep.errors), 3)


class TestWarnings(unittest.TestCase):
    def test_no_fraud_warns(self):
        df = small()
        df["is_fraud"] = 0
        rep = validate_transactions(df)
        self.assertTrue(rep.ok)
        self.assertTrue(any("no fraud" in w for w in rep.warnings))

    def test_all_fraud_warns(self):
        df = small()
        df["is_fraud"] = 1
        self.assertTrue(any("every row is fraud" in w for w in validate_transactions(df).warnings))

    def test_high_fraud_rate_warns(self):
        df = small()
        df["is_fraud"] = [1] * 120 + [0] * 80
        self.assertTrue(any("unusually high" in w for w in validate_transactions(df).warnings))

    def test_no_warning_on_healthy_data(self):
        self.assertEqual(validate_transactions(make_df()).warnings, [])


if __name__ == "__main__":
    unittest.main()
