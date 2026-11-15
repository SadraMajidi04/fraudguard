import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraudguard.data import DataValidationError, generate_transactions
from fraudguard.evaluation import CostModel
from fraudguard.features import FEATURE_COLUMNS
from fraudguard.pipeline import (
    ExperimentConfig,
    format_report,
    load_bundle,
    recall_by_pattern,
    run_experiment,
    save_bundle,
    to_jsonable,
)
from tests.helpers import FeatureEchoScorer, small_config


@lru_cache(maxsize=1)
def experiment():
    return run_experiment(generate_transactions(8000, 0.04), small_config())


class TestExperiment(unittest.TestCase):
    def test_structure_and_counts(self):
        m = experiment().metrics
        d = m["data"]
        self.assertEqual(d["n_train"] + d["n_valid"] + d["n_test"], d["n_rows"])
        self.assertEqual(d["n_rows"], 8000)
        self.assertEqual(d["n_fraud"], 320)
        self.assertLess(d["valid_start"], d["test_start"])
        self.assertEqual(set(m["models"]),
                         {"rules", "logistic_regression", "gradient_boosting", "autoencoder"})

    def test_metrics_are_strictly_json_serializable(self):
        json.dumps(experiment().metrics, allow_nan=False)

    def test_winner_is_chosen_by_validation_cost_not_test(self):
        m = experiment().metrics
        by_valid = min(m["models"], key=lambda k: m["models"][k]["valid"]["cost"])
        self.assertEqual(m["best_model"], by_valid)

    def test_selected_model_beats_doing_nothing_on_held_out_data(self):
        best = experiment().metrics["models"][experiment().metrics["best_model"]]["test"]
        self.assertGreater(best["savings_vs_approve_all"], 0.3)
        self.assertLess(best["cost"], best["cost_approve_all"])

    def test_test_metrics_are_complete_and_consistent(self):
        for name, m in experiment().metrics["models"].items():
            t = m["test"]
            with self.subTest(model=name):
                self.assertTrue(0 <= t["pr_auc"] <= 1 and 0 <= t["roc_auc"] <= 1)
                lo, hi = m["test_pr_auc_ci95"]
                self.assertLessEqual(lo, hi)
                self.assertGreater(t["cost_approve_all"], 0)
                self.assertEqual(t["n"], experiment().metrics["data"]["n_test"])

    def test_gradient_boosting_params_are_reported(self):
        self.assertIn("selected_params",
                      experiment().metrics["models"]["gradient_boosting"])

    def test_reproducible(self):
        a = run_experiment(generate_transactions(4000, 0.05), small_config())
        b = run_experiment(generate_transactions(4000, 0.05), small_config())
        self.assertEqual(a.metrics, b.metrics)

    def test_label_shuffle_sanity_check_finds_no_signal(self):
        """If labels are random, every model must fall to ~chance: proves no leakage."""
        df = generate_transactions(8000, 0.04)
        rng = np.random.default_rng(0)
        df["is_fraud"] = rng.permutation(df["is_fraud"].to_numpy())
        res = run_experiment(df, small_config())
        prev = res.metrics["data"]["fraud_rate"]
        for name, m in res.metrics["models"].items():
            with self.subTest(model=name):
                self.assertLess(m["test"]["pr_auc"], 3 * prev)
                self.assertLess(m["test"]["roc_auc"], 0.62)

    def test_expensive_reviews_make_the_deployed_threshold_infinite(self):
        cfg = small_config()
        cfg.cost_model = CostModel(review_cost=1e9)
        res = run_experiment(generate_transactions(4000, 0.05), cfg)
        self.assertEqual(res.bundle["threshold"], float("inf"))
        json.dumps(res.metrics, allow_nan=False)  # inf became null, not Infinity
        self.assertIn("n/a", format_report(res.metrics))

    def test_invalid_input_is_rejected_early(self):
        df = generate_transactions(4000, 0.05)
        bad = df.copy()
        bad.loc[0, "amount"] = -1
        with self.assertRaises(DataValidationError):
            run_experiment(bad, small_config())
        with self.assertRaises(ValueError):
            run_experiment(df.head(20), small_config())
        with self.assertRaises(ValueError):
            run_experiment(df, ExperimentConfig(train_frac=0.9, valid_frac=0.2))

    def test_input_frame_is_not_modified(self):
        df = generate_transactions(4000, 0.05)
        before = df.copy(deep=True)
        run_experiment(df, small_config())
        pd.testing.assert_frame_equal(df, before)


class TestBundle(unittest.TestCase):
    def test_round_trip_keeps_predictions(self):
        res = experiment()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_bundle(res.bundle, Path(tmp) / "sub" / "m.joblib")
            back = load_bundle(path)
        self.assertEqual(back["model_name"], res.bundle["model_name"])
        self.assertEqual(back["threshold"], res.bundle["threshold"])
        self.assertEqual(list(back["feature_columns"]), FEATURE_COLUMNS)
        X = res.bundle["reference_features"].head(50)
        np.testing.assert_array_equal(back["model"].score(X), res.bundle["model"].score(X))
        self.assertLessEqual(len(back["reference_features"]), 5000)

    def test_load_errors(self):
        res = experiment()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                load_bundle(tmp / "nope.joblib")
            joblib.dump([1, 2, 3], tmp / "list.joblib")
            with self.assertRaisesRegex(ValueError, "not a FraudGuard"):
                load_bundle(tmp / "list.joblib")
            joblib.dump({"model": 1}, tmp / "partial.joblib")
            with self.assertRaisesRegex(ValueError, "missing keys"):
                load_bundle(tmp / "partial.joblib")
            joblib.dump({**res.bundle, "bundle_version": 99}, tmp / "ver.joblib")
            with self.assertRaisesRegex(ValueError, "version"):
                load_bundle(tmp / "ver.joblib")
            joblib.dump({**res.bundle, "feature_columns": ["amount"]}, tmp / "feat.joblib")
            with self.assertRaisesRegex(ValueError, "feature set"):
                load_bundle(tmp / "feat.joblib")

    def test_save_rejects_incomplete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "missing"):
            save_bundle({"model": 1}, Path(tmp) / "x.joblib")


class TestRecallByPattern(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_transactions(20_000, 0.03, include_pattern=True)
        cls.out = recall_by_pattern(cls.df, FeatureEchoScorer("amount"), threshold=100.0)

    def test_columns_and_counts(self):
        from fraudguard.splits import time_split

        self.assertEqual(list(self.out.columns), ["pattern", "n_fraud", "caught", "missed_amount",
                                                  "recall"])
        te = time_split(self.df).test
        self.assertEqual(int(self.out["n_fraud"].sum()), int(self.df.loc[te, "is_fraud"].sum()))
        self.assertTrue(((self.out["recall"] >= 0) & (self.out["recall"] <= 1)).all())

    def test_known_answers_for_an_amount_rule(self):
        r = self.out.set_index("pattern")["recall"]
        self.assertEqual(r["card_testing"], 0.0)  # tiny amounts never reach 100
        self.assertEqual(r["atm_cashout"], 1.0)  # always 200-1000
        self.assertGreater(r["account_takeover"], 0.9)
        self.assertTrue(self.out["recall"].is_monotonic_increasing)  # weakest first

    def test_missed_amount_is_zero_when_everything_is_caught(self):
        row = self.out.set_index("pattern").loc["atm_cashout"]
        self.assertEqual(row["missed_amount"], 0.0)

    def test_requires_pattern_column(self):
        with self.assertRaisesRegex(ValueError, "fraud_pattern"):
            recall_by_pattern(self.df.drop(columns="fraud_pattern"), FeatureEchoScorer(), 1.0)


class TestHelpers(unittest.TestCase):
    def test_to_jsonable(self):
        out = to_jsonable({
            "a": np.float64(1.5), "b": float("inf"), "c": float("nan"), "d": np.int64(3),
            "e": [np.bool_(True), (1, 2)], "f": pd.Timestamp("2025-01-01 10:00"),
            1: "int key",
        })
        self.assertEqual(out["a"], 1.5)
        self.assertIsNone(out["b"])
        self.assertIsNone(out["c"])
        self.assertEqual(out["d"], 3)
        self.assertEqual(out["e"], [True, [1, 2]])
        self.assertEqual(out["f"], "2025-01-01T10:00:00")
        self.assertEqual(out["1"], "int key")
        json.dumps(out, allow_nan=False)

    def test_report_lists_every_model_and_marks_the_selected_one(self):
        report = format_report(experiment().metrics)
        for name in experiment().metrics["models"]:
            self.assertIn(name, report)
        self.assertEqual(report.count("(selected)"), 1)
        self.assertIn("Approve-everything", report)


if __name__ == "__main__":
    unittest.main()
