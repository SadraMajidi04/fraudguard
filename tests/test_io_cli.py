import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fraudguard.cli import main
from fraudguard.data import (
    DataValidationError,
    generate_transactions,
    load_transactions,
    save_transactions,
)
from tests.helpers import make_df


class TestRoundTrip(unittest.TestCase):
    def test_csv_and_gzip_round_trip_preserve_data(self):
        df = make_df().head(500)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.csv", "nested/dir/b.csv.gz"):
                with self.subTest(name=name):
                    path = save_transactions(df, Path(tmp) / name)
                    back = load_transactions(path)
                    self.assertEqual(len(back), len(df))
                    self.assertEqual(back["transaction_id"].tolist(), df["transaction_id"].tolist())
                    self.assertEqual(int(back["is_fraud"].sum()), int(df["is_fraud"].sum()))
                    pd.testing.assert_series_equal(
                        back["amount"], df["amount"].reset_index(drop=True), check_exact=False
                    )
                    self.assertTrue((back["timestamp"] == df["timestamp"].to_numpy()).all())

    def test_leading_zero_ids_survive_round_trip(self):
        df = make_df().head(50)
        with tempfile.TemporaryDirectory() as tmp:
            back = load_transactions(save_transactions(df, Path(tmp) / "x.csv"))
        self.assertEqual(back["customer_id"].tolist(), df["customer_id"].tolist())


class TestIOErrors(unittest.TestCase):
    def test_refuses_to_save_invalid_data_and_writes_nothing(self):
        df = make_df().head(20)
        df.loc[0, "amount"] = -5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            with self.assertRaises(DataValidationError):
                save_transactions(df, path)
            self.assertFalse(path.exists())

    def test_bad_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fn in (save_transactions, ):
                with self.assertRaises(ValueError):
                    fn(make_df().head(5), Path(tmp) / "x.parquet")
            with self.assertRaises(ValueError):
                load_transactions(Path(tmp) / "x.xlsx")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_transactions("/definitely/not/here.csv")

    def test_loading_corrupt_file_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            path.write_text("transaction_id,timestamp,amount\n1,2025-01-01,5\n")
            with self.assertRaises(DataValidationError):
                load_transactions(path)


class TestCli(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_generate_writes_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv.gz"
            code, out, _ = self.run_cli("generate", "--n", "2000", "--fraud-rate", "0.05",
                                        "--out", str(path))
            self.assertEqual(code, 0)
            self.assertIn("2,000 rows", out)
            df = load_transactions(path)
            self.assertEqual(len(df), 2000)
            self.assertEqual(int(df["is_fraud"].sum()), 100)

    def test_generate_is_reproducible_with_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            for p in (a, b):
                self.assertEqual(self.run_cli("generate", "--n", "1000", "--seed", "3",
                                              "--out", str(p))[0], 0)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_invalid_arguments_return_error_code_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = self.run_cli("generate", "--n", "1000", "--fraud-rate", "5",
                                        "--out", str(Path(tmp) / "x.csv"))
            self.assertEqual(code, 2)
            self.assertIn("error:", err)
            code, _, err = self.run_cli("generate", "--out", str(Path(tmp) / "x.txt"))
            self.assertEqual(code, 2)

    def test_missing_required_argument_exits(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            main(["generate"])


class TestGeneratedFromScratchEndToEnd(unittest.TestCase):
    def test_generate_save_load_validate(self):
        df = generate_transactions(3000, 0.02, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            back = load_transactions(save_transactions(df, Path(tmp) / "e2e.csv"))
        self.assertEqual(int(back["is_fraud"].sum()), 60)


if __name__ == "__main__":
    unittest.main()


class TestCliTrainAndDrift(unittest.TestCase):
    """Slow-ish end-to-end tests: real training through the command line."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "models"
        cls.data = Path(cls.tmp.name) / "tx.csv.gz"
        save_transactions(generate_transactions(6000, 0.04, seed=5), cls.data)
        cls.code, cls.stdout, cls.stderr = cls._run("train", "--data", str(cls.data),
                                                    "--out-dir", str(cls.out))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _run(*argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_train_writes_model_metrics_and_report(self):
        import json

        self.assertEqual(self.code, 0, self.stderr)
        for name in ("model.joblib", "metrics.json", "report.md"):
            self.assertTrue((self.out / name).is_file(), name)
        metrics = json.loads((self.out / "metrics.json").read_text())
        self.assertIn(metrics["best_model"], metrics["models"])
        self.assertIn("selected model:", self.stdout)
        self.assertIn("| Model |", (self.out / "report.md").read_text())

    def test_trained_model_can_be_loaded_and_used_for_scoring(self):
        from fraudguard.features import build_features
        from fraudguard.pipeline import load_bundle

        bundle = load_bundle(self.out / "model.joblib")
        X = build_features(load_transactions(self.data)).head(100)
        self.assertEqual(len(bundle["model"].score(X)), 100)

    def test_drift_detects_a_changed_amount_distribution(self):
        df = load_transactions(self.data)
        df["amount"] = df["amount"] * 5
        shifted = save_transactions(df, Path(self.tmp.name) / "shifted.csv")
        code, out, _ = self._run("drift", "--model", str(self.out / "model.joblib"),
                                 "--data", str(shifted))
        self.assertEqual(code, 3)
        self.assertIn("MAJOR drift", out)
        amount_line = next(line for line in out.splitlines() if line.split()[:1] == ["amount"])
        self.assertIn("major", amount_line)

    def test_drift_report_runs_on_unchanged_data(self):
        code, out, _ = self._run("drift", "--model", str(self.out / "model.joblib"),
                                 "--data", str(self.data))
        self.assertIn(code, (0, 3))
        self.assertIn("psi", out)

    def test_errors_return_code_2(self):
        missing = str(Path(self.tmp.name) / "nope.joblib")
        self.assertEqual(self._run("drift", "--model", missing, "--data", str(self.data))[0], 2)
        self.assertEqual(self._run("train", "--data", "/no/such/file.csv")[0], 2)
        self.assertEqual(self._run("train", "--n", "10", "--out-dir",
                                   str(Path(self.tmp.name) / "m2"))[0], 2)

    def test_parser_knows_serve_command(self):
        from fraudguard.cli import _build_parser

        args = _build_parser().parse_args(["serve", "--model", "m.joblib", "--port", "9000"])
        self.assertEqual((args.command, args.port, args.host), ("serve", 9000, "127.0.0.1"))
