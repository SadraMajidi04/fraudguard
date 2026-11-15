import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from fraudguard.evaluation import (
    CostModel,
    bootstrap_ci,
    evaluate_at_threshold,
    optimal_threshold,
    pr_auc,
    recall_at_alert_budget,
    roc_auc,
    total_cost,
)


def brute_force_cost(y, score, amount, cm):
    best = np.inf
    for thr in np.r_[np.unique(score), np.inf]:
        best = min(best, total_cost(y, score >= thr, amount, cm))
    return best


class TestRankingMetrics(unittest.TestCase):
    def test_perfect_and_worst_ranking(self):
        y = [0, 0, 0, 1, 1]
        self.assertEqual(pr_auc(y, [0.1, 0.2, 0.3, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc(y, [0.1, 0.2, 0.3, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc(y, [0.9, 0.8, 0.7, 0.2, 0.1]), 0.0)

    def test_random_scores_pr_auc_close_to_prevalence(self):
        rng = np.random.default_rng(0)
        y = (rng.random(50_000) < 0.02).astype(int)
        self.assertAlmostEqual(pr_auc(y, rng.random(y.size)), 0.02, delta=0.01)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(1)
        y = (rng.random(2000) < 0.1).astype(int)
        s = rng.random(2000) + y * 0.5
        self.assertAlmostEqual(pr_auc(y, s), average_precision_score(y, s))

    def test_invalid_inputs(self):
        bad_cases = [
            ([1, 1, 1], [0.1, 0.2, 0.3]),  # single class
            ([0, 0, 0], [0.1, 0.2, 0.3]),
            ([0, 1, 2], [0.1, 0.2, 0.3]),  # not binary
            ([0, 1], [0.1, 0.2, 0.3]),  # length mismatch
            ([0, 1], [0.1, np.nan]),
            ([0, 1], [0.1, np.inf]),
            ([], []),
        ]
        for y, s in bad_cases:
            with self.subTest(y=y, s=s), self.assertRaises(ValueError):
                pr_auc(y, s)


class TestAlertBudget(unittest.TestCase):
    def test_hand_example(self):
        y = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        s = np.array([0.1, 0.9, 0.2, 0.3, 0.0, 0.05, 0.01, 0.02, 0.03, 0.04])
        r = recall_at_alert_budget(y, s, budget=0.2)  # top 2 -> idx 1 and 3
        self.assertEqual((r["n_flagged"], r["recall"], r["precision"]), (2, 1.0, 1.0))
        r = recall_at_alert_budget(y, s, budget=0.1)  # top 1 -> idx 1
        self.assertEqual((r["n_flagged"], r["recall"], r["precision"]), (1, 0.5, 1.0))

    def test_at_least_one_alert_and_budget_validation(self):
        self.assertEqual(recall_at_alert_budget([0, 1, 0], [0.1, 0.9, 0.2], 0.0001)["n_flagged"], 1)
        for b in (0, -0.1, 1.5):
            with self.subTest(b=b), self.assertRaises(ValueError):
                recall_at_alert_budget([0, 1], [0.1, 0.9], b)


class TestCosts(unittest.TestCase):
    def test_cost_model_validation(self):
        for kw in (dict(review_cost=-1), dict(review_cost=np.nan), dict(fraud_loss_rate=0),
                   dict(fraud_loss_rate=np.inf)):
            with self.subTest(**kw), self.assertRaises(ValueError):
                CostModel(**kw)

    def test_total_cost_hand_example(self):
        y = np.array([1, 1, 0, 0])
        amount = np.array([100.0, 40.0, 10.0, 10.0])
        flagged = np.array([True, False, True, False])
        # 2 alerts * 5 + missed fraud (40) = 50
        self.assertEqual(total_cost(y, flagged, amount, CostModel(5, 1.0)), 50.0)
        self.assertEqual(total_cost(y, flagged, amount, CostModel(5, 0.5)), 30.0)
        self.assertEqual(total_cost(y, np.zeros(4, bool), amount), 140.0)

    def test_total_cost_input_checks(self):
        with self.assertRaises(ValueError):
            total_cost([0, 1], [True], [1.0, 1.0])
        with self.assertRaises(ValueError):
            total_cost([0, 1], [True, False], [1.0, -1.0])

    def test_optimal_threshold_matches_brute_force_on_random_data(self):
        for seed in range(25):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(5, 300))
            y = (rng.random(n) < rng.uniform(0.02, 0.4)).astype(int)
            score = np.round(rng.random(n) + y * rng.uniform(0, 1), int(rng.integers(1, 4)))
            amount = np.round(rng.lognormal(3, 1, n), 2)
            cm = CostModel(float(rng.uniform(0, 30)), float(rng.uniform(0.2, 1.5)))
            with self.subTest(seed=seed):
                op = optimal_threshold(y, score, amount, cm)
                self.assertAlmostEqual(op.cost, brute_force_cost(y, score, amount, cm), places=6)
                # the reported threshold really produces the reported cost and alert count
                flagged = score >= op.threshold
                self.assertAlmostEqual(total_cost(y, flagged, amount, cm), op.cost, places=6)
                self.assertEqual(int(flagged.sum()), op.n_flagged)

    def test_huge_review_cost_means_flag_nothing(self):
        y = np.array([1, 0, 0, 1])
        op = optimal_threshold(y, [0.9, 0.1, 0.2, 0.8], [10, 10, 10, 10], CostModel(1e9))
        self.assertEqual(op.n_flagged, 0)
        self.assertEqual(op.threshold, float("inf"))

    def test_free_reviews_means_catch_all_fraud(self):
        y = np.array([1, 0, 0, 1, 0])
        s = np.array([0.9, 0.1, 0.2, 0.8, 0.3])
        op = optimal_threshold(y, s, [50, 5, 5, 70, 5], CostModel(0.0))
        self.assertEqual(op.cost, 0.0)

    def test_all_tied_scores_never_split_the_tie(self):
        y = np.array([1, 0, 0, 1])
        op = optimal_threshold(y, [0.5] * 4, [100, 1, 1, 100], CostModel(5))
        self.assertIn(op.n_flagged, (0, 4))

    def test_cheap_frauds_are_not_worth_an_alert(self):
        y = np.array([1, 1])
        op = optimal_threshold(y, [0.9, 0.8], [2.0, 2.0], CostModel(5))
        self.assertEqual(op.n_flagged, 0)


class TestEvaluateAtThreshold(unittest.TestCase):
    def test_hand_example(self):
        y = np.array([1, 1, 0, 0, 0, 0])
        s = np.array([0.9, 0.4, 0.8, 0.1, 0.2, 0.3])
        amount = np.array([100.0, 60.0, 5.0, 5.0, 5.0, 5.0])
        r = evaluate_at_threshold(y, s, amount, 0.5, CostModel(5, 1.0), budget=0.5)
        # flagged: idx0 (fraud), idx2 (legit) -> 2 alerts; missed fraud idx1 (60)
        self.assertEqual(r["cost"], 2 * 5 + 60)
        self.assertEqual(r["cost_approve_all"], 160)
        self.assertAlmostEqual(r["savings_vs_approve_all"], (160 - 70) / 160)
        self.assertEqual(r["recall"], 0.5)
        self.assertEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["caught_fraud_amount_share"], 100 / 160)
        self.assertAlmostEqual(r["flag_rate"], 2 / 6)
        self.assertEqual(r["n_fraud"], 2)

    def test_threshold_infinity_flags_nothing_without_crashing(self):
        r = evaluate_at_threshold([1, 0, 0, 1], [0.9, 0.1, 0.2, 0.8], [10] * 4, float("inf"))
        self.assertEqual(r["flag_rate"], 0.0)
        self.assertTrue(np.isnan(r["precision"]))
        self.assertEqual(r["savings_vs_approve_all"], 0.0)

    def test_single_class_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_at_threshold([0, 0], [0.1, 0.2], [1, 1], 0.5)


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.random(3000) < 0.1).astype(int)
        self.s = rng.random(3000) + self.y * 0.6

    def test_interval_contains_estimate_and_is_ordered(self):
        est, lo, hi = bootstrap_ci(pr_auc, self.y, self.s, n_boot=100, seed=1)
        self.assertLessEqual(lo, est)
        self.assertLessEqual(est, hi)
        self.assertLess(hi - lo, 0.2)

    def test_deterministic_for_seed(self):
        a = bootstrap_ci(pr_auc, self.y, self.s, n_boot=50, seed=3)
        b = bootstrap_ci(pr_auc, self.y, self.s, n_boot=50, seed=3)
        self.assertEqual(a, b)

    def test_bad_arguments(self):
        for kw in (dict(n_boot=5), dict(alpha=0), dict(alpha=1)):
            with self.subTest(**kw), self.assertRaises(ValueError):
                bootstrap_ci(pr_auc, self.y, self.s, **kw)


if __name__ == "__main__":
    unittest.main()
