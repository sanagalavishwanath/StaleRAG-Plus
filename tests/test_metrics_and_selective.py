import math
import unittest

import numpy as np

from stalerag.metrics import (
    bootstrap_ci,
    bootstrap_metric_ci,
    bootstrap_relative_reduction,
    detector_metrics,
    order_flip_rate,
)
from stalerag.schema import Passage
from stalerag.selective import (
    area_under_risk_coverage,
    bootstrap_selective,
    combined_confidence,
    risk_coverage_curve,
    selective_summary,
    threshold_for_coverage,
)


class MetricsTests(unittest.TestCase):
    def test_bootstrap_ci_brackets_mean_and_is_deterministic(self):
        values = np.random.default_rng(0).normal(5, 1, 200)
        low, high = bootstrap_ci(values, 500, seed=3)
        self.assertLess(low, values.mean())
        self.assertGreater(high, values.mean())
        self.assertEqual((low, high), bootstrap_ci(values, 500, seed=3))

    def test_bootstrap_ci_rejects_empty(self):
        with self.assertRaises(ValueError):
            bootstrap_ci(np.array([]))

    def test_metric_ci_rejects_single_class(self):
        with self.assertRaises(ValueError):
            bootstrap_metric_ci(np.zeros(10), np.random.rand(10), lambda y, s: 0.0, 20)

    def test_metric_ci_rejects_mismatched_sizes(self):
        with self.assertRaises(ValueError):
            bootstrap_metric_ci(np.array([0, 1]), np.array([0.1]), lambda y, s: 0.0)

    def test_detector_metrics_perfect_and_random(self):
        y = np.array([0, 0, 0, 1, 1, 1] * 20)
        perfect = detector_metrics(y, y.astype(float), n_boot=50)
        self.assertEqual(perfect["auroc"]["value"], 1.0)
        self.assertEqual(perfect["f1"]["value"], 1.0)
        inverted = detector_metrics(y, 1.0 - y, n_boot=50)
        self.assertEqual(inverted["auroc"]["value"], 0.0)

    def test_relative_reduction(self):
        point, low, high = bootstrap_relative_reduction(
            np.array([1.0, 1.0, 0.0, 0.0]), np.zeros(4), n_boot=100
        )
        self.assertEqual(point, 1.0)
        self.assertLessEqual(low, high)
        point, _, _ = bootstrap_relative_reduction(np.ones(10), np.ones(10) * 0.7, n_boot=50)
        self.assertAlmostEqual(point, 0.3)

    def test_relative_reduction_zero_baseline_is_nan(self):
        point, low, high = bootstrap_relative_reduction(np.zeros(5), np.zeros(5), n_boot=20)
        self.assertTrue(math.isnan(point) and math.isnan(low) and math.isnan(high))

    def test_relative_reduction_validates_shapes(self):
        with self.assertRaises(ValueError):
            bootstrap_relative_reduction(np.ones(3), np.ones(4))

    def test_order_flip_rate(self):
        self.assertEqual(order_flip_rate(["a", "b"], ["a", "c"]), 0.5)
        with self.assertRaises(ValueError):
            order_flip_rate(["a"], [])
        with self.assertRaises(ValueError):
            order_flip_rate([], [])


class SelectiveTests(unittest.TestCase):
    def test_threshold_hits_requested_coverage(self):
        confidence = np.linspace(0, 1, 101)
        threshold = threshold_for_coverage(confidence, 0.8)
        self.assertAlmostEqual((confidence >= threshold).mean(), 81 / 101, places=6)

    def test_full_coverage_answers_everything(self):
        confidence = np.random.default_rng(1).random(50)
        threshold = threshold_for_coverage(confidence, 1.0)
        self.assertTrue((confidence >= threshold).all())

    def test_ties_can_only_increase_coverage(self):
        confidence = np.array([0.5] * 10)
        threshold = threshold_for_coverage(confidence, 0.2)
        self.assertEqual((confidence >= threshold).mean(), 1.0)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            threshold_for_coverage(np.array([]), 0.8)
        for bad in (0.0, 1.5, -0.2):
            with self.assertRaises(ValueError):
                threshold_for_coverage(np.array([0.1, 0.2]), bad)

    def test_summary_values(self):
        correct = np.array([1, 1, 0, 0], dtype=bool)
        stale = np.array([0, 0, 1, 0], dtype=bool)
        confidence = np.array([0.9, 0.8, 0.3, 0.2])
        summary = selective_summary(correct, stale, confidence, 0.5)
        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["accuracy_answered"], 1.0)
        self.assertEqual(summary["stale_rate_all"], 0.0)
        none = selective_summary(correct, stale, confidence, 2.0)
        self.assertTrue(math.isnan(none["accuracy_answered"]))
        self.assertEqual(none["coverage"], 0.0)

    def test_combined_confidence_uses_weakest_supporting_passage(self):
        context = [
            Passage("a", "Ann Lee is the mayor of Oak.", None, "current"),
            Passage("b", "Bob Ray was before Ann Lee.", None, "stale"),
        ]
        reliability = {"a": 0.9, "b": 0.2}
        self.assertAlmostEqual(combined_confidence(1.0, "Ann Lee", context, reliability), 0.2)
        self.assertAlmostEqual(combined_confidence(1.0, "Bob Ray", context, reliability), 0.2)
        self.assertAlmostEqual(combined_confidence(0.8, "Zed Quinn", context, reliability), 0.4)
        self.assertAlmostEqual(combined_confidence(0.8, "Ann Lee", context[:1], reliability), 0.72)

    def test_bootstrap_selective_validates_shape(self):
        with self.assertRaises(ValueError):
            bootstrap_selective(np.ones(5), np.ones(5), np.ones(5))
        ok = np.ones((2, 6), dtype=bool)
        result = bootstrap_selective(ok, ~ok, ok, n_boot=20)
        self.assertEqual(result["coverage"], (1.0, 1.0))
        self.assertEqual(result["stale_rate_all"], (0.0, 0.0))

    def test_risk_coverage_and_aurc(self):
        correct = np.array([1, 1, 1, 0, 0], dtype=bool)
        good_conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        coverage, risk = risk_coverage_curve(correct, good_conf, points=5)
        self.assertTrue((np.diff(coverage) > 0).all())
        self.assertEqual(risk[0], 0.0)
        expected = np.mean([0, 0, 0, 0.25, 0.4])
        self.assertEqual(area_under_risk_coverage(correct, good_conf), expected)
        self.assertGreater(area_under_risk_coverage(correct, good_conf[::-1]), 0.3)
        with self.assertRaises(ValueError):
            risk_coverage_curve(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()


class FastMetricTests(unittest.TestCase):
    def test_fast_metrics_match_sklearn_including_ties(self):
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        from stalerag.metrics import fast_auprc, fast_auroc, fast_f1

        rng = np.random.default_rng(0)
        for trial in range(60):
            size = int(rng.integers(8, 300))
            y = rng.integers(0, 2, size)
            if y.min() == y.max():
                y[0] = 1 - y[0]
            scores = rng.random(size)
            if trial % 2:
                scores = np.round(scores, 1)
            self.assertAlmostEqual(fast_auroc(y, scores), roc_auc_score(y, scores), places=12)
            self.assertAlmostEqual(
                fast_auprc(y, scores), average_precision_score(y, scores), places=12
            )
            self.assertAlmostEqual(
                fast_f1(y, scores, 0.5), f1_score(y, scores >= 0.5, zero_division=0), places=12
            )

    def test_fast_metrics_reject_degenerate_labels(self):
        from stalerag.metrics import fast_auprc, fast_auroc, fast_f1

        with self.assertRaises(ValueError):
            fast_auroc(np.zeros(5), np.random.rand(5))
        with self.assertRaises(ValueError):
            fast_auprc(np.zeros(5), np.random.rand(5))
        self.assertEqual(fast_f1(np.zeros(4), np.zeros(4)), 0.0)
