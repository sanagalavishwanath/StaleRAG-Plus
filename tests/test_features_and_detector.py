import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from stalerag.data.synthetic import generate
from stalerag.detector.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    featurize,
    group_columns,
    is_relevant,
)
from stalerag.detector.model import ReliabilityDetector
from stalerag.nli.scorer import LexicalNLI
from stalerag.pipeline import reliability_map, reorder
from stalerag.schema import Passage


def build_rows(examples):
    nli = LexicalNLI()
    rows, labels = [], []
    for example in examples:
        matrix = featurize(example, example.passages, nli)
        for index, passage in enumerate(example.passages):
            if passage.role != "distractor":
                rows.append(matrix[index])
                labels.append(int(passage.role == "stale"))
    return np.array(rows), np.array(labels)


class FeatureTests(unittest.TestCase):
    def test_groups_partition_feature_names(self):
        flat = [n for names in FEATURE_GROUPS.values() for n in names]
        self.assertEqual(flat, FEATURE_NAMES)
        self.assertEqual(len(set(flat)), len(flat))
        columns = sorted(group_columns(list(FEATURE_GROUPS)))
        self.assertEqual(columns, list(range(len(FEATURE_NAMES))))

    def test_features_are_permutation_invariant(self):
        nli = LexicalNLI()
        for example in generate(40, 8):
            base = featurize(example, example.passages, nli)
            by_pid = {p.pid: base[i] for i, p in enumerate(example.passages)}
            shuffled = list(example.passages)
            random.Random(1).shuffle(shuffled)
            again = featurize(example, shuffled, nli)
            for i, passage in enumerate(shuffled):
                np.testing.assert_array_equal(again[i], by_pid[passage.pid])

    def test_irrelevant_passages_get_nan_rows(self):
        example = generate(1, 3)[0]
        matrix = featurize(example, example.passages, LexicalNLI())
        for index, passage in enumerate(example.passages):
            self.assertEqual(is_relevant(example, passage), passage.role != "distractor")
            self.assertEqual(bool(np.isnan(matrix[index]).all()), passage.role == "distractor")

    def test_no_relevant_passages_returns_all_nan(self):
        example = generate(1, 3)[0]
        matrix = featurize(example, example.distractors(), LexicalNLI())
        self.assertTrue(np.isnan(matrix).all())

    def test_missing_dates_yield_nan_temporal_features(self):
        example = generate(1, 3, p_date_missing=1.0)[0]
        matrix = featurize(example, example.passages, LexicalNLI())
        names = {n: i for i, n in enumerate(FEATURE_NAMES)}
        row = matrix[0]
        self.assertEqual(row[names["has_date"]], 0.0)
        self.assertTrue(math.isnan(row[names["date_rank"]]))
        self.assertTrue(math.isnan(row[names["days_behind_newest"]]))

    def test_dates_order_the_passages(self):
        example = generate(1, 3, p_date_missing=0.0, p_date_inverted=0.0)[0]
        matrix = featurize(example, example.passages, LexicalNLI())
        names = {n: i for i, n in enumerate(FEATURE_NAMES)}
        stale_row, current_row = matrix[0], matrix[1]
        self.assertEqual(stale_row[names["date_rank"]], 0.0)
        self.assertEqual(current_row[names["date_rank"]], 1.0)
        self.assertGreater(stale_row[names["days_behind_newest"]], 0.0)
        self.assertEqual(current_row[names["days_behind_newest"]], 0.0)

    def test_single_relevant_passage_has_nan_pairwise_features(self):
        example = generate(1, 3)[0]
        single = [example.passage("current"), *example.distractors()]
        matrix = featurize(example, single, LexicalNLI())
        names = {n: i for i, n in enumerate(FEATURE_NAMES)}
        self.assertTrue(math.isnan(matrix[0][names["nli_contra_max"]]))
        self.assertFalse(math.isnan(matrix[0][names["q_overlap"]]))


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = build_rows(generate(500, 21))

    def test_learns_signal_on_held_out_rows(self):
        cut = 700
        detector = ReliabilityDetector(seed=0).fit(self.x[:cut], self.y[:cut])
        auroc = roc_auc_score(self.y[cut:], detector.stale_probability(self.x[cut:]))
        self.assertGreater(auroc, 0.85)

    def test_reliability_is_complement_of_stale_probability(self):
        detector = ReliabilityDetector(seed=0).fit(self.x, self.y)
        np.testing.assert_allclose(
            detector.reliability(self.x[:20]) + detector.stale_probability(self.x[:20]), 1.0
        )

    def test_column_subset_is_respected(self):
        columns = group_columns(["temporal"])
        detector = ReliabilityDetector(seed=0, columns=columns).fit(self.x, self.y)
        altered = self.x.copy()
        other = [i for i in range(self.x.shape[1]) if i not in columns]
        altered[:, other] = 0.0
        np.testing.assert_allclose(
            detector.stale_probability(self.x), detector.stale_probability(altered)
        )

    def test_failure_modes(self):
        with self.assertRaises(RuntimeError):
            ReliabilityDetector().stale_probability(self.x)
        with self.assertRaises(ValueError):
            ReliabilityDetector().fit(self.x, np.zeros(len(self.x)))
        with self.assertRaises(ValueError):
            ReliabilityDetector().fit(self.x, self.y[:-1])
        with self.assertRaises(ValueError):
            ReliabilityDetector(backend="mystery")
        detector = ReliabilityDetector().fit(self.x, self.y)
        with self.assertRaises(ValueError):
            detector.stale_probability(self.x[0])
        self.assertEqual(len(detector.stale_probability(self.x[:0])), 0)

    def test_save_load_round_trip(self):
        detector = ReliabilityDetector(seed=1).fit(self.x, self.y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.joblib"
            detector.save(path)
            loaded = ReliabilityDetector.load(path)
        np.testing.assert_allclose(
            loaded.stale_probability(self.x[:30]), detector.stale_probability(self.x[:30])
        )

    def test_same_seed_is_reproducible(self):
        a = ReliabilityDetector(seed=3).fit(self.x, self.y).stale_probability(self.x[:50])
        b = ReliabilityDetector(seed=3).fit(self.x, self.y).stale_probability(self.x[:50])
        np.testing.assert_allclose(a, b)

    def test_explicit_lightgbm_request_fails_cleanly_when_missing(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                ReliabilityDetector(backend="lightgbm")
        else:
            self.assertEqual(ReliabilityDetector(backend="lightgbm").backend_name, "lightgbm")


class PipelineTests(unittest.TestCase):
    def test_reorder_is_input_order_invariant_and_sorted(self):
        example = generate(1, 5)[0]
        rel = {example.passages[0].pid: 0.2, example.passages[1].pid: 0.9}
        forward = reorder(example.passages, rel)
        backward = reorder(list(reversed(example.passages)), rel)
        self.assertEqual(forward, backward)
        self.assertEqual([p.role for p in forward], ["current", "stale", "distractor"])

    def test_filter_drops_low_reliability_and_unscored(self):
        example = generate(1, 5)[0]
        rel = {example.passages[0].pid: 0.2, example.passages[1].pid: 0.9}
        kept = reorder(example.passages, rel, drop_below=0.5)
        self.assertEqual([p.role for p in kept], ["current"])

    def test_filter_never_returns_empty_when_something_was_scored(self):
        example = generate(1, 5)[0]
        rel = {example.passages[0].pid: 0.1, example.passages[1].pid: 0.3}
        kept = reorder(example.passages, rel, drop_below=0.5)
        self.assertEqual([p.role for p in kept], ["current"])
        self.assertEqual(reorder(example.passages, {}, drop_below=0.5), [])

    def test_reliability_map_skips_unscored_rows(self):
        x, y = build_rows(generate(200, 4))
        detector = ReliabilityDetector(seed=0).fit(x, y)
        example = generate(1, 9)[0]
        matrix = featurize(example, example.passages, LexicalNLI())
        mapping = reliability_map(example.passages, matrix, detector)
        self.assertEqual(set(mapping), {example.passages[0].pid, example.passages[1].pid})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in mapping.values()))
        blank = Passage("x", "no entity", None, "distractor")
        nan_row = np.full((1, len(FEATURE_NAMES)), np.nan)
        empty = reliability_map([blank], nan_row, detector)
        self.assertEqual(empty, {})


if __name__ == "__main__":
    unittest.main()
