import unittest

from stalerag.data.synthetic import PROPERTIES, generate
from stalerag.retrieval.bm25 import bm25_scores, rank_passages, tokenize
from stalerag.retrieval.conditions import (
    CLOSED_BOOK,
    CONDITIONS,
    CURRENT_ONLY,
    MIXED_CURRENT_FIRST,
    MIXED_STALE_FIRST,
    STALE_ONLY,
    build_context,
    vanilla_context,
)


class SyntheticTests(unittest.TestCase):
    def test_size_uniqueness_and_property_balance(self):
        examples = generate(100, 5)
        self.assertEqual(len(examples), 100)
        self.assertEqual(len({e.qid for e in examples}), 100)
        self.assertEqual(len({e.entity for e in examples}), 100)
        self.assertEqual({e.prop for e in examples}, set(PROPERTIES))

    def test_answers_are_grounded_in_passages(self):
        for example in generate(200, 6):
            self.assertNotEqual(example.current_answer, example.stale_answer)
            self.assertIn(example.stale_answer, example.passage("stale").text)
            self.assertIn(example.current_answer, example.passage("current").text)
            self.assertNotIn(example.current_answer, example.passage("stale").text)
            self.assertNotIn(example.entity, example.distractors()[0].text)

    def test_deterministic_for_a_seed(self):
        self.assertEqual(generate(30, 11), generate(30, 11))
        self.assertNotEqual(generate(30, 11), generate(30, 12))

    def test_date_noise_controls(self):
        none_missing = generate(60, 1, p_date_missing=0.0, p_date_inverted=0.0)
        self.assertTrue(all(p.date for e in none_missing for p in e.passages))
        for e in none_missing:
            self.assertLess(e.passage("stale").date, e.passage("current").date)
        all_missing = generate(60, 1, p_date_missing=1.0)
        self.assertTrue(all(e.passage("stale").date is None for e in all_missing))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            generate(0, 1)
        with self.assertRaises(ValueError):
            generate(5, 1, p_date_missing=1.5)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.example = generate(1, 2)[0]

    def test_five_conditions_have_expected_shape(self):
        self.assertEqual(len(CONDITIONS), 5)
        roles = {c: [p.role for p in build_context(self.example, c)] for c in CONDITIONS}
        self.assertEqual(roles[CLOSED_BOOK], [])
        self.assertEqual(roles[STALE_ONLY], ["stale", "distractor"])
        self.assertEqual(roles[CURRENT_ONLY], ["current", "distractor"])
        self.assertEqual(roles[MIXED_STALE_FIRST], ["stale", "current", "distractor"])
        self.assertEqual(roles[MIXED_CURRENT_FIRST], ["current", "stale", "distractor"])

    def test_unknown_condition_raises(self):
        with self.assertRaises(ValueError):
            build_context(self.example, "nonsense")

    def test_vanilla_is_deterministic_and_complete(self):
        first = vanilla_context(self.example)
        self.assertEqual(first, vanilla_context(self.example))
        self.assertEqual({p.pid for p in first}, {p.pid for p in self.example.passages})

    def test_bm25_edge_cases(self):
        self.assertEqual(bm25_scores("query", []), [])
        self.assertEqual(bm25_scores("zzz", ["alpha beta", "gamma"]), [0.0, 0.0])
        scores = bm25_scores("mayor of oakton", ["the mayor of oakton", "unrelated text here"])
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(tokenize("Hello, World-1!"), ["hello", "world", "1"])
        self.assertEqual(rank_passages("q", []), [])


if __name__ == "__main__":
    unittest.main()
