import unittest

import numpy as np

from stalerag.data.synthetic import generate
from stalerag.models import make_reader
from stalerag.models.base import classify_answer, contains_phrase, normalize_answer
from stalerag.models.hf import build_prompt, clean_answer
from stalerag.models.simulated import SimulatedReader, extract_claim
from stalerag.nli import make_nli
from stalerag.nli.scorer import CONTRADICTION, ENTAILMENT, NEUTRAL, LexicalNLI, label_index
from stalerag.retrieval.conditions import (
    CLOSED_BOOK,
    CURRENT_ONLY,
    MIXED_CURRENT_FIRST,
    MIXED_STALE_FIRST,
    STALE_ONLY,
    build_context,
)


class AnswerNormalizationTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_answer("  The Anna  Reed. "), "anna reed")
        self.assertEqual(normalize_answer(None), "")

    def test_contains_phrase_respects_word_boundaries(self):
        self.assertTrue(contains_phrase("It is Anna Reed.", "Anna Reed"))
        self.assertFalse(contains_phrase("Annabel Reedman", "Anna Reed"))
        self.assertFalse(contains_phrase("anything", ""))

    def test_classify(self):
        example = generate(1, 0)[0]
        cur, stale = example.current_answer, example.stale_answer
        self.assertEqual(classify_answer(cur, example), "current")
        self.assertEqual(classify_answer(stale.upper(), example), "stale")
        self.assertEqual(classify_answer(f"{cur} or {stale}", example), "both")
        self.assertEqual(classify_answer("Nobody", example), "other")
        self.assertEqual(classify_answer(None, example), "other")


class SimulatedReaderTests(unittest.TestCase):
    def setUp(self):
        self.example = generate(1, 4)[0]

    def test_claim_extraction(self):
        texts = {
            "Ann Lee is the mayor of Oakton.": "Ann Lee",
            "Appointed in 2022, Ann Lee is currently the mayor of Oakton.": "Ann Lee",
            "Ann Lee became the mayor of Oakton in 2020, after Bo Ray stepped down.": "Ann Lee",
            "Since 2021, Ann Lee has been the mayor of Oakton.": "Ann Lee",
        }
        for text, expected in texts.items():
            self.assertEqual(extract_claim(text, "mayor", "Oakton"), expected)
        self.assertIsNone(extract_claim("Ann Lee is the mayor of Elsewhere.", "mayor", "Oakton"))

    def test_single_source_conditions_follow_the_passage(self):
        reader = SimulatedReader("r", seed=1)
        stale = reader.answer(self.example, build_context(self.example, STALE_ONLY))
        current = reader.answer(self.example, build_context(self.example, CURRENT_ONLY))
        self.assertEqual(stale.answer, self.example.stale_answer)
        self.assertEqual(current.answer, self.example.current_answer)

    def test_closed_book_uses_memory_and_low_confidence(self):
        reader = SimulatedReader("r", seed=1, memory_stale=1.0, memory_current=0.0)
        output = reader.answer(self.example, build_context(self.example, CLOSED_BOOK))
        self.assertEqual(output.answer, self.example.stale_answer)
        self.assertLess(output.confidence, 0.8)

    def test_fully_order_biased_reader_takes_first_passage(self):
        reader = SimulatedReader("r", seed=0, first_passage_bias=1.0, recency_use=0.0)
        first = reader.answer(self.example, build_context(self.example, MIXED_STALE_FIRST))
        second = reader.answer(self.example, build_context(self.example, MIXED_CURRENT_FIRST))
        self.assertEqual(first.answer, self.example.stale_answer)
        self.assertEqual(second.answer, self.example.current_answer)

    def test_deterministic_and_seed_dependent(self):
        context = build_context(self.example, MIXED_STALE_FIRST)
        a = SimulatedReader("r", seed=5).answer(self.example, context)
        b = SimulatedReader("r", seed=5).answer(self.example, context)
        self.assertEqual(a, b)
        confidences = {
            SimulatedReader("r", seed=s).answer(self.example, context).confidence
            for s in range(5)
        }
        self.assertGreater(len(confidences), 1)

    def test_invalid_parameters(self):
        for kwargs in (
            {"first_passage_bias": 1.2},
            {"memory_stale": 0.8, "memory_current": 0.4},
            {"noise": -1.0},
        ):
            with self.assertRaises(ValueError):
                SimulatedReader("r", seed=0, **kwargs)

    def test_confidence_stays_in_bounds(self):
        reader = SimulatedReader("r", seed=0, noise=5.0)
        for example in generate(40, 2):
            output = reader.answer(example, build_context(example, MIXED_STALE_FIRST))
            self.assertTrue(0.0 < output.confidence < 1.0)


class FactoryAndPromptTests(unittest.TestCase):
    def test_make_reader_and_nli_reject_unknown_kinds(self):
        with self.assertRaises(ValueError):
            make_reader({"kind": "magic"}, 0)
        with self.assertRaises(ValueError):
            make_nli({"kind": "magic"})
        self.assertIsInstance(make_nli({"kind": "lexical"}), LexicalNLI)

    def test_prompt_contains_dates_passages_and_question(self):
        example = generate(1, 3)[0]
        prompt = build_prompt(example.question, build_context(example, MIXED_STALE_FIRST))
        self.assertIn("[1]", prompt)
        self.assertIn(example.question, prompt)
        self.assertTrue(prompt.rstrip().endswith("Answer:"))
        closed = build_prompt(example.question, [])
        self.assertNotIn("[1]", closed)

    def test_clean_answer(self):
        self.assertEqual(clean_answer('  "Anna Reed."\nextra line'), "Anna Reed")
        self.assertEqual(clean_answer("   "), "")


class NliTests(unittest.TestCase):
    def test_probabilities_are_distributions(self):
        probs = LexicalNLI().probs(
            [
                ("Ann Lee is the mayor of Oakton.", "Bo Ray is the mayor of Oakton."),
                ("Ann Lee is the mayor of Oakton.", "Ann Lee is the mayor of Oakton."),
                ("Ann Lee is the mayor of Oakton.", "The weather is mild today."),
            ]
        )
        self.assertEqual(probs.shape, (3, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        self.assertEqual(int(probs[0].argmax()), CONTRADICTION)
        self.assertEqual(int(probs[1].argmax()), ENTAILMENT)
        self.assertEqual(int(probs[2].argmax()), NEUTRAL)

    def test_empty_input(self):
        self.assertEqual(LexicalNLI().probs([]).shape, (0, 3))

    def test_label_index_mapping_and_failure(self):
        mapping = label_index({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"})
        self.assertEqual(mapping, {CONTRADICTION: 0, NEUTRAL: 1, ENTAILMENT: 2})
        with self.assertRaises(ValueError):
            label_index({0: "yes", 1: "no"})


if __name__ == "__main__":
    unittest.main()
