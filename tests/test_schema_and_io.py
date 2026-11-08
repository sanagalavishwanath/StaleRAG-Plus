import tempfile
import unittest
from pathlib import Path

from stalerag.data.io import group_split, holdout_property, random_split, read_jsonl, write_jsonl
from stalerag.data.synthetic import generate
from stalerag.schema import Example, Passage


class SchemaTests(unittest.TestCase):
    def test_passage_rejects_unknown_role(self):
        with self.assertRaises(ValueError):
            Passage("p", "text", None, "bogus")

    def test_example_rejects_identical_answers(self):
        with self.assertRaises(ValueError):
            Example("q", "?", "E", "mayor", "Ann Lee", "ann lee", [])

    def test_missing_role_raises_key_error(self):
        example = generate(1, 0)[0]
        example.passages = [p for p in example.passages if p.role != "stale"]
        with self.assertRaises(KeyError):
            example.passage("stale")

    def test_dict_round_trip(self):
        example = generate(1, 0)[0]
        self.assertEqual(Example.from_dict(example.to_dict()), example)


class IoTests(unittest.TestCase):
    def test_jsonl_round_trip(self):
        examples = generate(12, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "data.jsonl"
            write_jsonl(path, examples)
            self.assertEqual(read_jsonl(path), examples)

    def test_read_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_jsonl("/nonexistent/file.jsonl")

    def test_group_split_is_disjoint_and_complete(self):
        examples = generate(100, 1)
        train, val, test = group_split(examples, (0.5, 0.2, 0.3), seed=4)
        entities = [{e.entity for e in part} for part in (train, val, test)]
        train_e, val_e, test_e = entities
        self.assertFalse(train_e & val_e or train_e & test_e or val_e & test_e)
        self.assertEqual(len(train) + len(val) + len(test), 100)
        self.assertEqual((len(train), len(val), len(test)), (50, 20, 30))

    def test_group_split_keeps_same_entity_together(self):
        examples = generate(30, 1)
        twin = Example.from_dict({**examples[0].to_dict(), "qid": "twin"})
        parts = group_split(examples + [twin], (0.5, 0.2, 0.3), seed=0)
        wanted = {"twin", examples[0].qid}
        holders = [i for i, part in enumerate(parts) if any(e.qid in wanted for e in part)]
        self.assertEqual(len(set(holders)), 1)

    def test_group_split_is_deterministic(self):
        examples = generate(60, 1)
        first = group_split(examples, (0.5, 0.2, 0.3), seed=9)
        second = group_split(examples, (0.5, 0.2, 0.3), seed=9)
        self.assertEqual([[e.qid for e in p] for p in first], [[e.qid for e in p] for p in second])

    def test_group_split_validation(self):
        examples = generate(30, 1)
        with self.assertRaises(ValueError):
            group_split(examples, (0.5, 0.2, 0.2), 0)
        with self.assertRaises(ValueError):
            group_split(examples, (-0.1, 0.6, 0.5), 0)
        with self.assertRaises(ValueError):
            group_split(examples[:2], (0.5, 0.2, 0.3), 0)

    def test_holdout_property_excludes_held_entities(self):
        examples = generate(50, 2)
        rest, held = holdout_property(examples, "mayor")
        self.assertTrue(all(e.prop == "mayor" for e in held))
        self.assertTrue(all(e.prop != "mayor" for e in rest))
        with self.assertRaises(ValueError):
            holdout_property(examples, "no such property")

    def test_random_split_sizes_and_validation(self):
        examples = generate(40, 2)
        train, test = random_split(examples, 0.25, seed=1)
        self.assertEqual((len(train), len(test)), (30, 10))
        with self.assertRaises(ValueError):
            random_split(examples, 1.0, 0)


if __name__ == "__main__":
    unittest.main()
