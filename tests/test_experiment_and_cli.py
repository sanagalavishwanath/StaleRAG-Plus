import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from stalerag.cli import _clean, main
from stalerag.experiments.run import load_examples, run_experiments

TINY = {
    "data": {"source": "synthetic", "n": 150, "synthetic_seed": 3},
    "split": [0.5, 0.2, 0.3],
    "data_seed": 5,
    "seeds": [0, 1],
    "n_boot": 40,
    "coverage": 0.8,
    "filter_threshold": 0.5,
    "nli": {"kind": "lexical"},
    "detector": {"backend": "auto"},
    "readers": [
        {
            "kind": "simulated",
            "name": "sim",
            "first_passage_bias": 0.8,
            "recency_use": 0.2,
            "memory_stale": 0.5,
            "memory_current": 0.2,
        }
    ],
}


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_experiments(copy.deepcopy(TINY))

    def test_structure(self):
        meta = self.results["meta"]
        self.assertEqual(meta["n_train"] + meta["n_val"] + meta["n_test"], 150)
        detector = self.results["detector"]
        self.assertEqual(
            set(detector["ablation"]),
            {
                "full",
                "without_temporal",
                "only_temporal",
                "without_nli",
                "only_nli",
                "without_lexical",
                "only_lexical",
            },
        )
        self.assertEqual(
            set(detector["generalization"]["unseen_property"]),
            {"chief executive officer", "head coach", "mayor", "president", "prime minister"},
        )
        conditions = self.results["readers"]["sim"]["conditions"]
        for key in (
            "closed_book",
            "stale_only",
            "current_only",
            "mixed_stale_first",
            "mixed_current_first",
        ):
            self.assertIn(key, conditions)

    def test_controlled_conditions_behave_as_designed(self):
        conditions = self.results["readers"]["sim"]["conditions"]
        self.assertEqual(conditions["stale_only"]["stale_rate"]["value"], 1.0)
        self.assertEqual(conditions["current_only"]["current_acc"]["value"], 1.0)
        self.assertGreater(
            conditions["mixed_stale_first"]["stale_rate"]["value"],
            conditions["mixed_current_first"]["stale_rate"]["value"],
        )

    def test_reordering_removes_order_sensitivity(self):
        order = self.results["readers"]["sim"]["order_sensitivity"]
        self.assertGreater(order["vanilla"]["flip_rate"]["value"], 0.0)
        self.assertEqual(order["rerank"]["flip_rate"]["value"], 0.0)
        self.assertEqual(order["filter"]["flip_rate"]["value"], 0.0)
        self.assertEqual(order["rerank"]["relative_reduction_vs_vanilla"]["value"], 1.0)

    def test_values_are_probabilities_and_cis_bracket_estimates(self):
        for metrics in self.results["detector"]["main"].values():
            if "ci_low" in metrics:
                self.assertLessEqual(metrics["ci_low"], metrics["value"] + 1e-9)
                self.assertGreaterEqual(metrics["ci_high"], metrics["value"] - 1e-9)
        for method in self.results["readers"]["sim"]["selective"].values():
            self.assertTrue(0.0 <= method["coverage"]["value"] <= 1.0)

    def test_deterministic(self):
        again = run_experiments(copy.deepcopy(TINY))
        again["meta"]["timings_seconds"] = self.results["meta"]["timings_seconds"]
        dump_again = json.dumps(_clean(again), sort_keys=True)
        dump_first = json.dumps(_clean(self.results), sort_keys=True)
        self.assertEqual(dump_again, dump_first)


class FailureTests(unittest.TestCase):
    def test_unknown_data_source(self):
        with self.assertRaises(ValueError):
            load_examples({"source": "nowhere"})

    def test_missing_jsonl(self):
        with self.assertRaises(FileNotFoundError):
            load_examples({"source": "jsonl", "path": "/does/not/exist.jsonl"})

    def test_unknown_reader_kind_in_config(self):
        cfg = copy.deepcopy(TINY)
        cfg["readers"] = [{"kind": "oracle", "name": "x"}]
        with self.assertRaises(ValueError):
            run_experiments(cfg)

    def test_too_few_entities_to_split(self):
        cfg = copy.deepcopy(TINY)
        cfg["data"]["n"] = 2
        with self.assertRaises(ValueError):
            run_experiments(cfg)


class CliTests(unittest.TestCase):
    def test_clean_replaces_nan_and_numpy_scalars(self):
        import numpy as np

        cleaned = _clean({"a": float("nan"), "b": np.float64(1.5), "c": (np.int64(2),)})
        self.assertEqual(cleaned, {"a": None, "b": 1.5, "c": [2]})

    def test_generate_and_run_commands_write_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "syn.jsonl"
            main(["generate-synthetic", "--n", "20", "--out", str(data_path)])
            self.assertEqual(len(data_path.read_text().strip().splitlines()), 20)
            cfg = copy.deepcopy(TINY)
            cfg["data"] = {"source": "jsonl", "path": str(data_path)}
            cfg["seeds"] = [0]
            cfg["data_seed"] = 1
            config_path = Path(tmp) / "cfg.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            out_dir = Path(tmp) / "out"
            main(["run", "--config", str(config_path), "--out-dir", str(out_dir)])
            expected = ("results.json", "summary.md", "risk_coverage.png", "detector_ablation.png")
            for name in expected:
                self.assertTrue((out_dir / name).stat().st_size > 0, name)
            self.assertIn("Simulated run", (out_dir / "summary.md").read_text())
            json.loads((out_dir / "results.json").read_text())


if __name__ == "__main__":
    unittest.main()


class ReportTests(unittest.TestCase):
    def test_render_handles_null_metrics(self):
        from stalerag.report import _fmt

        self.assertEqual(_fmt({"value": None, "ci_low": None, "ci_high": None}), "n/a")
        self.assertEqual(_fmt({"value": 0.5, "ci_low": None, "ci_high": None}), "0.500")
        interval = {"value": 0.5, "ci_low": 0.4, "ci_high": 0.6}
        self.assertEqual(_fmt(interval), "0.500 [0.400, 0.600]")
