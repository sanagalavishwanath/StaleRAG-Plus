# StaleRAG++

Temporal evidence reliability and stale-information detection for retrieval-augmented generation (RAG).

The project builds a temporal conflict benchmark (outdated vs. current evidence for the same question), measures how
readers behave under five controlled retrieval conditions, trains a contradiction-aware evidence reliability detector
(temporal features + NLI scores + gradient boosting), and uses it for reranking, filtering and confidence-aware abstention.

## Read this first: what is verified and what is not

This repository was developed in a sandbox **without internet access, GPU, or the LightGBM / transformers / torch /
ruff / pytest packages**. So:

| Component | Status |
|---|---|
| Data schema, splits, BM25, five retrieval conditions, features, detector, selective prediction, metrics, bootstrap CIs, experiment runner, report, plots, CLI | Implemented and covered by 97 passing unit tests (`unittest`) |
| End-to-end pipeline | Executed on **synthetic data with simulated readers**; real captured output is in `results/` |
| Wikidata SPARQL + Wikipedia revision clients and benchmark builder | Implemented; tested **only against a fake HTTP session**, never against the live APIs |
| LightGBM backend | Implemented but **not run** (package unavailable). The detector falls back to scikit-learn `HistGradientBoostingClassifier`, which produced every detector number in `results/` |
| Hugging Face readers (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B) and HF NLI model | Implemented in `models/hf.py` and `nli/scorer.py`; **never executed**. Only their pure helpers (prompt building, answer cleaning, label mapping) are unit-tested |
| `make lint` (ruff) | **Not run.** Only `compileall` and an ad hoc line-length / unused-import check were run. Expect possible import-order findings; `ruff check --fix` and `ruff format` should settle them |

**The numbers in `results/` are not results of the StaleRAG++ study.** The readers are rule-based simulations with
hand-set parameters (`configs/smoke.yaml`), the NLI scorer is a lexical heuristic, and the data is templated. They
demonstrate that the machinery works and that its statistics are computed correctly. The resume targets (0.90+ AUROC,
50% lower order sensitivity, 30-40% fewer stale answers, ...) are **targets that still have to be measured** with
`make benchmark` and `make full` on a machine with network access and a GPU.

### Why the synthetic numbers look so good (and mean so little)

- Lexical features alone reach 0.94 AUROC (`only_lexical` in the ablation) because synthetic stale and current passages
  come from different sentence templates. Real Wikipedia revisions will not leak this way.
- The lexical NLI stand-in adds nothing: `without_nli` equals `full`. A real NLI model is a genuinely different
  experiment.
- The order-sensitivity reduction is 1.0 for reranking **by construction**: features are permutation-invariant and ties
  break by passage id, so input order cannot matter. The meaningful question on real models is what that costs or gains
  in accuracy.
- Reader behaviour (position bias, recency use, parametric memory) is whatever the simulation parameters say.
- Seeds barely change the detector (AUROC std about 2e-5), so uncertainty is dominated by question-level sampling,
  which the bootstrap CIs capture.

## Layout

```
configs/        smoke.yaml (synthetic + simulated readers), full.yaml (benchmark jsonl + HF models)
src/stalerag/
  schema.py            Example / Passage dataclasses
  data/                synthetic generator, Wikidata + Wikipedia clients, benchmark builder, splits, jsonl IO
  retrieval/           BM25, the five retrieval conditions
  models/              Reader protocol, simulated reader, Hugging Face reader
  nli/                 NLI scorers (lexical stand-in, Hugging Face cross-encoder)
  detector/            permutation-invariant features, LightGBM / sklearn detector
  pipeline.py          reliability scores -> rerank / filter
  selective.py         confidence combination, coverage thresholds, risk-coverage, bootstrap
  metrics.py           AUROC / AUPRC / F1, bootstrap CIs, order flip rate, relative reduction
  experiments/run.py   seeds, ablations, generalization, five conditions, order sensitivity, abstention
  report.py, plots.py, cli.py
scripts/profile_pipeline.py
tests/              unit tests (unittest; also runnable with pytest)
results/            captured output of a real smoke run (log, JSON, markdown summary, plots, profiles)
```

## Method summary

- **Question and evidence.** "Who is the {property} of {entity}?" with a stale passage (older revision), a current
  passage (latest revision) and one distractor. Real data uses five Wikidata properties (head of government, head of
  state, CEO, head coach, chairperson); the stale holder comes from an ended statement, the passage from the Wikipedia
  revision 30 days before the change.
- **Five retrieval conditions:** `closed_book`, `stale_only`, `current_only`, `mixed_stale_first`,
  `mixed_current_first`. Vanilla RAG is the mixed set in BM25 order.
- **Detector.** 17 features in three groups (temporal, NLI, lexical), all computed relative to the retrieved set so they
  do not depend on passage order. Positive class for AUROC / AUPRC / F1 is **stale**.
- **StaleRAG++ context.** `rerank` sorts passages by detector reliability; `filter` additionally drops passages below
  0.5 reliability (keeping the best one if all would be dropped).
- **Selective prediction.** Confidence = reader confidence x the lowest reliability among passages containing the
  answer. The threshold is chosen on the validation split to hit 80% coverage and applied to test; actual test coverage
  is reported because ties and distribution shift move it (0.78-0.83 across readers and methods here).
- **Metrics.** Stale-answer rate is the fraction of **all** questions answered with the stale value (abstentions do not
  count). Order sensitivity is the answer flip rate between stale-first and current-first orders. All intervals are 95%
  percentile bootstrap intervals over questions; seed-to-seed standard deviations are reported separately.
- **Generalization.** Splits are by entity (no entity in two partitions); extra checks are a random split and
  leave-one-property-out.

## Quick start

Python 3.10+. Linux / macOS / WSL:

```bash
pip install -r requirements.txt            # add requirements-hf.txt for the real models
export PYTHONPATH=src
python -m unittest discover -s tests -v    # or: pytest
python -m stalerag.cli run --config configs/smoke.yaml --out-dir results
```

PowerShell: set `$env:PYTHONPATH = "src"` instead of `export`. The `Makefile` targets (`test`, `smoke`, `profile`,
`benchmark`, `full`, `lint`) wrap the same commands.

### Running the real study

1. `python -m stalerag.cli build-benchmark --out data/stalerag_benchmark.jsonl --target-size 2000` queries Wikidata and
   Wikipedia (rate-limited, can take a long time). Inspect the output: how many examples survive the filters is
   unknown until it is run against the live services, and a different property mix or filtering may be needed to
   reach about 2,000 questions.
2. `pip install -r requirements-hf.txt`, accept the Llama / Mistral licences on Hugging Face and log in.
3. `python -m stalerag.cli run --config configs/full.yaml --out-dir results_full`. Expect to adjust `dtype`,
   batch size or quantization for your GPU. The 7-8B readers are called about 8 times per question per seed (identical
   contexts are cached).

## What the captured smoke run shows (synthetic, simulated)

From `results/summary.md` (2,000 synthetic questions, 600 in the test split, 3 seeds, 1,000 bootstrap resamples):

- Detector: AUROC 0.986, F1 0.924, AUPRC 0.986 (scikit-learn backend); macro AUROC 0.958 when a whole property is held
  out.
- For the order-biased simulated reader: answer flip rate between orders 0.597 for vanilla vs 0.000 after reranking;
  stale-answer rate 0.504 (vanilla) vs 0.059 (rerank + filter, no abstention); accuracy on answered 0.978 at 0.791
  coverage vs 0.495 at 0.810 for vanilla confidence.

Treat these only as evidence that the pipeline is wired correctly.

## Performance notes

`results/profile_before_optimization.txt` vs `results/profile_after_optimization.txt`: the first profile showed about
79% of runtime in per-resample sklearn metric calls inside the bootstrap, not in model training. Replacing them with
vectorized NumPy implementations (`fast_auroc`, `fast_auprc`, `fast_f1`, unit-tested to match sklearn to 12 decimals,
including tied scores) cut the smoke run from 69 s to 27 s wall time with identical results. With real HF readers, model
inference will dominate instead.
