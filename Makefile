PY ?= python3
export PYTHONPATH := src

.PHONY: test lint smoke profile benchmark full

test:
	$(PY) -m unittest discover -s tests -v

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

smoke:
	$(PY) -m stalerag.cli run --config configs/smoke.yaml --out-dir results

profile:
	$(PY) scripts/profile_pipeline.py --config configs/smoke.yaml --out results/profile.txt

benchmark:
	$(PY) -m stalerag.cli build-benchmark --out data/stalerag_benchmark.jsonl --target-size 2000

full:
	$(PY) -m stalerag.cli run --config configs/full.yaml --out-dir results_full
