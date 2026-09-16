.PHONY: install test smoke lint fast
install:
	python -m pip install -e ".[dev]"
test:
	python -m pytest
smoke:
	python -m cfq.cli smoke --output-dir results/smoke
lint:
	python -m compileall -q cfq scripts tests
	python -m ruff check cfq scripts tests
fast:
	python scripts/run_all.py --fast
