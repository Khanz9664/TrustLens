.PHONY: install format lint test check clean

install:
	pip install -e ".[dev,full]"

format:
	ruff format .

lint:
	ruff check --fix --unsafe-fixes .

test:
	pytest tests/ -v

check: lint test

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf *.egg-info
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
