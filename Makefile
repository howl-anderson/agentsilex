.PHONY: clean lint format fix

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

lint:
	ruff check src/ tests/ demo/

format:
	ruff format src/ tests/ demo/

fix:
	ruff check --fix src/ tests/ demo/
	ruff format src/ tests/ demo/