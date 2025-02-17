.PHONY: install check clean database

install: pyproject.toml
	uv venv; source .venv/bin/activate
	uv sync

check: install
	uv tool run isort src
	uv tool run ruff check src

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf `find . -type d -name catboost_info`
	rm -rf .ruff_cache

database: check
	uv run python src/database.py
