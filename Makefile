.PHONY: install check clean database monitor runner_database runner_monitor

.DEFAULT_GOAL:=runner_monitor

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

database:
	uv run python src/database.py

monitor:
	uv run python src/monitor.py

runner_database: check database clean

runner_monitor: check monitor clean
