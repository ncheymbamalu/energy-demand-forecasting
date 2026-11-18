.PHONY: sync clean check fix etl

sync: .venv
	@echo "🔄 Synchronizing virtual environment with uv..."
	@uv sync

clean:
	@echo "Cleaning up codebase..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .ruff_cache .pytest_cache

check: .venv
	@echo "🔍 Running Ruff linter..."
	@uv run ruff check src
	@echo "⚡ Running ty type check..."
	@uv run ty check src

fix: .venv
	@echo "🛠️ Fixing lint issues with Ruff..."
	@uv run ruff check --fix src

etl:
	uv run dvc pull
	uv run python src/energy_demand_forecasting/pipelines/etl.py
