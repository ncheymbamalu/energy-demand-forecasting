.PHONY: install check clean data_pipeline monitor_pipeline update_artifacts

install: pyproject.toml
	uv sync

check: .venv
	uv run isort src && uv run ruff check src

clean:
	rm -rf `find . -type d -name __pycache__` ; rm -rf .ruff_cache

data_pipeline:
	dvc pull && \
	uv run python -Wignore src/pipelines/feature.py && \
	uv run python -Wignore src/pipelines/inference.py

monitor_pipeline:
	dvc pull && \
	uv run python -Wignore src/pipelines/train.py && \
	uv run python -Wignore src/pipelines/backfill.py

update_artifacts:
	dvc add ./artifacts && \
	git add artifacts.dvc && \
	git commit -m "Updating artifacts.dvc" && \
	dvc push && \
	git push
