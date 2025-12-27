.PHONY: install check clean features update_features

install: pyproject.toml
	uv sync

check: .venv
	uv run isort src && uv run ruff check src

clean:
	rm -rf `find . -type d -name __pycache__` ; rm -rf .ruff_cache

features:
	dvc pull && uv run python -Wignore src/pipelines/feature.py

update_features:
	dvc add ./artifacts && \
	git add artifacts.dvc && \
	git commit -m "Executing the feature pipeline and updating ./artifacts.dvc" && \
	dvc push && \
	git push