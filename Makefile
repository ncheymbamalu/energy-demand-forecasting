.PHONY: install check clean data update_data_artifacts train update_model_artifacts

install: pyproject.toml
	uv sync

check: .venv
	uv run isort src && uv run ruff check src

clean:
	rm -rf `find . -type d -name __pycache__` ; rm -rf .ruff_cache

data:
	dvc pull && \
	uv run python -Wignore src/pipelines/feature.py && \
	uv run python -Wignore src/pipelines/inference.py

update_data_artifacts:
	dvc add ./artifacts && \
	git add artifacts.dvc && \
	git commit -m "Executing the feature and inference pipelines" && \
	dvc push && \
	git push

train:
	dvc pull && uv run python -Wignore src/pipelines/train.py

update_model_artifacts:
	dvc add ./artifacts && \
	git add artifacts.dvc && \
	git commit -m "Executing the training pipeline" && \
	dvc push && \
	git push
