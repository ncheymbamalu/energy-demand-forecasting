"""This script executes the training pipeline."""

import polars as pl

from src.data import load_preprocessed_data, transform_data
from src.logger import logger
from src.model import (
    build_model,
    create_model_metadata,
    save_model,
    save_model_metadata,
    tune_model,
)
from src.monitor import backtest_model


def main() -> None:
    """Executes the training pipeline."""
    try:
        if backtest_model():
            logger.info("The current forecasting model is fine.")
        else:
            logger.info("The current forecasting model is unsatisfactory and will be replaced.")
            data: pl.DataFrame = load_preprocessed_data().pipe(transform_data)
            model, metric = tune_model(data, build_model(data))
            data = create_model_metadata(data, model, metric)
            save_model(model)
            save_model_metadata(data)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
