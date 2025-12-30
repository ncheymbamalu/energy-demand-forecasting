"""This module contains functionality to backtest the current forecasting model."""

from pathlib import Path

import numpy as np
import polars as pl

from omegaconf import DictConfig

from src.config import load_params
from src.data import load_preprocessed_data
from src.data import params as data_params
from src.data import split_data, transform_data
from src.inference import get_multi_step_forecast
from src.utils import compute_evaluation_metrics

params: DictConfig = load_params(Path(__file__).stem)


def backtest_model():
    """Backtests the current forecasting model on historical data, that is,
    the previous 12 time steps, and evaluates its R² against a user-defined
    threshold and the R² produced by the naive forecast.

    Returns:
        bool: Output that determines if the current forecasting model is sufficient.
    """
    try:
        # load and transform the data
        data: pl.DataFrame = load_preprocessed_data().pipe(transform_data)

        # get the threshold, target, and temporal column
        threshold: float = params.threshold
        target: str = data_params.target_column
        temporal_col: str = data_params.temporal_column

        # backtest the current model on the previous 12 time steps
        metrics: list[float] = []
        naive_metrics: list[float] = []
        evals: list[bool] = []
        for idx, eval_size in enumerate(range(1, 13)):
            train_data, eval_data = data.pipe(split_data, eval_size)
            eval_data = (
                eval_data
                .select("company_id", temporal_col, target)

                # model-based forecast
                .join(
                    other=(
                        train_data
                        .pipe(get_multi_step_forecast, eval_size)
                        .select("company_id", temporal_col, "forecast")
                    ),
                    how="inner",
                    on=["company_id", temporal_col],
                    maintain_order="left"
                )

                # naive forecast
                .join(
                    other=(
                        train_data
                        .select("company_id", temporal_col, target)
                        .join(
                            other=(
                                train_data
                                .group_by("company_id")
                                .agg(pl.col(temporal_col).max())
                            ),
                            how="inner",
                            on=["company_id", temporal_col],
                            maintain_order="left"
                        )
                        .select(
                            "company_id",
                            pl.col(target).alias("naive_forecast")
                        )
                    ),
                    how="inner",
                    on="company_id",
                    maintain_order="left"
                )
                .select(
                    "company_id",
                    temporal_col,
                    "naive_forecast",
                    "forecast",
                    target
                )
            )
            metrics.append(
                compute_evaluation_metrics(eval_data[target], eval_data["forecast"]).get("r2")
            )
            naive_metrics.append(
                compute_evaluation_metrics(eval_data[target], eval_data["naive_forecast"]).get("r2")
            )
            evals.append(metrics[idx] > threshold)
        condition_one: bool = np.mean(metrics) > max(threshold, np.mean(naive_metrics))
        condition_two: bool = np.mean(evals) == 1
        return sum([condition_one, condition_two]) == 2
    except Exception as e:
        raise e
