"""This module contains functionality to monitor the model and forecast data artifacts."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from omegaconf import DictConfig

from src.config import Paths, load_params
from src.data import load_forecasts, load_preprocessed_data
from src.data import params as data_params
from src.data import split_data, transform_data
from src.forecast import get_multi_step_forecast, get_one_step_forecast
from src.logger import logger
from src.utils import compute_evaluation_metrics

params: DictConfig = load_params(Path(__file__).stem)


def backtest_model(data: pl.DataFrame):
    """Backtests the current forecasting model on historical data, that is,
    the previous 12 time steps, and evaluates its R² against a user-defined
    threshold and the R² produced by the naive forecast.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.

    Returns:
        bool: Output that determines if the current forecasting model is sufficient.
    """
    try:
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


def backfill_forecasts() -> None:
    """Identifies and backfills the forecast data's missing values."""
    try:
        # get the target, temporal column, and lookback value
        target: str = data_params.target_column
        temporal_col: str = data_params.temporal_column
        lookback: int = data_params.lookback

        # load the latest pre-processed and forecast data
        processed_data: pl.DataFrame = load_preprocessed_data()
        forecast_data: pl.DataFrame = load_forecasts()

        # create a pl.DataFrame that identifies records that are missing its one-step forecast
        backfill_data: pl.DataFrame = (
            processed_data
            .join(
                other=(
                    forecast_data
                    .group_by("company_id")
                    .agg(
                        pl.col(temporal_col).min().alias(f"min_{temporal_col}"),
                        pl.col(temporal_col).max().alias(f"max_{temporal_col}")
                    )
                ),
                how="inner",
                on="company_id",
                maintain_order="left"
            )
            .filter(
                pl.col(temporal_col)
                .is_between(pl.col(f"min_{temporal_col}"), pl.col(f"max_{temporal_col}"))
            )
            .drop(f"min_{temporal_col}", f"max_{temporal_col}", target)
            .join(
                other=forecast_data,
                how="left",
                on=["company_id", temporal_col],
                maintain_order="left"
            )
            .filter(pl.col("forecast").is_null())
        )
        if backfill_data.is_empty():
            logger.info("The forecasts are up-to-date. Skipping the backfilling process.")
        else:
            dfs: list[pl.DataFrame] = []
            hrs_to_backfill: list[datetime] = sorted(backfill_data[temporal_col].unique())
            logger.info(
                f"The one-step forecast is missing for the following hour(s): \
{', '.join(f'<green>{str(hr)}</green>' for hr in hrs_to_backfill)}.\nStarting the backfilling \
process."
            )
            for hr in hrs_to_backfill:
                end: datetime = hr - timedelta(hours=1)
                start: datetime = end - timedelta(days=lookback)
                dfs.append(
                    pl.read_parquet(Paths.PROCESSED_DATA)
                    .filter(pl.col(temporal_col).is_between(start, end))
                    .pipe(transform_data)
                    .pipe(get_one_step_forecast)
                    .select(
                        "company_id",
                        temporal_col,
                        pl.col("forecast").alias("backfilled_forecast")
                    )
                )
            backfill_data = (
                pl.concat(dfs, how="vertical")
                .join(
                    other=backfill_data,
                    how="inner",
                    on=["company_id", temporal_col],
                    maintain_order="right"
                )
                .select(
                    "company_id",
                    temporal_col,
                    pl.coalesce("forecast", "backfilled_forecast").alias("forecast")
                )
            )
            forecast_data = (
                pl.concat((forecast_data, backfill_data), how="vertical")
                .sort(by=["company_id", temporal_col])
            )
            logger.info("The backfilling process is complete. The forecasts are now up-to-date.")
            forecast_data.write_parquet(Paths.FORECAST_DATA)
    except Exception as e:
        raise e
