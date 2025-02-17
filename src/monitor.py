"""This module provides functionality to evaluate the current forecasting model."""

import random

import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.data import DATA_CONFIG, fetch_data, transform_data
from src.inference import generate_multi_step_forecast, generate_one_step_forecast
from src.logger import logger
from src.model import build_model, compute_metrics, save_model, tune_model


def backtest_model(
    n_tests: int,
    target_col: str = DATA_CONFIG.target_column,
    timestamp_col: str = DATA_CONFIG.timestamp_column
) -> tuple[pd.DataFrame, str]:
    """Backtests the current model on historical data, across different forecast
    horizons, by comparing its R² to the R² produced by the naive forecast.

    Args:
        n_tests (int): Number of forecast horizons to backtest. 
        target_col (str, optional): Name of the column that contains the target variable.
        Defaults to DATA_CONFIG.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to DATA_CONFIG.timestamp_column.

    Returns:
        dict[int, tuple[float, float]]: _description_
    """
    try:
        data: pd.DataFrame = fetch_data().pipe(transform_data)
        report: dict[int, tuple[float, float]] = {}
        for forecast_horizon in range(1, n_tests + 1):
            splits: dict[str, pd.Timestamp] = {
                company_id: (
                    data.query(f"company_id == '{company_id}'")[timestamp_col].max()
                    - pd.Timedelta(hours=forecast_horizon)
                )
                for company_id in sorted(data["company_id"].unique())
            }

            # iterate over each ID and create a test set
            dfs: list[tuple[pd.DataFrame, pd.DataFrame]] = [
                (
                    data.query(
                        f"company_id == '{company_id}' \
                        & {timestamp_col} <= @pd.Timestamp('{split}')"
                    ),
                    data.query(
                        f"company_id == '{company_id}' \
                        & {timestamp_col} > @pd.Timestamp('{split}')"
                    )
                )
                for company_id, split in splits.items()
            ]

            # vertically concatenate the train sets into a single train set; same for the test sets
            train_data: pd.DataFrame = pd.concat([tup[0] for tup in dfs], axis=0, ignore_index=True)
            test_data: pd.DataFrame = pd.concat([tup[1] for tup in dfs], axis=0, ignore_index=True)

            # a dictionary to map each company ID to its last known value
            naive_forecaster: dict[str, float] = {
                company_id: (
                    train_data
                    .query(
                        f"company_id == '{company_id}' \
                        & {timestamp_col} == @pd.Timestamp('{split}')"
                    )
                    [target_col]
                    .squeeze()
                )
                for company_id, split in splits.items()
            }

            # a pd.DataFrame that contains the target, naive forecast, and model forecast
            forecast_data: pd.DataFrame = (
                test_data
                .assign(naive=test_data["company_id"].map(naive_forecaster))
                .merge(
                    train_data.pipe(generate_multi_step_forecast, forecast_horizon),
                    how="inner",
                    on=["company_id", timestamp_col]
                )
                .rename(columns={target_col: "target", "forecast": "model"})
                [[
                    "company_id",
                    timestamp_col,
                    "target",
                    "naive",
                    "model"
                ]]
            )
            report[forecast_horizon] = (
                compute_metrics(forecast_data["target"], forecast_data["model"]).get("r_squared"),
                compute_metrics(forecast_data["target"], forecast_data["naive"]).get("r_squared")
            )
        response: str = (
            pd.DataFrame
            .from_dict(report, orient="index", columns=["r2_model", "r2_naive"])
            .reset_index(names="forecast_horizon")
            .rename(columns={"r2_model": "pass", "r2_naive": "fail"})
            .drop(columns="forecast_horizon")
            .mean()
            .to_frame(name="r2_mean")
            .sort_values(by="r2_mean", ascending=False)
            .index[0]
        )
        return data, response
    except Exception as e:
        raise e


def evaluate_model(
    target_col: str = DATA_CONFIG.target_column,
    timestamp_col: str = DATA_CONFIG.timestamp_column
) -> None:
    """Evaluates the current model on historical data and replaces it if its
    predictions are worse than the naive forecast.

    NOTE: this function encapsulates the training pipeline, and in an end-to-end ML project,
    should be a GitHub Actions workflow that runs on a consistent cron job.

    Args:
        target_col (str, optional): Name of the column that contains the target variable.
        Defaults to DATA_CONFIG.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to DATA_CONFIG.timestamp_column.
    """
    try:
        # backtest the current model
        data, response = backtest_model(random.choice(range(4, 25)))

        # if the current model passes backtesting, do nothing
        if response == "pass":
            logger.info("The current forecasting model is fine.")
        # otherwise, replace it
        else:
            logger.info("The current forecasting model is unsatisfactory and will be replaced.")
            train_dfs: list[pd.DataFrame] = []
            test_dfs: list[pd.DataFrame] = []
            for company_id in sorted(data["company_id"].unique()):
                split: pd.Timestamp = data.query(f"company_id=='{company_id}'")[timestamp_col].max()
                train_dfs.append(
                    data.query(
                        f"company_id == '{company_id}' \
                        & {timestamp_col} < @pd.Timestamp('{split}')"
                    )
                )
                test_dfs.append(
                    data.query(
                        f"company_id == '{company_id}' \
                        & {timestamp_col} == @pd.Timestamp('{split}')"
                    )
                )
            train_data: pd.DataFrame = pd.concat(train_dfs, axis=0, ignore_index=True)
            test_data: pd.DataFrame = pd.concat(test_dfs, axis=0, ignore_index=True)

            # create a replacement model, with default hyperparameters, and ...
            # generate its one-step forecast
            default_replacement: CatBoostRegressor | LGBMRegressor | XGBRegressor = (
                build_model(train_data)
            )
            save_model(default_replacement)
            forecast_data: pd.DataFrame = (
                train_data
                .pipe(generate_one_step_forecast)
                .rename(columns={"forecast": "default_replacement"})
            )

            # create a replacement model, with Bayesian-tuned hyperparameters, and ...
            # generate its one-step forecast
            tuned_replacement: CatBoostRegressor | LGBMRegressor | XGBRegressor = (
                tune_model(train_data)
            )
            save_model(tuned_replacement)
            forecast_data = (
                forecast_data
                .merge(
                    (
                        train_data
                        .pipe(generate_one_step_forecast)
                        .rename(columns={"forecast": "tuned_replacement"})
                    ),
                    how="inner",
                    on=["company_id", timestamp_col]
                )
                .merge(
                    test_data[["company_id", timestamp_col, target_col]],
                    how="inner",
                    on=["company_id", timestamp_col]
                )
                [[
                    "company_id",
                    timestamp_col,
                    target_col,
                    "default_replacement",
                    "tuned_replacement"
                ]]
            )

            # a dictionary that maps each replacement model to its corresponding Python object
            models: dict[str, CatBoostRegressor | LGBMRegressor | XGBRegressor] = {
                "default_replacement": default_replacement,
                "tuned_replacement": tuned_replacement
            }

            # a dictionary where each replacement model is mapped to its corresponding test RMSE ...
            # and is sorted w.r.t. the test RMSEs, in ascending order
            report: dict[str, float] = {
                col: compute_metrics(forecast_data[target_col], forecast_data[col]).get("rmse")
                for col in ["default_replacement", "tuned_replacement"]
            }
            report = dict(sorted(report.items(), key=lambda kv: kv[1]))

            # save the replacement model that produced the lowest test RMSE
            replacement: str = list(report.keys())[0]
            save_model(models.get(replacement))
    except Exception as e:
        raise e


if __name__ == "__main__":
    evaluate_model()
