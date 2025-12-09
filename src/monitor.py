"""This module provides functionality to evaluate the current forecasting model."""

import random

import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import data_config
from src.data import fetch_data, transform_data
from src.inference import generate_multi_step_forecast, generate_one_step_forecast
from src.logger import logger
from src.model import build_model, tune_model
from src.utils import compute_metrics, save_model, split_data


def backtest_model(
    n_tests: int,
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column
) -> tuple[pd.DataFrame, str]:
    """Backtests the current model on historical data, across different forecast
    horizons, by comparing its R² to the R² produced by the naive forecast.

    Args:
        n_tests (int): Number of forecast horizons to backtest. 
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.

    Returns:
        tuple[pd.DataFrame, str]: _description_
    """
    try:
        data: pd.DataFrame = fetch_data().pipe(transform_data)
        report: dict[int, tuple[float, float]] = {}
        for forecast_horizon in range(1, n_tests + 1):

            # split the data into train and test sets
            train_data, test_data = split_data(data, forecast_horizon)

            # a dictionary to map each company ID to its last known value
            naive_forecaster: dict[str, float] = {
                company_id: train_data.query(f"company_id == '{company_id}'").iloc[-1][target_col]
                for company_id in train_data["company_id"].unique()
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
            .rename(columns={"r2_model": "pass", "r2_naive": "fail"})
            .mean()
            .to_frame(name="r2_mean")
            .sort_values(by="r2_mean", ascending=False)
            .index[0]
        )
        return data, response
    except Exception as e:
        raise e


def evaluate_model(
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column
) -> None:
    """Evaluates the current model on historical data and replaces it if its
    predictions are worse than the naive forecast.

    NOTE: this function encapsulates the training pipeline, and in an end-to-end ML project,
    should be a GitHub Actions workflow that runs on a consistent cron job.

    Args:
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.
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
            # split the 'data' pd.DataFrame into train and test sets
            train_data, test_data = split_data(data, 1)

            # create a replacement model, with default hyperparameters, and ...
            # generate its one-step forecast
            model: CatBoostRegressor | LGBMRegressor | XGBRegressor = (
                build_model(train_data)
            )
            save_model(model)
            forecast_data: pd.DataFrame = (
                train_data
                .pipe(generate_one_step_forecast)
                .rename(columns={"forecast": "model"})
            )

            # create a replacement model, with Bayesian-tuned hyperparameters, and ...
            # generate its one-step forecast
            tuned_model: CatBoostRegressor | LGBMRegressor | XGBRegressor = (
                tune_model(train_data, model)
            )
            save_model(tuned_model)
            forecast_data = (
                forecast_data
                .merge(
                    (
                        train_data
                        .pipe(generate_one_step_forecast)
                        .rename(columns={"forecast": "tuned_model"})
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
                    "model",
                    "tuned_model"
                ]]
            )

            # a dictionary that maps each potential replacement model to its Python object
            models: dict[str, CatBoostRegressor | LGBMRegressor | XGBRegressor] = {
                "model": model, "tuned_model": tuned_model
            }

            # a dictionary where each potential replacement model is mapped to its test RMSE ...
            # and is sorted w.r.t. the test RMSEs, in ascending order
            report: dict[str, float] = {
                col: compute_metrics(forecast_data[target_col], forecast_data[col]).get("rmse")
                for col in ["model", "tuned_model"]
            }
            report = dict(sorted(report.items(), key=lambda kv: kv[1]))

            # save the replacement model that produced the lowest test RMSE
            replacement: str = list(report.keys())[0]
            save_model(models.get(replacement))
    except Exception as e:
        raise e


if __name__ == "__main__":
    evaluate_model()
