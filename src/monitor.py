"""This module provides functionality to evaluate the current forecasting model."""

import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import Paths
from src.data import DATA_CONFIG, fetch_data, transform_data
from src.inference import generate_one_step_forecast, load_model
from src.logger import logger
from src.model import build_model, compute_metrics, save_model, tune_model


def evaluate_model(
    target_col: str = DATA_CONFIG.target_column,
    timestamp_col: str = DATA_CONFIG.timestamp_column
) -> None:
    """Evaluates the ML model that's currently in production by comparing its one-step
    forecast to the one-step forecast of two potential replacement models, one with default
    hyperparameters and the other with Bayesian-tuned hyperparameters. The model with the
    lowest test RMSE is retained.

    NOTE: this function encapsulates the training pipeline, and in an end-to-end ML project,
    should be a GitHub Actions workflow that runs on a consistent cron job.

    Args:
        target_col (str, optional): Name of the column that contains the target variable.
        Defaults to INGEST_CONFIG.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to INGEST_CONFIG.timestamp_column.
    """
    try:
        # fetch the latest pre-processed data and transform it into features and targets
        data: pd.DataFrame = fetch_data().pipe(transform_data)

        # map each company ID to its max timestamp
        company_id_to_max_timestamp: dict[str, pd.Timestamp] = {
            company_id: data.query(f"company_id == '{company_id}'")[timestamp_col].max()
            for company_id in sorted(data["company_id"].unique())
        }

        # iterate over each company ID and create its train and test set
        dfs: list[list[pd.DataFrame]] = [
            [
                data.query(
                    f"company_id == '{company_id}' & {timestamp_col} < @pd.Timestamp('{max_ts}')"
                ),
                data.query(
                    f"company_id == '{company_id}' & {timestamp_col} == @pd.Timestamp('{max_ts}')"
                )
            ]
            for company_id, max_ts in company_id_to_max_timestamp.items()
        ]

        # vertically concatenate the train sets, one per company ID, to create a single train ...
        # set, and do the same for the test sets
        train_data: pd.DataFrame = pd.concat([lst[0] for lst in dfs], axis=0, ignore_index=True)
        test_data: pd.DataFrame = pd.concat([lst[1] for lst in dfs], axis=0, ignore_index=True)

        # if a ML model doesn't exist, create one
        if not Paths.MODEL.exists():
            build_model(train_data)
        else:
            # otherwise, load the current model and generate its forecast
            current_model: CatBoostRegressor | LGBMRegressor | XGBRegressor = load_model()
            forecast: pd.DataFrame = (
                train_data
                .pipe(generate_one_step_forecast)
                .rename(columns={"forecast": "current_model"})
            )

            # create the current model's potential replacement, and generate its forecast
            default_replacement: CatBoostRegressor | LGBMRegressor | XGBRegressor = (
                build_model(train_data)
            )
            default_forecast: pd.DataFrame = (
                train_data
                .pipe(generate_one_step_forecast)
                .rename(columns={"forecast": "default_replacement"})
            )

            # create another potential replacement, with Bayesian-tuned hyperparameters, and ...
            # generate its forecast
            tuned_replacement: CatBoostRegressor | LGBMRegressor | XGBRegressor = (
                tune_model(train_data)
            )
            tuned_forecast: pd.DataFrame = (
                train_data
                .pipe(generate_one_step_forecast)
                .rename(columns={"forecast": "tuned_replacement"})
            )
            forecast = (
                forecast
                .merge(
                    default_forecast,
                    how="inner",
                    on=["company_id", timestamp_col]
                )
                .merge(
                    tuned_forecast,
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
                    "current_model",
                    "default_replacement",
                    "tuned_replacement"
                ]]
            )

            # a dictionary that (A) maps each model, that is, the current one, a potential ...
            # replacement with default hyperparameters, and a potential replacement with ...
            # Bayesian-tuned hyperparameters, to its corresponding test RMSE, and ...
            # (B) is sorted w.r.t. its values, i.e., the test RMSEs, in ascending order
            report: dict[str, float] = {
                col: compute_metrics(forecast[target_col], forecast[col]).get("rmse")
                for col in ["current_model", "default_replacement", "tuned_replacement"]
            }
            report = dict(sorted(report.items(), key=lambda kv: kv[1]))
            model = (
                default_replacement if list(report.keys())[0] == "default_replacement"
                else tuned_replacement if list(report.keys())[0] == "tuned_replacement"
                else current_model
            )
            if list(report.keys())[0] == "current_model":
                logger.info("The current forecasting model is fine.")
                save_model(model)
            else:
                logger.info("The current forecasting model is unsatisfactory and will be replaced.")
                save_model(model)
    except Exception as e:
        raise e


if __name__ == "__main__":
    evaluate_model()
