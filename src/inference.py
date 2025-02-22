"""This module provides functionality to generate a one-step or multi-step forecast."""

import string

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import data_config
from src.logger import logger
from src.model import load_model


class NaiveForecast:
    """A class to serve as the baseline model."""

    def __str__(self):
        return "NaiveForecast"

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Returns the naive forecast.

        Args:
            data (pd.DataFrame): Dataset containing the 1st lagged values.

        Returns:
            np.ndarray: Naive forecast.
        """
        return data["lag_1"].values


def generate_one_step_forecast(
    data: pd.DataFrame,
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column
) -> pd.DataFrame:
    """Returns a pd.DataFrame that contains each company ID's one-step forecast, that is,
    the predicted energy demand for the upcoming hour.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.

    Returns:
        pd.DataFrame: Dataset containing the one-step (one hour) forecasted energy demand
        for each company ID.
    """
    try:
        # load the model
        model: CatBoostRegressor | LGBMRegressor | XGBRegressor = load_model()

        # a list of all the features the model was trained on
        features: list[str] = (
            model.feature_names_ if isinstance(model, CatBoostRegressor)
            else model.feature_names_in_.tolist()
        )

        # a list of lag features
        lag_features: list[str] = [col for col in features if col.startswith("lag")]

        # a list of window features ...
        # as well as the 'start', 'step', and 'end' variables that are needed to create them
        window_features: list[str] = [col for col in features if col.endswith("lags")]
        digit_filter: dict[int, None] = str.maketrans(
            "", "", string.whitespace + string.punctuation + string.ascii_letters
        )
        start: int = min(int(col.translate(digit_filter)) for col in window_features)
        step = start
        end: int = len(lag_features) + 1

        # a list of the final output columns
        output_cols: list[str] = data.drop(columns=target_col).columns.tolist() + ["forecast"]

        # an empty list to store each (1, D) forecasted record, one per company ID
        dfs: list[pd.DataFrame] = []
        for company_id in sorted(data["company_id"].unique()):
            # extract the current record, which is the last row
            x: pd.DataFrame = data.query(f"company_id == '{company_id}'").iloc[-1:, :]

            # create the forecasted record's timestamp and corresponding datetime features
            timestamp: pd.Timestamp = x.squeeze()[timestamp_col] + pd.Timedelta(hours=1)
            est_hour: int = timestamp.hour + data_config.utc_to_est
            x_datetime: list[int] = [
                timestamp.day_of_week + 1,
                1 if est_hour in range(5, 12)  # morning
                else 2 if est_hour in range(12, 17)  # afternoon
                else 3 if est_hour in range(17, 21)  # evening
                else 4,  # night
                timestamp.hour
            ]

            # create the forecasted record's lag features
            x_lag: list[float] = x[lag_features[1:] + [target_col]].squeeze().tolist()

            # create the forecasted record's window features
            x_window: list[float] | list[list[float]] = [
                [
                    pd.Series(x_lag[-lag:]).mean(),
                    pd.Series(x_lag[-lag:]).std()
                ]
                for lag in reversed(range(start, end, step))
            ]
            x_window = np.array(x_window).ravel().tolist()

            # horizontally concatenate the forecasted record's datetime, window, and lag features
            # NOTE: this (1, D) pd.DataFrame is the model's input to generate the forecast
            x: pd.DataFrame = pd.DataFrame(data=[x_datetime + x_window + x_lag], columns=features)

            # add the forecast, company ID, and timestamp, and re-arrange the output columns
            x["forecast"] = max(0, float(round(model.predict(x)[0])))
            x["company_id"] = company_id
            x[timestamp_col] = timestamp
            x = x[output_cols]

            # append the forecasted record to the 'dfs' list
            dfs.append(x)
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e


def generate_multi_step_forecast(
    data: pd.DataFrame,
    forecast_horizon: int,
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column
) -> pd.DataFrame:
    """Returns a pd.DataFrame that contains a multi-step forecast for each company ID.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.
        forecast_horizon (int, optional): Number of time steps (hours) to forecast into the future.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.

    Returns:
        pd.DataFrame: Dataset containing the multi-step (multi-hour) forecasted energy demand
        for each company ID.
    """
    try:
        logger.info(f"Generating each company ID's {forecast_horizon} hour forecast.")
        data = data.pipe(generate_one_step_forecast).rename(columns={"forecast": target_col})
        dfs: list[pd.DataFrame] = [data]
        for _ in range(forecast_horizon - len(dfs)):
            data = data.pipe(generate_one_step_forecast).rename(columns={"forecast": target_col})
            dfs.append(data)
        return (
            pd.concat(dfs, axis=0)
            .rename(columns={target_col: "forecast"})
            .sort_values(by=["company_id", timestamp_col])
            .reset_index(drop=True)
            [["company_id", timestamp_col, "forecast"]]
        )
    except Exception as e:
        raise e
