"""This module contains functionality for making forecasts."""

import numpy as np
import polars as pl

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.data import params as data_params
from src.logger import logger
from src.model import load_model
from src.utils import encode_hour


def get_one_step_forecast(data: pl.DataFrame) -> pl.DataFrame:
    """Returns a pl.DataFrame that contains each company IDs's one-step forecast, that is,
    the predicted energy demand one hour into the future.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.

    Returns:
        pl.DataFrame: One-step forecast for each company ID.
    """
    try:
        # load the model
        model: LGBMRegressor | XGBRegressor = load_model()

        # get the features, lag features, target column, and temporal column
        features: list[str] = data.drop(data_params.columns).columns
        lag_features: list[str] = [feature for feature in features if feature.startswith("lag")]
        target: str = data_params.target_column
        temporal_col: str = data_params.temporal_column

        # an empty list to store each company ID's one-step forecast
        dfs: list[pl.DataFrame] = []

        # iterate over each company ID, and ...
        for company_id in sorted(data["company_id"].unique()):
            # extract the record for the current time step
            x: pl.DataFrame = data.filter(pl.col("company_id").eq(pl.lit(company_id)))[-1:, :]

            # create the lag features for the next time step
            lag_data: pl.DataFrame = x.select(lag_features + [target])[:, 1:]
            lag_data.columns = lag_features

            # create the window features (average lags) for the next time step
            start = step = data_params.step
            window_dfs: list[pl.DataFrame] = [
                lag_data[:, -lag:].mean_horizontal().to_frame(f"avg_{lag}_lags")
                for lag in reversed(range(start, len(lag_features) + 1, step))
            ]
            window_data: pl.DataFrame = pl.concat(window_dfs, how="horizontal")

            # create the datetime features for the next time step
            datetime_data: pl.DataFrame = (
                x
                .with_columns(
                    (pl.col(temporal_col) + pl.duration(hours=1)).alias(temporal_col),
                    (
                        (pl.col(temporal_col) + pl.duration(hours=1))
                        .dt.replace_time_zone("UTC")
                        .dt.convert_time_zone("EST")
                        .dt.hour()
                        .alias("hour")
                    )
                )
                .with_columns(
                    (
                        pl.col("hour")
                        .map_elements(lambda hr: encode_hour(hr, "sine"), return_dtype=pl.Float64)
                        .alias("sine_hour")
                    ),
                    (
                        pl.col("hour")
                        .map_elements(lambda hr: encode_hour(hr, "cosine"), return_dtype=pl.Float64)
                        .alias("cosine_hour")
                    )
                )
                .sql(
                    f"""\
                    SELECT
                        '{company_id}' AS company_id,
                        {temporal_col},
                        sine_hour,
                        cosine_hour,
                        CASE
                            WHEN hour >= 5 AND hour < 12 THEN TRUE
                            ELSE FALSE
                        END AS is_morning,
                        CASE
                            WHEN hour >= 12 AND hour < 17 THEN TRUE
                            ELSE FALSE
                        END AS is_afternoon,
                        CASE
                            WHEN hour >= 17 AND hour < 21 THEN TRUE
                            ELSE FALSE
                        END AS is_evening,
                        CASE
                            WHEN hour >= 5 AND hour < 12 THEN FALSE
                            WHEN hour >= 12 AND hour < 17 THEN FALSE
                            WHEN hour >= 17 AND hour < 21 THEN FALSE
                            ELSE TRUE
                        END AS is_night
                    FROM self\
                    """
                )
            )

            # create the record for the next time step
            x = pl.concat((datetime_data, window_data, lag_data), how="horizontal")

            # compute the forecast for the next time step
            yhat: np.float64 = max(0.0, round(model.predict(x.select(features))[0], 4))
            x = x.with_columns(pl.lit(yhat.item()).alias("forecast"))

            # add the record to the 'dfs' list
            dfs.append(x)
        return pl.concat(dfs, how="vertical").sort(by="company_id")
    except Exception as e:
        raise e


def get_multi_step_forecast(data: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Returns a pl.DataFrame that contains a multi-step forecast for each company ID.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.
        horizon: Forecast horizon, the number of time steps (hours) to forecast
        into the future.

    Returns:
        pl.DataFrame: Multi-step forecast for each company ID.
    """
    try:
        logger.info(f"Generating each company ID's <green>{horizon}</green> hour forecast.")
        target: str = data_params.target_column
        temporal_col: str = data_params.temporal_column
        dfs: list[pl.DataFrame] = [data.pipe(get_one_step_forecast)]
        for idx in range(horizon - len(dfs)):
            dfs.append(dfs[idx].rename({"forecast": target}).pipe(get_one_step_forecast))
        return pl.concat(dfs, how="vertical").sort(by=["company_id", temporal_col])
    except Exception as e:
        raise e
