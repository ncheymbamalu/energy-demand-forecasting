"""Recursive/multi-step forecasting"""

import random

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from src.ingest import ingest_data, query_ids
from src.logger import logging
from src.train import train_model
from src.transform import create_datetime_features, transform_data


# pylint: disable=too-many-locals
def generate_forecast(
    model: XGBRegressor,
    feature_matrix: pd.DataFrame,
    target_vector: pd.Series,
    stationary_data: pd.DataFrame,
    forecast_horizon: int = 24,
) -> pd.Series:
    """
    Returns the recursive forecast

    Args:
        model (XGBRegressor): Regressor with Bayesian optimized hyperparameters
        feature_matrix (pd.DataFrame): Matrix of lag features, window features,
        and datetime features
        target_vector (pd.Series): Target vector
        stationary_data (pd.DataFrame): Original stationary univariate time
        series that 'feature_matrix' and 'target_vector' are based on
        forecast_horizon (int): Number of time steps to forecast into
        the future. Defaults to 24

    Returns:
        pd.Series: Recursive forecast
    """
    try:
        logging.info("Forecasting initiated.")
        # fit the model to the entire dataset
        model.fit(feature_matrix.values, target_vector.values)

        # create the out-of-sample timestamps
        forecast_indices: pd.DatetimeIndex = pd.date_range(
            feature_matrix.index[-1] + pd.Timedelta(hours=1),
            feature_matrix.index[-1] + pd.Timedelta(hours=forecast_horizon),
            freq="H",
        )

        # create the out-of-sample datetime features
        x_dts: list[list[int]] = create_datetime_features(
            pd.DataFrame(index=forecast_indices)
        ).values.tolist()

        # create the lag features for the 1st input
        lag_cols: list[str] = [col for col in feature_matrix.columns if col.startswith("lag")]
        x_lag: list[float] = feature_matrix.iloc[-1][lag_cols].tolist()[1:] + [
            target_vector.iloc[-1]
        ]

        # create the window features, i.e., the 'average lag' features, for the 1st input
        window_indices: list[int] = [
            -int(col.split("_")[1]) for col in feature_matrix.columns if "avg" in col
        ]
        x_window: list[float] = [np.mean(x_lag[idx:]) for idx in window_indices]

        # create the 1st input
        x: np.ndarray = np.array(x_dts[0] + x_window + x_lag)

        # get the 1st prediction and add it to a list named, 'predictions'
        yhat: float = model.predict(x.reshape(1, -1))[0]
        predictions: list[float] = [yhat]

        # get the remaining predictions
        for x_dt in x_dts[1:]:
            x_lag = x_lag[1:] + [yhat]
            x_window = [np.mean(x_lag[idx:]) for idx in window_indices]
            x = np.array(x_dt + x_window + x_lag)
            yhat = model.predict(x.reshape(1, -1))[0]
            predictions.append(yhat)

        # create the recursive forecast
        original_target: str = stationary_data.columns[0]
        forecast: np.ndarray = (
            stationary_data.loc[feature_matrix.index[-1], original_target] + np.cumsum(predictions)
            if target_vector.name == "diff"
            else np.array(predictions)
        )
        logging.info("The forecast has been generated.")
        return pd.Series(forecast, index=forecast_indices)
    except Exception as e:
        raise e


if __name__ == "__main__":
    ids: list[str] = query_ids()
    df_stationary, target = ingest_data(random.choice(ids))
    x_matrix, y_vector = transform_data(df_stationary, target)
    reg: XGBRegressor = train_model(x_matrix, y_vector)
    recursive_forecast: pd.Series = generate_forecast(reg, x_matrix, y_vector, df_stationary)
    print(recursive_forecast)
