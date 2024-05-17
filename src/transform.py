"""Data transformation"""

import random

import pandas as pd

from src.ingest import ingest_data, query_ids
from src.logger import logging


def create_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates datetime/temporal features based on a pd.DatetimeIndex

    Args:
        data (pd.DataFrame): pd.DataFrame that contains a pd.DatetimeIndex

    Returns:
        pd.DataFrame: pd.DataFrame that contains two datetime features,
        'hour' and 'time_of_day'
    """
    try:
        return pd.DataFrame(
            {
                "hour": data.index.hour,
                "time_of_day": [
                    (
                        1
                        if x in range(5, 12)
                        else 2 if x in range(12, 17) else 3 if x in range(17, 21) else 4
                    )
                    for x in data.index.hour
                ],
            },
            index=data.index,
        )
    except Exception as e:
        raise e


def transform_data(
    data: pd.DataFrame, target_name: str, max_lag: int = 24, window_size: int = 4
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transforms a stationary univariate time series into a matrix of lag features,
    window features, datetime features, and the corresponding target

    Args:
        data (pd.DataFrame): pd.DataFrame that contains the stationary univariate
        time series of interest
        target_name (str): Column name of the stationary univariate time series
        max_lag (int): Number of lag features to create. Defaults to 24
        window_size (int): Window size to compute window features. Defaults to 4

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and the corresponding
        target vector
    """
    try:
        logging.info("Transforming the data...")
        # create the lag features
        lags: list[pd.Series] = [
            data[target_name].dropna().shift(periods=p) for p in reversed(range(1, max_lag + 1))
        ]
        df_lags: pd.DataFrame = pd.concat(lags, axis=1).dropna()
        df_lags.columns = [f"lag_{p}" for p in reversed(range(1, max_lag + 1))]

        # create the window features, i.e., the 'average lag' features
        avg_lags: list[pd.Series] = [
            df_lags.iloc[:, i:].mean(axis=1) for i in range(-len(df_lags.columns), 0, window_size)
        ]
        df_windows: pd.DataFrame = pd.concat(avg_lags, axis=1)
        df_windows.columns = [
            f"avg_{p}_lags" for p in reversed(range(window_size, max_lag + 1, window_size))
        ]

        # create the datetime/temporal features
        df_temporal: pd.DataFrame = create_datetime_features(df_lags)

        # create the feature matrix by horizontally concatenating the 'df_temporal', ...
        # 'df_windows', and 'df_lags' pd.DataFrames
        feature_matrix: pd.DataFrame = pd.concat((df_temporal, df_windows, df_lags), axis=1)

        # extract the corresponding target vector
        target_vector: pd.Series = data.loc[feature_matrix.index, target_name]
        logging.info("Data transformation complete.")
        return feature_matrix, target_vector
    except Exception as e:
        raise e


if __name__ == "__main__":
    ids: list[str] = query_ids()
    df_stationary, target = ingest_data(random.choice(ids))
    x_matrix, y_vector = transform_data(df_stationary, target)
    print(pd.concat((x_matrix, y_vector), axis=1).head())
