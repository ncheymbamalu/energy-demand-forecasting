"""This module contains utility/helper functions."""

import pickle

from pathlib import PosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sqlalchemy import Connection, text
from xgboost import XGBRegressor

from src.config import Paths, data_config, db_config, model_config
from src.database import get_db_connection


# -------------------------------------------------------------------------------------------------
# Data-related utility functions
# -------------------------------------------------------------------------------------------------
def query_data(query: str) -> pd.DataFrame:
    """Queries data from a SQL database and returns it as a pd.DataFrame.

    Args:
        query (str): SQL query.

    Returns:
        pd.DataFrame: Queried data
    """
    try:
        db_connection: Connection = get_db_connection()
        data: pd.DataFrame = pd.DataFrame(db_connection.execute(text(query)))
        db_connection.close()
        return data
    except Exception as e:
        raise e


def create_features(
    data: pd.DataFrame,
    company_id: str,
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column,
    max_lag: int = 24
) -> pd.DataFrame:
    """Creates lag features, window features, and datetime features from a 1-D time series.

    Args:
        data (pd.DataFrame): DataFrame containing a 1-D time series of validated and
        pre-processed hourly energy demand data.
        company_id (str): ID of the company whose data to create features for.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.
        max_lag (int, optional): Maximum number of lag features to create. Defaults to 24.

    Returns:
        pd.DataFrame: DataFrame containing the lag features, window features, and
        datetime features for the input company ID.
    """
    try:
        # create the lag features
        dfs: list[pd.DataFrame] = [
            (
                data
                .query(f"company_id == '{company_id}'")
                .sort_values(by=timestamp_col)
                .set_index(timestamp_col)
                [target_col]
                .shift(periods=lag)
                .to_frame(name=f"lag_{lag}")
            )
            for lag in reversed(range(1, max_lag + 1))
        ]
        df_lag: pd.DataFrame = pd.concat(dfs, axis=1).dropna()

        # create the window features
        # NOTE: the start/step, which is hardcoded and set equal to 4, is a hyperparameter; ...
        # however, different values should be tested, and the values that are tested should be ...
        # a factor of the max_lag
        start = step = 4
        dfs = []
        for lag in reversed(range(start, max_lag + 1, step)):
            df_mean: pd.DataFrame = (
                df_lag.iloc[:, -lag:].mean(axis=1).to_frame(name=f"mean_{lag}_lags")
            )
            df_std: pd.DataFrame = (
                df_lag.iloc[:, -lag:].std(axis=1).to_frame(name=f"std_{lag}_lags")
            )
            dfs.append(pd.concat((df_mean, df_std), axis=1))
        df_window: pd.DataFrame = pd.concat(dfs, axis=1)

        # create the datetime features
        df_datetime: pd.DataFrame = pd.DataFrame(
            data={
                "day_of_week": [idx + 1 for idx in df_lag.index.day_of_week],
                "time_of_day": [
                    1 if hour in range(5, 12)  # morning
                    else 2 if hour in range(12, 17)  # afternoon
                    else 3 if hour in range(17, 21)  # evening
                    else 4  # night
                    for hour in (df_lag.index.hour + data_config.utc_to_est)
                ],
                "hour": df_lag.index.hour
            },
            index=df_lag.index
        )
        output_cols: list[str] = (
            ["company_id", timestamp_col]
            + df_datetime.columns.tolist()
            + df_window.columns.tolist()
            + df_lag.columns.tolist()
        )
        return (
            pd.concat((df_datetime, df_window, df_lag), axis=1)
            .assign(company_id=company_id)
            .sort_index()
            .reset_index()
            [output_cols]
        )
    except Exception as e:
        raise e


def split_data(
    data: pd.DataFrame,
    test_size: int,
    timestamp_col: str = data_config.timestamp_column
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test sets.

    Args:
        data (pd.DataFrame): ML-ready dataset that contains datetime features, window
        features, lag features, and the corresponding target.
        test_size (int): Number of test records (hours) for each company ID.
        Defaults to data_config.test_size. NOTE: the test_size must be a positive integer. 
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test sets.
    """
    try:
        # a dictionary that maps each company ID to its splitting timestamp
        splits: dict[str, pd.Timestamp] = {
            company_id: (
                data.query(f"company_id == '{company_id}'")[timestamp_col].max()
                - pd.Timedelta(hours=test_size)
            )
            for company_id in sorted(data["company_id"].unique())
        }
        train_dfs: list[pd.DataFrame] = []
        test_dfs: list[pd.DataFrame] = []
        for company_id, split in splits.items():
            train_dfs.append(
                data
                .query(
                    f"company_id == '{company_id}' \
                    & {timestamp_col} <= @pd.Timestamp('{split}')"
                )
            )
            test_dfs.append(
                data
                .query(
                    f"company_id == '{company_id}' \
                    & {timestamp_col} > @pd.Timestamp('{split}')"
                )
            )
        train_data: pd.DataFrame = (
            pd.concat(train_dfs, axis=0)
            .sort_values(by=["company_id", timestamp_col])
            .reset_index(drop=True)
        )
        test_data: pd.DataFrame = (
            pd.concat(test_dfs, axis=0, ignore_index=True)
            .sort_values(by=["company_id", timestamp_col])
            .reset_index(drop=True)
        )
        return train_data, test_data
    except Exception as e:
        raise e


# -------------------------------------------------------------------------------------------------
# Model-related utility functions
# -------------------------------------------------------------------------------------------------
def compute_metrics(y: np.ndarray | pd.Series, yhat: np.ndarray | pd.Series) -> dict[str, float]:
    """Computes the root mean squared error, RMSE, and coefficient of
    determination, R², between y and yhat.

    Args:
        y (np.ndarray | pd.Series): Observations.
        yhat (np.ndarray | pd.Series): Predictions.

    Returns:
        dict[str, float]: RMSE and R²
    """
    try:
        t: np.ndarray | pd.Series = y - y.mean()
        sst: float = t.dot(t)
        e: np.ndarray | pd.Series = y - yhat
        sse: float = e.dot(e)
        rmse: float = np.sqrt(sse / e.shape[0])
        r_squared: float = 1 - (sse / sst)
        return {"rmse": round(rmse, 4), "r_squared": round(r_squared, 4)}
    except Exception as e:
        raise e


def get_time_series_splits(
    data: pd.DataFrame,
    timestamp_col: str = data_config.timestamp_column,
    n_splits: int = model_config.n_splits,
    train_size: float = model_config.train_size
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Returns two lists, one containing the train set splits, and the other containing the
    corresponding validation set splits.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.
        timestamp_col (str, opitonal): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.
        n_splits (int, optional): Number of train/validation splits to generate.
        Defaults to model_config.n_splits.
        train_size (float, optional): Percentage of training data per split.
        Defaults to model_config.train_size.

    Returns:
        tuple[list[pd.Timestamp], list[pd.Timestamp]]: Train set and validation set splits.
    """
    try:
        unique_timestamps: list[pd.Timestamp] = sorted(data[timestamp_col].unique())
        split_size: int = len(unique_timestamps) // n_splits
        indices: list[int] = [
            min(split_size*idx, len(unique_timestamps) - 1) for idx in range(1, n_splits + 1)
        ]
        train_splits: list[pd.Timestamp] = [
            unique_timestamps[int(train_size*idx)] for idx in indices
        ]
        val_splits: list[pd.Timestamp] = [unique_timestamps[idx] for idx in indices]
        return train_splits, val_splits
    except Exception as e:
        raise e


def plot_time_series_splits(
    data: pd.DataFrame,
    company_id: str,
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column
) -> None:
    """Plots the time series splits, i.e., k-fold walk-forward validation, for a given company
    ID's hourly energy demand.

    Args:
        data (pd.DataFrame): DataFrame containing a 1-D time series of hourly energy demand.
        company_id (str): ID of the company that the time-series splits will be plotted for.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        temporal_column (str, opitonal): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.
    """
    try:
        data = data.query(f"company_id == '{company_id}'")
        train_splits, val_splits = data.pipe(get_time_series_splits)
        k: int = len(train_splits)

        # plot the company ID's k-fold walk-forward validation
        _, ax = plt.subplots(k, figsize=(20, 14), sharex=True)
        for fold, (train_split, val_split) in enumerate(zip(train_splits, val_splits)):
            y_label: str = f"{company_id.title()} Energy Demand (MW)"
            if fold == 0:
                query: str = f"{timestamp_col} <= '{train_split}'"
                train: pd.Series = data.set_index(timestamp_col).query(query)[target_col]
                train.plot(ax=ax[fold], style="-", label="Train Set", color="black")
                query = f"{timestamp_col} > '{train_split}' & {timestamp_col} <= '{val_split}'"
                val: pd.Series = data.set_index(timestamp_col).query(query)[target_col]
                val.plot(ax=ax[fold], style="--", label="Validation Set", color="black")
                ax[fold].axvline(val.index.min(), color="red", lw=3, ls="--")
                ax[fold].set_title(f"Fold {fold+1}")
                ax[fold].set_ylabel(y_label)
                ax[fold].grid(which="both", alpha=0.3)
                ax[fold].legend(loc="best", frameon=True)
            else:
                query: str = f"{timestamp_col} <= '{train_split}'"
                train: pd.Series = data.set_index(timestamp_col).query(query)[target_col]
                train.plot(ax=ax[fold], style="-", label="Train Set", color="black")
                query = f"{timestamp_col} > '{train_split}' & {timestamp_col} <= '{val_split}'"
                val: pd.Series = data.set_index(timestamp_col).query(query)[target_col]
                val.plot(ax=ax[fold], style="--", label="Validation Set", color="black")
                ax[fold].axvline(val.index.min(), color="red", lw=3, ls="--")
                ax[fold].set_title(f"Fold {fold+1}")
                ax[fold].set_xlabel("Timestamp (UTC)")
                ax[fold].set_ylabel(y_label)
                ax[fold].grid(which="both", alpha=0.3)
        plt.tight_layout()
    except Exception as e:
        raise e


def train_and_validate_model(
    data: pd.DataFrame,
    model: CatBoostRegressor | LGBMRegressor | XGBRegressor,
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column
) -> tuple[CatBoostRegressor | LGBMRegressor | XGBRegressor, float]:
    """Trains an and returns an ML model along with its average validation RMSE.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.
        model (CatBoostRegressor | LGBMRegressor | XGBRegressor): Pre-trained ML model.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.

    Returns:
        tuple[CatBoostRegressor | LGBMRegressor | XGBRegressor, float]: Trained ML model
        and its average validation RMSE.
    """
    try:
        # a list containing the names of the features
        features: list[str] = [col for col in data.columns if col not in db_config.table.columns]

        # get the train and validation set splits
        train_splits, val_splits = data.pipe(get_time_series_splits)

        # an empty list to store the model's validation set metrics, one per split
        val_metrics: list[float] = []
        for train_split, val_split in zip(train_splits, val_splits):
            query: str = f"{timestamp_col} <= '{train_split}'"
            x_train: pd.DataFrame = data.query(query)[features]
            y_train: pd.Series = data.query(query)[target_col]
            query = f"{timestamp_col} > '{train_split}' & {timestamp_col} <= '{val_split}'"
            x_val: pd.DataFrame = data.query(query)[features]
            y_val: pd.Series = data.query(query)[target_col]

            # if the model is an object of type, 'CatBoostRegressor'
            if isinstance(model, CatBoostRegressor):
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
                metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

            # if the model is an object of type, 'LGBMRegressor'
            elif isinstance(model, LGBMRegressor):
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
                metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

            # if the model is an object of type, 'XGBRegressor'
            else:
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

            # append the validation set metric to the 'val_metrics' list
            val_metrics.append(metric)
        return model, np.mean(val_metrics)
    except Exception as e:
        raise e


def save_model(model: CatBoostRegressor | LGBMRegressor | XGBRegressor) -> None:
    """Saves model to Paths.MODEL.

    Args:
        model (CatBoostRegressor | LGBMRegressor | XGBRegressor): ML model.
    """
    try:
        artifacts_dir: PosixPath = Paths.ARTIFACTS_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(Paths.MODEL, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise e
