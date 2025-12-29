"""This module contains helper functions that are used in other modules."""

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from omegaconf import ListConfig
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import pacf

from src.logger import logger


#---------------------------------------------------------------------------------------------------
# data utility functions
#---------------------------------------------------------------------------------------------------
def get_current_time() -> datetime:
    """Returns the current time in UTC, truncated at the hour.

    Returns:
        datetime: Current time in UTC, truncated at the hour.
    """
    try:
        return (
            datetime
            .now(timezone.utc)
            .replace(
                minute=0,
                second=0,
                microsecond=0,
                tzinfo=None
            )
        )
    except Exception as e:
        raise e


def get_max_lag(
    data: pl.DataFrame,
    target: str,
    step: int
) -> int:
    """Returns the maximum number of lags to use for creating lag and window features.

    Args:
        data (pl.DataFrame): Pre-processed data.
        target (str): Name of the target variable.
        step (int): Step size.

    Returns:
        int: Maximum number of lags to use for creating lag features and window features.
    """
    try:
        max_lags: list[int] = []
        for company_id in sorted(data["company_id"].unique()):
            lag_correlations: np.ndarray = pacf(
                x=data.filter(pl.col("company_id").eq(pl.lit(company_id)))[target],
                nlags=30,
                method="ywmle"
            )
            max_lag: int = np.where(np.abs(lag_correlations) > 0.1)[0][-1]
            max_lags.append(max_lag)
        max_lag = int(round(max(max_lags) / step)) * step
        return max_lag
    except Exception as e:
        raise e


def encode_hour(hour: int, func: str) -> float:
    """Encodes the input hour, that is, converts it to an angle in radians.

    Args:
        hour (int): Input hour.
        func (str): Function used for the conversion, that is, sine or cosine.

    Returns:
        float: Sine or cosine of an angle, in radians.
    """
    try:
        assert func.lower() in ("sine", "cosine"), "Invalid argument for the 'func' parameter. \
Valid arguments are 'sine' and 'cosine'."
        period: int = 24
        if func == "sine":
            encoded_hour: np.float64 = np.sin(hour / period * 2 * np.pi)
        else:
            encoded_hour = np.cos(hour / period * 2 * np.pi)
        return round(encoded_hour.item(), 2)
    except Exception as e:
        raise e


def select_relevant_features(
    data: pl.DataFrame,
    features: list[str] | ListConfig,
    target: str,
    threshold: float
) -> list[str]:
    """Selects the most relevant features based on their mutual information score
    with the target. 

    Args:
        data (pl.DataFrame): ML-ready data consisting of lag features, window
        features (average lags), datetime features, and the corresponding labels.
        features (list[str]): Names of the all the features.
        target (str): Name of the target variable.
        threshold (float): Number between 0 and 1, inclusive, that's used to
        filter out less relevant features.

    Returns:
        list[str]: Relevant features.
    """
    try:
        logger.info("Selecting the most relevant features...")
        scores: np.ndarray = mutual_info_regression(data.select(features), data[target])
        relevant_features: list[str] = (
            pl.DataFrame({
                "feature": features,
                "mutual_info": (scores - scores.min()) / (scores.max() - scores.min())
            })
            .filter(
                pl.col("mutual_info").ge(pl.col("mutual_info").quantile(threshold))
            )
            .sort(by="mutual_info", descending=True)
            .select("feature")
            .to_series()
            .to_list()
        )
        return [feature for feature in features if feature in relevant_features]
    except Exception as e:
        raise e


#---------------------------------------------------------------------------------------------------
# model utility functions
#---------------------------------------------------------------------------------------------------
def get_time_series_splits(
    data: pl.DataFrame,
    temporal_col: str,
    k: int
) -> tuple[list[datetime], ...]:
    """Returns two lists, both containing k datetime objects, where k is the
    number of folds. The datetime objects in the 1st list mark the end of each
    fold's train set, and the datetime objects in the 2nd list mark the end of
    each fold.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.
        temporal_col (str): Name of the column that contains the datetime objects.
        k (int): Number of train/validation folds to generate.

    Returns:
        tuple[list[datetime], list[datetime]]: Time series splits.
    """
    try:
        data = (
            data
            .select(temporal_col)
            .sort(by=temporal_col)
            .unique(maintain_order=True)
            .with_row_index()
        )
        step: int = len(data) // (k + 1)
        end_of_fold: list[int] = [data["index"].max() - (step * i) for i in reversed(range(k))]
        end_of_train_set: list[int] = [idx - (idx // (k + 1)) for idx in end_of_fold]
        return tuple(
            data.filter(pl.col("index").is_in(indices))[temporal_col].to_list()
            for indices in (end_of_train_set, end_of_fold)
        )
    except Exception as e:
        raise e


def plot_time_series_splits(
    data: pl.DataFrame,
    company_id: int,
    target: str,
    temporal_col: str,
    k: int
) -> None:
    """Plots the time series split, i.e., walk-forward validation, for the input
    comapny ID's hourly energy demand.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.
        company_id (str): Name of the company ID whose time series splits to plot.
        target (str): Name of the target variable.
        temporal_col (str): Name of the column that contains the datetime objects.
        k (int): Number of train/validation folds to generate.
    """
    try:
        data = data.filter(pl.col("company_id").eq(pl.lit(company_id)))
        train_splits, fold_splits = data.pipe(get_time_series_splits, temporal_col, k)

        # plot the company ID's time series split
        fig, ax = plt.subplots(k, figsize=(20, 14), sharex=True)
        fig.suptitle(f"Hourly Energy Demand, Company ID: {company_id}", fontsize=16)
        fig.supxlabel("Datetime (UTC)", fontsize=12)
        for fold, (train_split, fold_split) in enumerate(zip(train_splits, fold_splits)):
            (
                data
                .filter(pl.col(temporal_col).le(pl.lit(train_split)))
                .select(temporal_col, target)
                .to_pandas()
                .set_index(temporal_col)
                [target]
                .plot(ax=ax[fold], style="-", label="Train Set", color="black")
            )
            (
                data
                .filter(pl.col(temporal_col).is_between(train_split, fold_split, closed="right"))
                .select(temporal_col, target)
                .to_pandas()
                .set_index(temporal_col)
                [target]
                .plot(ax=ax[fold], style="--", label="Validation Set", color="black")
            )
            ax[fold].axvline(train_split, color="red", lw=3, ls="--")
            ax[fold].set_title(f"Fold {fold + 1}")
            ax[fold].set_xlabel(None)
            ax[fold].set_ylabel("Energy Demand (MW)")
            ax[fold].grid(which="both", alpha=0.3)
            ax[fold].legend(loc="best", frameon=True)
        plt.tight_layout()
    except Exception as e:
        raise e


def compute_evaluation_metrics(
    y: np.ndarray | pl.Series,
    yhat: np.ndarray | pl.Series
) -> dict[str, float]:
    """Computes the coefficient of determination, R², and root mean squared error,
    RMSE, between y and yhat.

    Args:
        y (np.ndarray | pl.Series): Labels.
        yhat (np.ndarray | pl.Series): Predictions.

    Returns:
        dict[str, float]: R² and RMSE.
    """
    try:
        t: np.ndarray | pl.Series = y - y.mean()
        sst: float = t.dot(t)
        e: np.ndarray | pl.Series = y - yhat
        sse: float = e.dot(e)
        r2: float = 1 - (sse / sst)
        rmse: np.float64 = np.sqrt(sse / len(y))
        return {"r2": round(r2, 4), "rmse": round(rmse.item(), 4)}
    except Exception as e:
        raise e
