"""This module contains helper functions that are used in other modules."""

from datetime import datetime, timezone

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
