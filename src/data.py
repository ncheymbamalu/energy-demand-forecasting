"""This module contains functionality for loading, pre-processing, validating, transforming, and
splitting data.
"""

from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path, PosixPath

import polars as pl

from omegaconf import DictConfig
from tqdm import tqdm

from src.config import Paths, load_params
from src.logger import logger
from src.utils import encode_hour, get_current_time

params: DictConfig = load_params(Path(__file__).stem)


def load_raw_data(path: PosixPath | str = Paths.RAW_DATA) -> pl.DataFrame:
    """Loads raw data from path.

    Args:
        path (PosixPath | str, optional): Raw data file path. Defaults to Paths.RAW_DATA

    Returns:
        pl.DataFrame: Raw data.
    """
    try:
        logger.info(f"Loading raw data from ./{path.relative_to(Paths.PROJECT_DIR)}.")
        return pl.read_parquet(path)
    except Exception as e:
        raise e


def preprocess_data(data: pl.DataFrame) -> pl.DataFrame:
    """Pre-processes raw data, that is, manipulates raw data to synthesize
    real time data.

    Args:
        data (pl.DataFrame): Raw data.

    Returns:
        pl.DataFrame: Pre-processed data.
    """
    try:
        temporal_col: str = params.temporal_column
        offset_year: int = params.offset_year
        current: datetime = get_current_time()
        start: datetime = current - timedelta(days=params.lookback)
        logger.info(
            f"Pre-processing the raw data between \
<green>{start.strftime('%Y-%m-%d %H:%M:%S')}</green> and \
<green>{current.strftime('%Y-%m-%d %H:%M:%S')}</green> inclusive."
        )
        data = (
            data
            .join(
                other=(
                    data
                    .group_by("company_id", maintain_order=True)
                    .agg(pl.col(temporal_col).min().alias("min_timestamp"))
                    .select(
                        "company_id",
                        (
                            (pl.datetime(offset_year, 1, 1) - pl.col("min_timestamp"))
                            .dt.total_hours()
                            .alias("n_hours")
                        )
                    )
                ),
                on="company_id",
                how="left",
                maintain_order="left"
            )
            .with_columns(
                (pl.col(temporal_col) + pl.duration(hours=pl.col("n_hours"))).alias(temporal_col)
            )
            .filter(pl.col(temporal_col).is_between(start, current))
            .sort(by=["company_id", temporal_col])
            .select(params.columns)
        )
        dfs: list[pl.DataFrame] = [
            (
                data
                .filter(pl.col("company_id").eq(pl.lit(company_id)))
                .upsample(
                    time_column=temporal_col,
                    every="1h",
                    maintain_order=True
                )
                .fill_null(strategy="forward")
                .select(data.columns)
            )
            for company_id in data["company_id"].unique()
        ]
        return (
            pl.concat(dfs, how="vertical")
            .unique(subset=["company_id", temporal_col], keep="first")
            .sort(by=["company_id", temporal_col])
        )
    except Exception as e:
        raise e


def validate_data(data: pl.DataFrame) -> None:
    """Validates pre-processed data, that is, checks for schema conformance,
    duplicates, and null values.

    Args:
        data (pl.DataFrame): Pre-processed data.

    Returns:
        pl.DataFrame: Pre-processed and validated data.
    """
    try:
        messages: dict[str, str] = {
            "schema": "Invalid schema!",
            "duplicates": "There are duplicate records!",
            "nulls": "There are null values!"
        }
        if Paths.PROCESSED_DATA.exists():
            assert data.schema == pl.scan_parquet(Paths.PROCESSED_DATA).schema, messages["schema"]
        assert data.is_duplicated().sum() == 0, messages.get("duplicates")
        assert data.null_count().sum_horizontal()[0] == 0, messages.get("nulls")
        logger.info("The pre-processed data has been validated!")
    except Exception as e:
        raise e


def load_preprocessed_data(path: PosixPath | str = Paths.PROCESSED_DATA) -> pl.DataFrame:
    """Loads the latest pre-processed data from path.

    Args:
        path (PosixPath | str, optional): Pre-processed data file path.
        Defaults to Paths.PROCESSED_DATA

    Returns:
        pl.DataFrame: Latest pre-processed data.
    """
    try:
        temporal_col: str = params.temporal_column
        lookback: int = params.lookback
        data: pl.DataFrame = (
            pl.read_parquet(path)
            .join(
                other=(
                    pl.read_parquet(path)
                    .group_by("company_id", maintain_order=True)
                    .agg(pl.col(temporal_col).max())
                    .select(
                        "company_id",
                        (pl.col(temporal_col) - pl.duration(days=lookback))
                        .alias(f"earliest_{temporal_col}")
                    )
                ),
                how="left",
                on="company_id",
                maintain_order="left"
            )
            .filter(pl.col(temporal_col).ge(pl.col(f"earliest_{temporal_col}")))
            .drop(f"earliest_{temporal_col}")
        )
        logger.info(
            f"Loading the pre-processed data between \
<green>{str(data[temporal_col].min())}</green> and \
<green>{str(data[temporal_col].max())}</green>, inclusive."
        )
        return data
    except Exception as e:
        raise e


def transform_data(data: pl.DataFrame) -> pl.DataFrame:
    """Transforms the pre-processed data into ML-ready data, that is,
    features consisting of lag features, window features (average lags),
    and datetime features, and their corresponding labels.

    Args:
        data (pl.DataFrame): Pre-processed data.

    Returns:
        pl.DataFrame: ML-ready data.
    """
    try:
        logger.info("Transforming the pre-processed data into ML-ready data.")
        # NOTE: change 'data_params' to 'params' once this function is moved to ./src/data.py
        target: str = params.target_column
        temporal_col: str = params.temporal_column
        step = start = params.step
        max_lag: int = params.max_lag

        # an empty list to store the transformed pl.DataFrames, one per company ID
        transformed_dfs: list[pl.DataFrame] = []
        for company_id in sorted(data["company_id"].unique()):
            # an empty dictionary to store the features
            features: dict[str, pl.DataFrame] = {}

            # create the lag features
            dfs: list[pl.DataFrame] = [
                data
                .filter(pl.col("company_id").eq(pl.lit(company_id)))
                .select(temporal_col)
                .with_row_index()
            ]
            dfs += [
                (
                    data
                    .filter(pl.col("company_id").eq(pl.lit(company_id)))
                    .select(pl.col(target).alias(f"lag_{lag}"))
                    .shift(n=lag)
                    .with_row_index()
                )
                for lag in reversed(range(1, max_lag + 1))
            ]
            features["lag"] = (
                reduce(lambda left, right: left.join(other=right, on="index", how="inner"), dfs)
                .sort(by="index")
                .drop("index")
                .drop_nulls()
            )

            # create the window features, i.e., average lags
            dfs = [
                (
                    features.get("lag")
                    .drop(temporal_col)[:, -lag:]
                    .mean_horizontal()
                    .to_frame(name=f"avg_{lag}_lags")
                )
                for lag in reversed(range(start, max_lag + 1, step))
            ]
            features["window"] = pl.concat(dfs, how="horizontal")

            # create the datetime features
            features["datetime"] = (
                features.get("lag")
                .select(
                    pl.col(temporal_col)
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone("EST")
                    .dt.hour()
                    .alias("hour")
                )
                .with_columns(
                    (
                        pl.col("hour").map_elements(
                            lambda hour: encode_hour(hour, "sine"),
                            return_dtype=pl.Float64
                        )
                        .alias("sine_hour")
                    ),
                    (
                        pl.col("hour").map_elements(
                            lambda hour: encode_hour(hour, "cosine"),
                            return_dtype=pl.Float64
                        )
                        .alias("cosine_hour")
                    )
                )
                .sql(
                    """\
                    SELECT
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

            # create the machine learning-ready dataset and append to the 'transformed_dfs' list
            transformed_data: pl.DataFrame = (
                pl.concat(
                    (
                        (
                            features.get("lag")
                            .select(
                                pl.lit(company_id).alias("company_id"),
                                temporal_col
                            )
                        ),
                        features.get("datetime"),
                        features.get("window"),
                        features.get("lag").drop(temporal_col)
                    ),
                    how="horizontal"
                )
                .join(
                    other=data,
                    on=["company_id", temporal_col],
                    how="left",
                    maintain_order="left"
                )
            )
            transformed_dfs.append(transformed_data)
        return pl.concat(transformed_dfs, how="vertical").sort(by=["company_id", temporal_col])
    except Exception as e:
        raise e


def split_data(
    data: pl.DataFrame,
    test_size: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits data into train and test sets.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.
        test_size (int): Number of records for each company ID's test set.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Train and test sets.
    """
    try:
        logger.info("Splitting the ML-ready data into train and test sets.")
        temporal_col: str = params.temporal_column

        # a dictionary that maps each company ID to its temporal split
        splits: dict[str, datetime] = {
            company_id: (
                data
                .filter(pl.col("company_id").eq(pl.lit(company_id)))
                .select(temporal_col)
                .to_series()
                .max()
                - timedelta(hours=test_size)
            )
            for company_id in sorted(data["company_id"].unique())
        }

        # two empty lists to store the train and test sets for each company ID
        train_dfs, test_dfs = [], []
        for company_id, split in tqdm(splits.items(), unit="Company ID"):
            train_dfs.append(
                data
                .filter(
                    pl.col("company_id").eq(pl.lit(company_id))
                    & pl.col(temporal_col).le(pl.lit(split))
                )
            )
            test_dfs.append(
                data
                .filter(
                    pl.col("company_id").eq(pl.lit(company_id))
                    & pl.col(temporal_col).gt(pl.lit(split))
                )
            )
        return pl.concat(train_dfs, how="vertical"), pl.concat(test_dfs, how="vertical")
    except Exception as e:
        raise e
