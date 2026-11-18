"""Data pipeline for forecasting daily PJM subregion energy demand.

Ingestion, pre-processing, and validation: fetches hourly energy demand for each PJM subregion
from the EIA Open Data API (v2), converts it to hour-beginning UTC timestamps, fills gaps with
imputed values flagged by `is_imputed`, and validates the schema, uniqueness, and absence of
nulls before it's upserted into the processed data file.

Transformation: aggregates the pre-processed hourly data into average hourly demand per
Eastern Time day, then creates lag features, trailing window averages, and calendar features
for modeling.

Splitting: splits the daily data into training and validation sets by date, either by a
proportion of dates for model selection or by a number of most recent days for retraining.
"""

import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

import polars as pl
import requests
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from polars.datatypes.classes import DataTypeClass
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from energy_demand_forecasting.config import Config
from energy_demand_forecasting.logger import logger
from energy_demand_forecasting.utils import get_current_hour, get_us_holidays


load_dotenv(Config.Paths.ENV)

PARAMS: DictConfig = Config.load_params(Path(__file__).stem)

pl.Config.set_engine_affinity(engine="streaming")


#---------------------------------------------------------------------------------------------------
# Ingestion // Pre-processing // Validation
#---------------------------------------------------------------------------------------------------
def get_subregion_query_parameters() -> list[dict[str, str]]:
    """Builds the per-subregion `start` and `end` query parameters for fetching data from the
    EIA API.

    The subregions to fetch are those in `PARAMS.eia.subregion_ids`. Each subregion with data in
    the processed data file gets a `start` of its latest stored `timestamp_utc` minus 8 days, so
    recent hours are re-fetched to pick up EIA revisions and fill any gaps. Every other
    subregion, including all of them when the file doesn't exist or is empty, gets a `start` 730
    days before the current hour. In all cases, `end` is the current UTC hour.

    Returns:
        list[dict[str, str]]: One record per subregion, in the order of
            `PARAMS.eia.subregion_ids`, with keys `subregion_id` (uppercase), `start`, and `end`;
            timestamps are formatted as `%Y-%m-%dT%H`.
    """
    end: datetime = get_current_hour()
    backfill_start: datetime = end - timedelta(days=730)
    path: Path = Config.Paths.PROCESSED_DATA
    subregion_ids = cast(list[str], OmegaConf.to_object(PARAMS.eia.subregion_ids))

    data: pl.DataFrame = pl.DataFrame({"subregion_id": subregion_ids})
    if path.is_file():
        data = (
            data
            .join(
                other=(
                    pl.scan_parquet(path)
                    .group_by(pl.col("subregion_id").str.to_uppercase())
                    .agg((pl.max("timestamp_utc") - pl.duration(days=8)).alias("start"))
                    .collect()
                ),
                on="subregion_id",
                how="left",
            )
        )
    else:
        data = data.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("start"))

    return (
        data
        .with_columns(
            pl.col("start").fill_null(backfill_start).dt.to_string("%Y-%m-%dT%H"),
            pl.lit(end).dt.to_string("%Y-%m-%dT%H").alias("end"),
        )
        .to_dicts()
    )


def fetch_data() -> pl.DataFrame:
    """Fetches hourly electricity demand for PJM zones from the EIA Open Data API (v2).

    Returns:
        pl.DataFrame: Raw hourly data.
    """
    url: str = urljoin(PARAMS.eia.base_url, PARAMS.eia.endpoints.subregion_demand)
    base_params: dict[str, str] = {
        "api_key": os.environ["EIA_API_KEY"],
        "frequency": "hourly",
        "data[0]": "value",
        "facets[parent][]": "PJM",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": "5000",
    }

    schema = cast(
        dict[str, pl.DataType],
        {
            "period": pl.String,
            "subba": pl.String,
            "subba-name": pl.String,
            "parent": pl.String,
            "parent-name": pl.String,
            "value": pl.String,
            "value-units": pl.String,
        }
    )

    retry: Retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))

        logger.info("Starting the data ingestion process...")
        records: list[dict[str, str]] = get_subregion_query_parameters()
        dfs: list[pl.DataFrame] = []
        for record in records:
            subregion_id: str = record["subregion_id"]
            logger.info("Fetching data for sub-region ID <green>{}</>", subregion_id)
            subregion_params: dict[str, str] = {
                "facets[subba][]": subregion_id,
                "start": record["start"],
                "end": record["end"],
            }
            offset: int = 0
            while True:
                response: Response = session.get(
                    url,
                    params=base_params | subregion_params | {"offset": offset},
                    timeout=(10, 60),
                )
                response.raise_for_status()

                payload: Any = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Unexpected response for sub-region ID {subregion_id} at "
                        f"offset {offset}:\npayload type={type(payload).__name__}"
                    )

                response_body: Any = payload.get("response")
                if not isinstance(response_body, dict) or "data" not in response_body:
                    raise ValueError(
                        f"Unexpected response for sub-region ID {subregion_id} at "
                        f"offset {offset}:\npayload keys={list(payload)}\n"
                        f"payload error={payload.get('error')!r}"
                    )

                rows: list[dict[str, str]] = response_body["data"]
                offset += len(rows)
                total_rows: int = int(response_body.get("total") or 0)
                if not rows:
                    break
                dfs.append(pl.DataFrame(rows, schema=schema))
                if len(rows) < int(base_params["length"]):
                    break

            if offset != total_rows:
                raise RuntimeError(
                    f"Incomplete data fetch for sub-region ID {subregion_id}: "
                    f"expected {total_rows} rows, fetched {offset}."
                )
            time.sleep(0.5)

    return pl.DataFrame(schema=schema) if not dfs else pl.concat(dfs)


def preprocess_data(data: pl.DataFrame) -> pl.DataFrame:
    """Pre-processes raw hourly electricity demand data.

    Args:
        data (pl.DataFrame): Raw hourly data.

    Returns:
        pl.DataFrame: Pre-processed hourly data.
    """
    if data.is_empty():
        logger.warning(
            "Data not fetched for any sub-region IDs; returning an empty Polars DataFrame."
        )
        schema: dict[str, DataTypeClass | pl.Datetime] = {
            "subregion_id": pl.String,
            "timestamp_utc": pl.Datetime(time_unit="us"),
            "energy_demand_mwh": pl.Float64,
            "is_imputed": pl.Boolean,
            "processed_at_utc": pl.Datetime(time_unit="us"),
        }
        return pl.DataFrame(schema=schema)

    subregion_ids = cast(list[str], OmegaConf.to_object(PARAMS.eia.subregion_ids))
    data = (
        data
        .rename({
            "subba": "subregion_id",
            "period": "timestamp_utc",
            "value": "energy_demand_mwh"
        })
        .select(
            pl.col("subregion_id").str.to_lowercase(),
            (
                pl.col("timestamp_utc").add(":00:00").str.to_datetime("%Y-%m-%dT%H:%M:%S")
                - pl.duration(hours=1)
            ),
            pl.col("energy_demand_mwh").cast(pl.Float64)
        )
        .unique(subset=["subregion_id", "timestamp_utc"], keep="last")
        .sort("subregion_id", "timestamp_utc")
        .upsample(
            time_column="timestamp_utc",
            every="1h",
            group_by="subregion_id",
            maintain_order=True
        )
        .with_columns(
            pl.col("energy_demand_mwh").is_null().alias("is_imputed"),
            (
                pl.coalesce(
                    pl.col("energy_demand_mwh").forward_fill(limit=1),
                    pl.col("energy_demand_mwh").shift(24),
                    pl.col("energy_demand_mwh").shift(168),
                    pl.col("energy_demand_mwh").forward_fill()
                )
                .over(partition_by="subregion_id")
                .alias("energy_demand_mwh")
            ),
            (
                pl.lit(datetime.now(UTC))
                .dt.replace_time_zone(None)
                .dt.truncate("1s")
                .alias("processed_at_utc")
            )
        )
    )
    missing_ids: list[str] = sorted(
        {v.lower() for v in subregion_ids} - set(data.get_column("subregion_id").unique())
    )
    if missing_ids:
        logger.warning(
            "Data not fetched for the following sub-region IDs: {}", ", ".join(missing_ids)
        )
    return data


def validate_data(data: pl.DataFrame) -> None:
    """Validates pre-processed hourly electricity demand data, that is, checks for schema
    conformance, duplicate records, and null values.

    Args:
        data (pl.DataFrame): Pre-processed hourly data.
    """
    schema: pl.Schema = pl.Schema({
        "subregion_id": pl.String,
        "timestamp_utc": pl.Datetime(time_unit="us"),
        "energy_demand_mwh": pl.Float64,
        "is_imputed": pl.Boolean,
        "processed_at_utc": pl.Datetime(time_unit="us"),
    })
    if data.schema != schema:
        raise ValueError(f"Invalid schema!\nexpected: {dict(schema)}\ngot: {dict(data.schema)}")

    if data.select("subregion_id", "timestamp_utc").is_duplicated().any():
        raise ValueError("Duplicate (subregion_id, timestamp_utc) records found!")

    null_counts: dict[str, int] = {
        col: count
        for col, count in data.null_count().row(0, named=True).items()
        if count > 0
    }
    if null_counts:
        raise ValueError(f"NULL values found: {null_counts}")
    logger.info("The pre-processed data has been validated!")


#---------------------------------------------------------------------------------------------------
# Transformation // Splitting
#---------------------------------------------------------------------------------------------------
def create_daily_data(data: pl.DataFrame) -> pl.DataFrame:
    """Aggregates hourly energy demand into daily averages for each subregion in Eastern Time.

    Naive UTC timestamps are converted to `America/New_York` so that every subregion shares
    the same day boundaries. Daily demand is the mean of the hourly values rather than the sum,
    so 23-hour and 25-hour DST days stay comparable to 24-hour days. Only complete days, i.e.,
    records spanning local hours 0 through 23, are kept, which drops partial days at the edges
    of the data.

    Args:
        data (pl.DataFrame): Pre-processed and validated hourly data with columns `subregion_id`,
            `timestamp_utc`, `is_imputed`, and `energy_demand_mwh`.

    Returns:
        pl.DataFrame: Data with columns `subregion_id`, `date`, `avg_hourly_demand_mwh`, and
            `n_imputed_records`, sorted by `subregion_id` and `date`.
    """
    return (
        data
        .with_columns(
            pl.col("timestamp_utc")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York")
            .dt.replace_time_zone(None)
            .alias("timestamp_et")
        )
        .group_by(
            "subregion_id",
            pl.col("timestamp_et").dt.date().alias("date")
        )
        .agg(
            pl.mean("energy_demand_mwh").alias("avg_hourly_demand_mwh"),
            pl.sum("is_imputed").cast(pl.Int8).alias("n_imputed_records"),
            pl.col("timestamp_et").dt.hour().min().alias("min_hour"),
            pl.col("timestamp_et").dt.hour().max().alias("max_hour")
        )
        # NOTE: this only keeps (subregion_id, date) pairs with a 'complete' day's worth of records
        .filter(
            pl.col("min_hour").eq(0)
            & pl.col("max_hour").eq(23)
        )
        .drop("min_hour", "max_hour")
        .sort("subregion_id", "date")
    )


def create_lag_features(data: pl.DataFrame, max_lag: int = 21) -> pl.DataFrame:
    """Creates lagged daily energy demand features for each subregion.

    Args:
        data (pl.DataFrame): Daily data with columns `subregion_id`, `date`, and
            `avg_hourly_demand_mwh`.
        max_lag (int, optional): Largest lag, in days, to create. Defaults to `21`.

    Returns:
        pl.DataFrame: Daily data with columns `subregion_id`, `date`, `lag_{max_lag}` through
            `lag_1`, and `avg_hourly_demand_mwh`, sorted by `subregion_id` and `date`.
    """
    lags: list[pl.Expr] = [
        (
            pl.col("avg_hourly_demand_mwh")
            .shift(n=lag)
            .over(partition_by="subregion_id")
            .alias(f"lag_{lag}")
        )
        for lag in reversed(range(1, max_lag + 1))
    ]
    return (
        data
        .sort("subregion_id", "date")
        .select("subregion_id", "date", *lags, "avg_hourly_demand_mwh")
        .drop_nulls(f"lag_{max_lag}")
    )


def create_window_features(data: pl.DataFrame, step: int = 7) -> pl.DataFrame:
    """Creates trailing window averages of daily energy demand from the lag features.

    For each multiple of `step` up to the number of lag columns, averages `lag_1` through
    `lag_{window}`. For example, with 21 lags and a `step` of 7, this creates `avg_21_lags`,
    `avg_14_lags`, and `avg_7_lags`, the average daily demand over the previous 21, 14, and 7
    days.

    Args:
        data (pl.DataFrame): Daily data with columns `subregion_id`, `date`, and lag features.
        step (int, optional): Increment, in days, between window sizes. Defaults to `7`.

    Returns:
        pl.DataFrame: Daily data with columns `subregion_id`, `date`, the window averages from
            largest to smallest window, and the remaining columns of `data`.
    """
    id_cols: list[str] = ["subregion_id", "date"]
    lag_cols: list[str] = sorted(
        (col for col in data.columns if col.startswith("lag_")),
        key=lambda col: int(col.removeprefix("lag_"))
    )
    windows: list[int] = sorted(range(step, len(lag_cols) + 1, step), reverse=True)
    window_avgs: list[pl.Expr] = [
        pl.mean_horizontal(lag_cols[:window]).alias(f"avg_{window}_lags")
        for window in windows
    ]
    return data.select(*id_cols, *window_avgs, pl.exclude(id_cols))


def create_datetime_features(data: pl.DataFrame) -> pl.DataFrame:
    """Creates calendar features from the Eastern Time `date` column.

    The features are `month` (1-12), `day_of_year` (1-366), `day_of_month` (1-31),
    `day_of_week` (1 = Monday through 7 = Sunday), `is_weekend` (True on Saturdays and
    Sundays), and `is_holiday` (True on US federal holidays, including observed dates).

    Args:
        data (pl.DataFrame): Daily data with columns `subregion_id`, `date`, lag features, and
            window averages.

    Returns:
        pl.DataFrame: Daily data with columns `subregion_id`, `date`, the calendar features, and
            the remaining columns of `data`.
    """
    id_cols: list[str] = ["subregion_id", "date"]
    us_holidays: list[date] = data.pipe(get_us_holidays)
    return (
        data
        .select(
            *id_cols,
            pl.col("date").dt.month().cast(pl.Int8).alias("month"),
            pl.col("date").dt.ordinal_day().cast(pl.Int16).alias("day_of_year"),
            pl.col("date").dt.day().cast(pl.Int8).alias("day_of_month"),
            pl.col("date").dt.weekday().alias("day_of_week"),
            pl.col("date").dt.weekday().is_in([6, 7]).alias("is_weekend"),
            pl.col("date").is_in(us_holidays).alias("is_holiday"),
            pl.exclude(id_cols)
        )
    )


def transform_data(data: pl.DataFrame) -> pl.DataFrame:
    """Transforms pre-processed and validated hourly data into daily features for modeling.

    Applies `create_daily_data`, `create_lag_features`, `create_window_features`, and
    `create_datetime_features`, in that order.

    Args:
        data (pl.DataFrame): Pre-processed and validated hourly data with columns `subregion_id`,
            `timestamp_utc`, `is_imputed`, and `energy_demand_mwh`.

    Returns:
        pl.DataFrame: Daily data with columns `subregion_id`, `date`, calendar features, window
            averages, lag features (`lag_21` through `lag_1`), and `avg_hourly_demand_mwh`, sorted
            by `subregion_id` and `date`.
    """
    return (
        data
        .pipe(create_daily_data)
        .pipe(create_lag_features)
        .pipe(create_window_features)
        .pipe(create_datetime_features)
    )


def split_data(
    data: pl.DataFrame, train_size: float = 0.75, val_days: int | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits daily data into training and validation sets by date.

    Every subregion is split on the same date, and every validation date comes after every
    training date. Splitting by date rather than by row prevents information from the
    validation period leaking into training. If `val_days` is set, the last `val_days` unique
    dates form the validation set, which suits recurring retraining. Otherwise, the split date
    is the unique date at the `train_size` proportion of all unique dates, which suits model
    selection.

    Args:
        data (pl.DataFrame): Daily data with columns `subregion_id` and `date`.
        train_size (float, optional): Proportion of unique dates to use for training; ignored
            if `val_days` is set. Defaults to `0.75`.
        val_days (int | None, optional): Number of most recent unique dates to use for
            validation. Defaults to `None`.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Training data (dates before the split date) and
            validation data (dates on or after the split date).
    """
    dates: list[date] = data.get_column("date").unique().sort().to_list()
    if val_days is not None:
        if not 0 < val_days < len(dates):
            raise ValueError(
                f"Invalid `val_days`: {val_days}. Must be between 1 and {len(dates) - 1}."
            )
        split_idx: int = len(dates) - val_days
    else:
        if not 0 < train_size < 1:
            raise ValueError(f"Invalid `train_size`: {train_size}. Must be between 0 and 1.")
        split_idx = int(len(dates) * train_size)
        if split_idx == 0:
            raise ValueError(f"`train_size` of {train_size} leaves no training dates.")

    split_date: date = dates[split_idx]
    logger.info("Splitting data at {}: training < {} <= validation", split_date, split_date)
    return (
        data.filter(pl.col("date").lt(split_date)),
        data.filter(pl.col("date").ge(split_date)),
    )
