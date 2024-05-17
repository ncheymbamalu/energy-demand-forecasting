"""Data ingestion and pre-processing"""

import os
import random

from datetime import timedelta

import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import Connection, create_engine, text
from statsmodels.tsa.stattools import adfuller

from src.logger import logging

load_dotenv()

HOST: str | None = os.getenv("HOST")
DB: str | None = os.getenv("DB")
USER: str | None = os.getenv("USER")
PWD: str | None = os.getenv("PASSWORD")
PORT: str | None = os.getenv("PORT")
PG_CONN: Connection = create_engine(f"postgresql://{USER}:{PWD}@{HOST}:{PORT}/{DB}").connect()


def query_ids(schema: str = "time_series", table: str = "energy_demand") -> list[str]:
    """Queries a table from Postgres and returns a list of IDs

    Args:
        schema (str, optional): Postgres schema. Defaults to "time_series".
        table (str, optional): Postgres table. Defaults to "energy_demand".

    Returns:
        list[str]: IDs, which will be used to query a subset of raw hourly
        energy demand data
    """
    try:
        query: str = f"""
        SELECT
            DISTINCT "id"
        FROM
            {schema}.{table};
        """
        ids: list[str] = pd.DataFrame(PG_CONN.execute(text(query)))["id"].tolist()
        return ids
    except Exception as e:
        raise e


def query_data(
    query_id: str, schema: str = "time_series", table: str = "energy_demand"
) -> pd.DataFrame:
    """Queries a table from Postgres and returns a pd.DataFrame

    Args:
        query_id: (str): ID that's used to query a subset of raw data from Postgres
        schema (str, optional): Postgres schema. Defaults to "time_series".
        table (str, optional): Postgres table. Defaults to "energy_demand".

    Returns:
        pd.DataFrame: Energy demand time series, in hourly increments
    """
    try:
        query: str = f"""
        SELECT
            *
        FROM
            {schema}.{table}
        WHERE
            "id" = '{query_id}';
        """
        logging.info("Querying raw data from %s.%s", schema, table)
        raw_data: pd.DataFrame = (
            pd.DataFrame(PG_CONN.execute(text(query)))
            .sort_values("datetime")
            .drop_duplicates(subset="datetime", keep="first")
            .rename({"values": "energy_demand"}, axis=1)
            .drop("id", axis=1)
            .reset_index(drop=True)
        )
        return raw_data
    except Exception as e:
        raise e


def get_timestamps(raw_data: pd.DataFrame, duration: int = 10) -> tuple[str, str]:
    """Returns the starting and ending timestamps, which will be used to filter 'raw_data'

    Args:
        raw_data (pd.DataFrame): Energy demand time series, in hourly increments
        duration (int): The amount of in-sample data to train, measured in days. Defaults to 10

    Returns:
        tuple[str, str]: Starting and ending timestamps
    """
    try:
        min_ts: str = raw_data.iloc[0]["datetime"]
        max_ts: str = str(
            pd.to_datetime(raw_data.iloc[-1]["datetime"]) - timedelta(days=duration + 2)
        )
        timestamps: list[str] = [str(ts) for ts in pd.date_range(min_ts, max_ts, freq="H")]
        start: str = random.choice(timestamps)
        end: str = str(pd.to_datetime(start) + timedelta(days=duration))
        return start, end
    except Exception as e:
        raise e


def ensure_stationarity(
    data: pd.DataFrame, target_name: str = "energy_demand"
) -> tuple[pd.DataFrame, str]:
    """Returns a DataFrame that contains a stationary univariate time series
    and its column name

    Args:
        data (pd.DataFrame): DataFrame that contains a univariate time series
        target_name (str): Column name of the univariate time series

    Returns:
        tuple[pd.DataFrame, str]: DataFrame that contains a stationary univariate
        time series and its column name
    """
    try:
        p_value: float = adfuller(data[target_name].dropna())[1]
        df: pd.DataFrame = (
            data.dropna()
            .assign(
                prev=data[target_name].dropna().shift(periods=1),
                diff=data[target_name].dropna().diff(periods=1),
            )
            .copy(deep=True)
        )
        return (data, target_name) if p_value < 0.05 else (df, "diff")
    except Exception as e:
        raise e


def ingest_data(query_id: str) -> tuple[pd.DataFrame, str]:
    """Ingests, pre-processe, and returns a stationary univariate time series
    that's sourced from raw hourly energy demand data

    Args:
        query_id: (str): ID that's used to query a subset of raw data from Postgres

    Returns:
        tuple[pd.DataFrame, str]: pd.DataFrame containing a stationary univariate
        time series and its column name
    """
    try:
        raw_data: pd.DataFrame = query_data(query_id)
        initial_timestamp, final_timestamp = get_timestamps(raw_data)
        processed_data, target_name = (
            raw_data.assign(datetime=pd.to_datetime(raw_data["datetime"]))
            .loc[
                (raw_data["datetime"] > initial_timestamp)
                & (raw_data["datetime"] <= final_timestamp)
            ]
            .set_index("datetime")
            .asfreq(freq="H")
            .rename_axis(None)
            .pipe(ensure_stationarity)
        )
        logging.info("Data ingestion and pre-processing complete.")
        return processed_data, target_name
    except Exception as e:
        raise e


if __name__ == "__main__":
    df_stationary, target = ingest_data(random.choice(query_ids()))
    print(df_stationary.head())
