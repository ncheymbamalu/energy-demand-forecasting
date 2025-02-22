"""This module provides functionality for interacting with PostgeSQL's 'postgres' database."""

import glob
import os

from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from omegaconf import ListConfig
from sqlalchemy import URL, Connection, create_engine, text

from src.config import Paths, db_config
from src.logger import logger

load_dotenv(Paths.ENV)


def get_db_connection() -> Connection:
    """Returns an object that connects to the 'postgres' database.

    Returns:
        Connection: Object that points to and can be used to interact with
        the 'postgres' database.
    """
    try:
        url: URL = URL.create(
            drivername=db_config.drivername,
            username=db_config.user,
            host=db_config.host,
            database=db_config.dbname,
            port=db_config.port,
            password=os.getenv("PG_PASSWORD")
        )
        db_connection: Connection = create_engine(url).connect()
        return db_connection
    except Exception as e:
        raise e


def create_schema(schema: str = db_config.schema) -> None:
    """Creates a schema named, 'time_series', under the 'postgres' database.

    Args:
        schema (str, optional): Db schema. Defaults to db_config.schema, that is, 'time_series'.
    """
    try:
        db_connection: Connection = get_db_connection()
        db_connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        db_connection.commit()
        db_connection.close()
    except Exception as e:
        raise e


@logger.catch
def create_table(schema: str = db_config.schema, table: str = db_config.table.name) -> None:
    """Creates a table named, 'energy_demand', which exists under the 'postgres' database's
    'time_series' schema, and populates it with raw data from ~/data/*.parquet.

    Args:
        schema (str, optional): Database schema. Defaults to db_config.schema.
        table (str, optional): Database table. Defaults to db_config.table.name.
    """
    try:
        # create the table
        db_connection: Connection = get_db_connection()
        db_connection.execute(text(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table}
            (
                company_id TEXT NOT NULL,
                timestamp_utc TIMESTAMP,
                energy_demand_mw REAL
            )
            """
        ))
        db_connection.commit()

        # populate the table with raw data from ~/data/*.parquet
        cols: ListConfig = db_config.table.columns
        dfs: list[pd.DataFrame] = [
            (
                pd.read_parquet(path)
                .apply(lambda col: pd.to_datetime(col) if col.name == "timestamp_utc" else col)
                .assign(company_id=Path(path).stem.split("_")[0].lower())
                .sort_values(by="timestamp_utc")
                .reset_index(drop=True)
                [cols]
            ) for path in sorted(glob.glob(os.path.join(Paths.DATA_DIR, "*.parquet")))
        ]
        (
            pd.concat(dfs, axis=0, ignore_index=True)
            .to_sql(
                name=table,
                schema=schema,
                con=db_connection,
                if_exists="append",
                index=False
            )
        )
        db_connection.close()
        logger.info(
            f"SUCCESS: The '{db_config.dbname}' database's '{schema}.{table}' table has been \
created and populated with raw data."
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    create_schema()
    create_table()
