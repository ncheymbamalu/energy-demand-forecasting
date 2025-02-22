"""This module provides functionality to ingest, pre-process, validate, and transform raw data."""

import pandas as pd

from omegaconf import ListConfig

from src.config import data_config, db_config
from src.logger import logger
from src.utils import create_features, query_data


# -------------------------------------------------------------------------------------------------
# Data ingestion, pre-processing, and validation
# -------------------------------------------------------------------------------------------------
@logger.catch
def fetch_data(
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column,
    offset_timestamp: str = data_config.offset_timestamp,
    duration: int = 14
) -> pd.DataFrame:
    """Fetches raw hourly energy demand data from a PostgreSQL database, pre-processes,
    validates, and returns it as a pd.DataFrame.

    Args:
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.
        offset_timestamp (str, optional): Timestamp that's used to compute the number of days
        each company's timestamps will be offset into the future, in order to synthesize a
        real-time data ingestion scenario. Defaults to data_config.offset_timestamp.
        duration (int, optional): The amount of pre-processed and validated data that's returned
        for each company, in days. Defaults to 14.

    Returns:
        pd.DataFrame: Pre-processed and validated data.
    """
    try:
        logger.info(
            f"Fetching raw data from the '{db_config.dbname}' database's \
'{db_config.schema}.{db_config.table.name}' table."
        )
        # a dictionary that maps each company ID to the number of days to add to its min timestamp
        # NOTE: this is done to synthesize a real-time data ingestion scenario
        future_mapper: dict[str, int] = (
            query_data(
                f"""
                WITH min_timestamps AS
                (
                    SELECT
                        company_id,
                        MIN(timestamp_utc) AS min_timestamp
                    FROM {db_config.dbname}.{db_config.schema}.{db_config.table.name}
                    GROUP BY 1
                    ORDER BY 1
                )
                SELECT
                    company_id,
                    CAST((EXTRACT(EPOCH FROM CAST('{offset_timestamp}' AS TIMESTAMP)) - \
                EXTRACT(EPOCH FROM min_timestamp)) / (60*60*24) AS INTEGER) AS n_days
                FROM min_timestamps
                ORDER BY 1\
                """
            )
            .set_index("company_id")
            ["n_days"]
            .to_dict()
        )

        # query each company's 'real-time' raw data, then pre-process and validate it, that is, ...
        # ensure that time steps are regularly spaced, and all nulls and duplicates and removed
        cols: ListConfig = db_config.table.columns
        dfs: list[pd.DataFrame] = []
        for company_id, n_days in future_mapper.items():
            data: pd.DataFrame = (
                query_data(
                    f"""
                    WITH real_time_scenario AS
                    (
                        SELECT
                            company_id,
                            {timestamp_col} + INTERVAL '{n_days}' DAY AS {timestamp_col},
                            {target_col},
                            ROW_NUMBER() OVER (PARTITION BY {timestamp_col}) AS rn
                        FROM {db_config.dbname}.{db_config.schema}.{db_config.table.name}
                        WHERE company_id = '{company_id}'
                    )
                    SELECT
                        company_id,
                        {timestamp_col},
                        {target_col}
                    FROM real_time_scenario
                    WHERE
                        {timestamp_col} BETWEEN DATE_TRUNC('hour', TIMEZONE('UTC', NOW())) - \
INTERVAL '{duration}' DAY AND DATE_TRUNC('hour', TIMEZONE('UTC', NOW()))
                        AND rn = 1
                    ORDER BY {timestamp_col}\
                    """
                )
                .set_index(timestamp_col)
                .asfreq("h")  # this ensures that the hourly time steps are reguarly spaced
            )
            to_energy_demand: dict[pd.Timestamp, float] = data[target_col].to_dict()
            data = (
                data
                .reset_index()
                .drop(columns=["company_id", target_col])
                .assign(
                    company_id=company_id,
                    target=data.reset_index()[timestamp_col].map(to_energy_demand).ffill()
                )
                .rename(columns={"target": target_col})
                [cols]
            )
            assert data.isna().sum().sum() + data.duplicated().sum() == 0
            dfs.append(data)
        logger.info("The raw data has been pre-processed and validated.")
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e


# -------------------------------------------------------------------------------------------------
# Data transformation
# -------------------------------------------------------------------------------------------------
@logger.catch
def transform_data(
    data: pd.DataFrame,
    target_col: str = data_config.target_column,
    timestamp_col: str = data_config.timestamp_column
) -> pd.DataFrame:
    """Transforms pre-processed and validated hourly energy demand data into an ML-ready
    dataset that contains datetime features, window features, lag features, and the
    corresponding target.

    Args:
        data (pd.DataFrame): DataFrame containing a 1-D time series of pre-processed and
        validated hourly energy demand data.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        timestamp_col (str, opitonal): Name of the column that contains the timestamps.
        Defaults to data_config.timestamp_column.

    Returns:
        pd.DataFrame: ML-ready dataset that contains datetime features, window features, lag
        features, and the corresponding target.
    """
    try:
        logger.info("Transforming the pre-processed data into features and targets.")
        dfs: list[pd.DataFrame] = []
        # iterate over each unique company ID, and ...
        for company_id in sorted(data["company_id"].unique()):
            # create its features, ...
            feature_matrix: pd.DataFrame = data.pipe(create_features, company_id)

            # add the corresponding target, and ...
            idx: pd.DatetimeIndex = feature_matrix.set_index(timestamp_col).index
            feature_matrix[target_col] = (
                data
                .query(f"company_id == '{company_id}'")
                .set_index(timestamp_col)
                .loc[idx, target_col]
                .tolist()
            )
            # append its ML-ready pd.DataFrame to the 'dfs' list
            dfs.append(feature_matrix)
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise e
