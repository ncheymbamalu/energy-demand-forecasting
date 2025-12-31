"""This script executes the inference pipeline."""

from datetime import datetime, timedelta

import polars as pl

from src.config import Paths
from src.data import load_preprocessed_data
from src.data import params as data_params
from src.data import transform_data
from src.forecast import get_one_step_forecast
from src.logger import logger


def main() -> None:
    """Executes the inference pipeline."""
    try:
        temporal_col: str = data_params.temporal_column
        data: pl.DataFrame = load_preprocessed_data().pipe(transform_data)
        next_hour: datetime = data[temporal_col].max() + timedelta(hours=1)
        logger.info(f"Forecasting the energy demand for <green>{str(next_hour)}</green>.")
        data = data.pipe(get_one_step_forecast).select("company_id", temporal_col, "forecast")
        if Paths.FORECAST_DATA.exists():
            (
                pl.concat((pl.read_parquet(Paths.FORECAST_DATA), data), how="vertical")
                .unique(subset=["company_id", temporal_col], keep="first")
                .sort(by=["company_id", temporal_col])
                .write_parquet(Paths.FORECAST_DATA)
            )
        else:
            data.sort(by=["company_id", temporal_col]).write_parquet(Paths.FORECAST_DATA)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
