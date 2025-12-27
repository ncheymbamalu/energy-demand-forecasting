"""This script executes the feature pipeline."""

import polars as pl

from src.config import Paths
from src.data import load_data, params, preprocess_data, validate_data
from src.logger import logger


def main() -> None:
    """Executes the feature pipeline."""
    try:
        temporal_col: str = params.temporal_column

        # load and pre-process the current batch of raw data
        data: pl.DataFrame = load_data().pipe(preprocess_data)

        # vertically concatenate the existing pre-processed data the current batch
        data = (
            pl.concat((pl.read_parquet(Paths.PROCESSED_DATA), data), how="vertical")
            .unique(subset=["company_id", temporal_col], keep="first")
            .sort(by=["company_id", temporal_col])
        )

        # validate the vertically concatenated data
        data.pipe(validate_data)

        # write the vertically concatenated data to ./artifacts/data/processed.parquet
        logger.info(
            f"Updating ./{Paths.PROCESSED_DATA.relative_to(Paths.PROJECT_DIR)} with the current \
batch of pre-processed data."
        )
        data.write_parquet(Paths.PROCESSED_DATA)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
