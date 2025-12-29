"""This script executes the feature pipeline."""

import polars as pl

from src.config import Paths
from src.data import load_raw_data, params, preprocess_data, validate_data
from src.logger import logger


def main() -> None:
    """Executes the feature pipeline."""
    try:
        # load the existing data from ./artifacts/data/processed.parquet
        existing_data: pl.DataFrame = pl.read_parquet(Paths.PROCESSED_DATA)

        # load the current batch of raw data, then pre-process and validate
        data: pl.DataFrame = load_raw_data().pipe(preprocess_data)
        data.pipe(validate_data)

        # filter the current batch
        temporal_col: str = params.temporal_column
        data = (
            data
            .join(
                other=(
                    existing_data
                    .group_by("company_id", maintain_order=True)
                    .agg(pl.col(temporal_col).max().alias(f"latest_processed_{temporal_col}"))
                ),
                how="left",
                on="company_id",
                maintain_order="left"
            )
            .filter(pl.col(temporal_col).gt(pl.col(f"latest_processed_{temporal_col}")))
            .select(data.columns)
        )

        # (1) vertically concatenate the existing data with the current batch
        # (2) write to ./artifacts/data/processed.parquet
        logger.info(
            f"Updating ./{Paths.PROCESSED_DATA.relative_to(Paths.PROJECT_DIR)} with the current \
batch of pre-processed data."
        )
        (
            pl.concat((existing_data, data), how="vertical")
            .sort(by=["company_id", temporal_col])
            .write_parquet(Paths.PROCESSED_DATA)
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
