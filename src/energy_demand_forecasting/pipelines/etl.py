"""ETL pipeline entry point for the processed hourly energy demand data.

Fetches the latest batch of hourly energy demand for each PJM subregion from the EIA Open Data
API (v2), pre-processes and validates it, and upserts it into the processed data file
(`./artifacts/data/processed.parquet`). Runs on a schedule via GitHub Actions; the processed
data file is versioned with DVC.

Usage:
    `uv run python src/energy_demand_forecasting/pipelines/etl.py`
"""

from pathlib import Path

import polars as pl

from energy_demand_forecasting.config import Config
from energy_demand_forecasting.data import fetch_data, preprocess_data, validate_data
from energy_demand_forecasting.logger import logger
from energy_demand_forecasting.utils import sink_atomically


pl.Config.set_engine_affinity(engine="streaming")


def main() -> None:
    """Runs the ETL pipeline and upserts the latest batch of data into the processed data file.

    Fetches the latest batch of hourly energy demand data from the EIA API, then pre-processes
    and validates it. If the processed data file exists, the batch is merged into it: for each
    (`subregion_id`, `timestamp_utc`) pair, real records take precedence over imputed records,
    and among records with the same `is_imputed` value, the one with the latest
    `processed_at_utc` is kept. Otherwise, the batch becomes the initial processed data file.
    In both cases, the file is written atomically.
    """
    # process the latest batch of data
    data: pl.DataFrame = fetch_data().pipe(preprocess_data)
    data.pipe(validate_data)

    # write to ./artifacts/data/processed.parquet
    path: Path = Config.Paths.PROCESSED_DATA
    data_dir: Path = path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        logger.info(
            "Updating ./{} with the latest batch of data.", path.relative_to(Config.Paths.HOME_DIR)
        )
        (
            pl.concat((pl.scan_parquet(path), data.lazy()))
            .sort(
                "subregion_id",
                "timestamp_utc",
                "is_imputed",
                "processed_at_utc",
                descending=[False, False, False, True]
            )
            .unique(subset=["subregion_id", "timestamp_utc"], keep="first", maintain_order=True)
            .pipe(sink_atomically, path)
        )
    else:
        logger.info("Writing to ./{}", path.relative_to(Config.Paths.HOME_DIR))
        data.lazy().pipe(sink_atomically, path)


if __name__ == "__main__":
    main()
