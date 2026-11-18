"""This module contains functionality that is used in other modules and scripts."""

from datetime import UTC, date, datetime
from pathlib import Path

import holidays
import polars as pl


def get_current_hour() -> datetime:
    """Returns the current UTC time truncated to the start of the hour.

    Returns:
        datetime: The current hour in UTC, as a naive datetime (no tzinfo) with minutes,
            seconds, and microseconds set to zero.
    """
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)


def sink_atomically(plan: pl.LazyFrame, path: Path) -> None:
    """Streams plan to a temporary file, then atomically swaps it into place.

    Args:
        plan (pl.LazyFrame): Query whose result is written to path.
        path (Path): Destination file path.
    """
    tmp_path: Path = path.with_name(f"{path.stem}_tmp{path.suffix}")
    try:
        plan.sink_parquet(tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def get_us_holidays(data: pl.DataFrame) -> list[date]:
    """Returns the US federal holidays for every year covered by the data.

    Holidays come from the `holidays` package and include observed dates, e.g., the Friday
    before a July 4th that falls on a Saturday.

    Args:
        data (pl.DataFrame): Data with a `date` column of dtype `pl.Date`.

    Returns:
        list[date]: Holiday dates from the minimum year through the maximum year of the `date`
            column, inclusive.
    """
    min_year, max_year = (
        data
        .select(
            pl.min("date").dt.year().alias("min_year"),
            pl.max("date").dt.year().alias("max_year")
        )
        .row(0)
    )
    return list(holidays.country_holidays(country="US", years=range(min_year, max_year + 1)))
