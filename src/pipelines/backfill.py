"""This script executes a pipeline that backfills missing forecasts."""

from src.monitor import backfill_forecasts

if __name__ == "__main__":
    backfill_forecasts()
