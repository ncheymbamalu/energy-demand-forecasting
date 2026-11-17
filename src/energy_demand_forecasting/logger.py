"""This module configures the project's logging."""

import sys
from pathlib import Path

from loguru import logger

from energy_demand_forecasting.config import Config


logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")

LOGS_DIR: Path = Config.Paths.HOME_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE: Path = LOGS_DIR / "file_{time:YYYY_MM_DD_HH_mm_ss}.log"
logger.add(LOG_FILE, retention="2 days", level="DEBUG")
logger = logger.opt(colors=True)
