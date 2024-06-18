import logging

from datetime import datetime
from pathlib import PosixPath

from src.config import Config

FILENAME: str = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGS_DIR: PosixPath = Config.Path.HOME_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / FILENAME,
    format="[%(asctime)s] - %(pathname)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
