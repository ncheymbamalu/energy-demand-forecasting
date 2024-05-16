import logging
import os

from datetime import datetime

FILENAME: str = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
DIRECTORY: str = os.path.join(os.getcwd(), "logs")
os.makedirs(DIRECTORY, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(DIRECTORY, FILENAME),
    format="[%(asctime)s] - %(pathname)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
