"""This module configures the project's paths and parameters."""

from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf


class Config:
    """Central configuration namespace for the project."""

    class Paths:
        """Defines and manages the absolute paths for the project's main files and directories.

        Attributes:
            HOME_DIR (Path): Project's root directory.
            ARTIFACTS_DIR (Path): Artifacts directory, `./artifacts/`.
            DATA_DIR (Path): Data directory, `./artifacts/data/`.
            MODEL_DIR (Path): Models directory, `./artifacts/model/`.
            ENV (Path): Environment variables file, `./.env`.
            PARAMS (Path): Parameters file, `./params.yaml`.
            PROCESSED_DATA (Path): Processed data, `./artifacts/data/processed.parquet`.
            FORECAST_DATA (Path): Forecast data, `./artifact/data/forecast.parquet`.
            MODEL (Path): Trained model file, `./artifacts/model/model.pkl`.
            MODELS_METADATA (Path): Trained models metadata file,
            `./artifacts/model/metadata.parquet`.
        """
        # directories
        HOME_DIR: Path = Path(__file__).parent.parent.parent.resolve()
        ARTIFACTS_DIR: Path = HOME_DIR / "artifacts"
        DATA_DIR: Path = ARTIFACTS_DIR / "data"
        MODEL_DIR: Path = ARTIFACTS_DIR / "model"

        # files
        ENV: Path = HOME_DIR / ".env"
        PARAMS: Path = HOME_DIR / "params.yaml"
        PROCESSED_DATA: Path = DATA_DIR / "processed.parquet"
        FORECAST_DATA: Path = DATA_DIR / "forecast.parquet"
        MODEL: Path = MODEL_DIR / "model.pkl"
        MODELS_METADATA: Path = MODEL_DIR / "metadata.parquet"

    @classmethod
    def make_dirs(cls) -> None:
        """Creates the project's main directories if they do not exist."""
        try:
            dirs: tuple[Path, ...] = (
                cls.Paths.ARTIFACTS_DIR,
                cls.Paths.DATA_DIR,
                cls.Paths.MODEL_DIR,
            )
            for _dir in dirs:
                _dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise e

    @classmethod
    def load_params(cls, module: str) -> DictConfig:
        """Loads module-specific parameters from cls.Paths.PARAMS.

        Args:
            module (str): Name of a user-defined module.

        Returns:
            DictConfig: Module-specific parameters in the form of user-defined
            key-value pairs.
        """
        try:
            params = cast(DictConfig, OmegaConf.load(cls.Paths.PARAMS))
            return params.get(module)
        except Exception as e:
            raise e
