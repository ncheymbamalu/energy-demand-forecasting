"""This module configures the project's paths and parameters."""

from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


class Paths:
    """A class that defines the main directories and files.

    Attributes:
        PROJECT_DIR (PosixPath): Project's root directory.
        ARTIFACTS_DIR (PosixPath): Artifacts directory, ./artifacts/.
        DATA_DIR (PosixPath): Data directory, ./artifacts/data/.
        MODEL_DIR (PosixPath): Models directory, ./artifacts/model/.
        ENV (PosixPath): Environment variables file, ./.env.
        PARAMS (PosixPath): Parameters file, ./params.yaml.
        RAW_DATA (PosixPath): Raw data, ./artifacts/data/raw.parquet.
        PROCESSED_DATA (PosixPath): Processed data, ./artifacts/data/processed.parquet.
        FORECAST_DATA (PosixPath): Forecast data, ./artifact/data/forecast.parquet
        MODEL (PosixPath): Trained model file, ./artifacts/model/model.pkl.
        MODELS_METADATA (PosixPath): Trained models metadata file,
        ./artifacts/model/metadata.parquet
    """
    # directories
    PROJECT_DIR: PosixPath = Path(__file__).parent.parent.absolute().resolve()
    ARTIFACTS_DIR: PosixPath = PROJECT_DIR / "artifacts"
    DATA_DIR: PosixPath = ARTIFACTS_DIR / "data"
    MODEL_DIR: PosixPath = ARTIFACTS_DIR / "model"

    # files
    ENV: PosixPath = PROJECT_DIR / ".env"
    PARAMS: PosixPath = PROJECT_DIR / "params.yaml"
    RAW_DATA: PosixPath = DATA_DIR / "raw.parquet"
    PROCESSED_DATA: PosixPath = DATA_DIR / "processed.parquet"
    FORECAST_DATA: PosixPath = DATA_DIR / "forecast.parquet"
    MODEL: PosixPath = MODEL_DIR / "model.pkl"
    MODELS_METADATA: PosixPath = MODEL_DIR / "metadata.parquet"


def load_params(module: str) -> DictConfig:
    """Loads module-specific parameters from ./params.yaml.

        Args:
            module (str): Name of a user-defined module.

        Returns:
            DictConfig: Module-specific parameters in the form of user-defined
            key-value pairs.
    """
    try:
        return OmegaConf.load(Paths.PARAMS).get(module)
    except Exception as e:
        raise e
