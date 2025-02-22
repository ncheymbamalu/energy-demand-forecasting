"""This module sets up the project's configurations."""

from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


class Paths:
    """Configuration for the project's primary directories and filepaths.

    Attributes:
        PROJECT_DIR (PosixPath): Project's root directory.
        DATA_DIR (PosixPath): Project's data directory, ~/data/.
        ARTIFACTS_DIR (PosixPath): Project's artifacts directory, ~/artifacts/.
        LOGS_DIR (PosixPath): Project's logs directory, ~/logs/.
        ENV (PosixPath): Project's .env file path, ~/.env.
        CONFIG (PosixPath): Project's configuration file path, ~/config.yaml.
        MODEL (PosixPath): Project's trained ML model file path, ~/artifacts/model.pkl.
    """
    PROJECT_DIR: PosixPath = Path(__file__).parent.parent.absolute()
    DATA_DIR: PosixPath = PROJECT_DIR / "data"
    ARTIFACTS_DIR: PosixPath = PROJECT_DIR / "artifacts"
    LOGS_DIR: PosixPath = PROJECT_DIR / "logs"
    ENV: PosixPath = PROJECT_DIR / ".env"
    CONFIG: PosixPath = PROJECT_DIR / "config.yaml"
    MODEL: PosixPath = ARTIFACTS_DIR / "model.pkl"


def load_config() -> DictConfig:
    """Loads Paths.CONFIG as a DictConfig object.

    Returns:
        DictConfig: Dictionary-like object with user-defined key-values pairs.
    """
    try:
        return OmegaConf.load(Paths.CONFIG)
    except Exception as e:
        raise e


db_config: DictConfig = load_config().database
data_config: DictConfig = load_config().data
model_config: DictConfig = load_config().model
