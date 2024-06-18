from pathlib import Path, PosixPath


class Config:
    class Path:
        HOME_DIR: PosixPath = Path(__file__).parent.parent
