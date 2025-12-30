"""This module contains functionality for training, validating, and tuning objects of type
LGBMRegressor and XGBRegressor."""

import json
import pickle

from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path, PosixPath

import numpy as np
import polars as pl

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import Apply
from lightgbm import LGBMRegressor
from omegaconf import DictConfig, ListConfig, OmegaConf
from xgboost import XGBRegressor

from src.config import Paths, load_params
from src.data import load_preprocessed_data
from src.data import params as data_params
from src.data import transform_data
from src.logger import logger
from src.utils import compute_evaluation_metrics, get_time_series_splits

params: DictConfig = load_params(Path(__file__).stem)


def train_model(
    data: pl.DataFrame,
    model: LGBMRegressor | XGBRegressor
) -> tuple[LGBMRegressor | XGBRegressor, float]:
    """Trains a model via walk-forward validation and returns it along with its
    average validation RMSE.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.
        model (LGBMRegressor | XGBRegressor): Pre-trained model.

    Returns:
        tuple[LGBMRegressor | XGBRegressor, float]: Trained model and its average
        validation RMSE.
    """
    try:
        target: str = data_params.target_column
        features: list[str] = data.drop(data_params.columns).columns
        temporal_col: str = data_params.temporal_column
        k: int = params.k

        # get the time series splits
        train_splits, fold_splits = get_time_series_splits(data, temporal_col, k)

        # an empty list to store each fold's validation RMSE
        metrics: list[float] = []

        # iterate over each fold, and ...
        for train_split, fold_split in zip(train_splits, fold_splits):
            # create the train set
            expression: pl.Expr = pl.col(temporal_col).le(pl.lit(train_split))
            x_train: pl.DataFrame = data.filter(expression).select(features)
            y_train: pl.Series = data.filter(expression)[target]

            # create the validation set
            expression = pl.col(temporal_col).is_between(train_split, fold_split, closed="right")
            x_val: pl.DataFrame = data.filter(expression).select(features)
            y_val: pl.Series = data.filter(expression)[target]

            # train and evaluate the model
            # NOTE: LightGBM doesn't accept a Polars Series object as input, so the train ...
            # and validation set target variables, 'y_train' and 'y_val', are converted to ...
            # lists when the model is an object of type LGBMRegressor
            if isinstance(model, LGBMRegressor):
                model.fit(x_train, y_train.to_list(), eval_set=[(x_val, y_val.to_list())])
            else:
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

            # compute the validation RMSE and append it to the 'metrics' list
            metric: float = compute_evaluation_metrics(y_val, model.predict(x_val)).get("rmse")
            metrics.append(metric)
        return model, np.mean(metrics).item()
    except Exception as e:
        raise e


def build_model(data: pl.DataFrame) -> LGBMRegressor | XGBRegressor:
    """Trains two models, an LGBMRegressor and XGBRegressor, and returns the one
    that produces the lowest average validation RMSE.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.

    Returns:
        LGBMRegressor | XGBRegressor: Trained model.
    """
    try:
        # a dictionary of select pre-trained models
        models: dict[str, LGBMRegressor | XGBRegressor] = {
            model.__class__.__name__: model
            for model in (
                LGBMRegressor(**params.lightgbm.base_params),
                XGBRegressor(**params.xgboost.base_params)
            )
        }

        # an empty dictionary to store each model's average validation RMSE
        report: dict[str, float] = {}
        for name, model in models.items():
            logger.info(f"Training initiated for the <green>{name}</green>.")
            model, metric = train_model(data, model)
            models[name] = model
            report[name] = metric

        # get the name of the model that produced the lowest average validation RMSE
        name = sorted(report.items(), key=lambda items: items[1])[0][0]

        logger.info(
            f"Training complete! The <green>{name}</green> produced the lowest validation RMSE."
        )
        return models.get(name)
    except Exception as e:
        raise e


def hyperopt_objective(
    param_space: dict[str, Apply],
    data: pl.DataFrame,
    model: LGBMRegressor | XGBRegressor
) -> dict[str, float | str]:
    """Updates the model's hyperparameters with those specified in param_space, then
    computes and returns the corresponding validation metric.

    Args:
        param_space (dict[str, Apply]): The model's hyperparameter search space.
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.
        model (LGBMRegressor | XGBRegressor): Pre-trained or trained model.

    Returns:
        dict[str, float | str]: Dictionary that contains the validation metric, which is
        the objective (loss) to be optimized (minimized).
    """
    try:
        model = model.set_params(**param_space)
        metric: float = train_model(data, model)[1]
        return {"loss": metric, "status": STATUS_OK}
    except Exception as e:
        raise e


def tune_model(
    data: pl.DataFrame,
    model: LGBMRegressor | XGBRegressor
) -> tuple[LGBMRegressor | XGBRegressor, float]:
    """Returns a trained model with Bayesian-tuned hyperparameters and its average
    validation RMSE.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.
        model (LGBMRegressor | XGBRegressor): Pre-trained or trained model.

    Returns:
        tuple[LGBMRegressor | XGBRegressor, float]: Trained model with Bayesian-tuned
        hyperparameters and its average validation RMSE.
    """
    try:
        logger.info(
            f"Hyperparameter tuning initiated for the <green>{model.__class__.__name__}</green>."
        )

        # define the model's hyperparameter search space
        param_space: dict[str, Apply] = (
            {
                "n_estimators": hp.randint("n_estimators", 100, 500),
                "max_depth": hp.randint("max_depth", 3, 10),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                "num_leaves": hp.randint("num_leaves", 8, 256),
                "min_data_in_leaf": hp.randint("min_data_in_leaf", 5, 300)
            }
            if isinstance(model, LGBMRegressor)
            else
            {
                "n_estimators": hp.randint("n_estimators", 100, 500),
                "max_depth": hp.randint("max_depth", 3, 10),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                "min_child_weight": hp.randint("min_child_weight", 0, 10),
                "reg_alpha": hp.uniform("reg_alpha", 0, 10)
            }
        )

        # tune the model's hyperparameters
        tuned_params: dict[str, float] = fmin(
            fn=partial(hyperopt_objective, data=data, model=model),
            space=param_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=Trials(),
            verbose=1
        )

        # instantiate a new model with the tuned hyperparameters, then train and validate it
        if isinstance(model, LGBMRegressor):
            base_params: dict[str, int | str] = OmegaConf.to_container(params.lightgbm.base_params)
            model = LGBMRegressor(**(base_params | tuned_params))
        else:
            base_params = OmegaConf.to_container(params.xgboost.base_params)
            model = XGBRegressor(**(base_params | tuned_params))
        model, metric = train_model(data, model)

        logger.info("Hyperparameter tuning complete.")
        return model, metric
    except Exception as e:
        raise e


def save_model(model: LGBMRegressor | XGBRegressor, path: PosixPath | str = Paths.MODEL) -> None:
    """Saves the model to path.

    Args:
        model (LGBMRegressor | XGBRegressor): Trained model.
        path (PosixPath | str, optional): Trained model file path. Defaults to Paths.MODEL
    """
    try:
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise e


def create_model_metadata(
    data: pl.DataFrame,
    model: LGBMRegressor | XGBRegressor,
    metric: float
) -> pl.DataFrame:
    """Creates the model's metadata.

    Args:
        data (pl.DataFrame): ML-ready data, that is, data that's been pre-processed,
        validated, and converted to lag features, window features (average lags),
        datetime features, and corresponding labels.
        model (LGBMRegressor | XGBRegressor): Trained model.
        metric (float): The model's average validation RMSE.

    Returns:
        pl.DataFrame: Model metadata.
    """
    try:
        temporal_col: str = data_params.temporal_column
        lookback: int = data_params.lookback
        n_lags: int = len([col for col in data.columns if col.startswith("lag")])
        start: datetime = data[temporal_col].min() - timedelta(hours=n_lags)
        end: datetime = start + timedelta(days=lookback)

        # get the model's parameters
        model_params: ListConfig = (
            list(params.lightgbm.base_params) + params.lightgbm.hyperparams
            if isinstance(model, LGBMRegressor)
            else list(params.xgboost.base_params) + params.xgboost.hyperparams
        )

        # create the model's metadata
        metadata: dict[str, datetime | float | int | str] = {
            "name": model.__class__.__name__,
            "data_location": f"./{Paths.PROCESSED_DATA.relative_to(Paths.PROJECT_DIR)}",
            "data_start": start,
            "data_end": end,
            "when_trained": datetime.now(timezone.utc).replace(microsecond=0),
            "params": json.dumps({
                param: value.item() if isinstance(value, np.number) else value
                for param, value in model.get_params().items()
                if param in model_params
            }),
            "rmse": metric
        }
        return pl.DataFrame(metadata)
    except Exception as e:
        raise e


def save_model_metadata(
    data: pl.DataFrame,
    path: PosixPath | str = Paths.MODELS_METADATA
) -> None:
    """Saves the model's metadata to path.

    Args:
        data (pl.DataFrame): Model metadata.
        path (PosixPath | str, optional). Trained models metadata file path.
        Defaults to Paths.MODELS_METADATA.
    """
    try:
        (
            (
                pl.concat((pl.read_parquet(path), data), how="vertical")
                .sort(by="when_trained", descending=True)
                .write_parquet(path)
            )
            if Path(path).exists()
            else data.write_parquet(path)
        )
    except Exception as e:
        raise e


def load_model() -> LGBMRegressor | XGBRegressor:
    """Loads Paths.MODEL, if it exists, otherwise the model building process
    is initiated, which trains, evaluates, and tunes either an LGBMRegressor
    or XGBRegressor on the latest data, saves it to Paths.MODEL, saves its
    metadata to Paths.MODELS_METADATA, and returns it as a Python object.

    Returns:
        LGBMRegressor | XGBRegressor: Trained model.
    """
    try:
        # load Paths.MODEL, if it exits, otherwise start the model building process
        if Paths.MODEL.exists():
            with open(Paths.MODEL, "rb") as file:
                model: XGBRegressor = pickle.load(file)
        else:
            logger.info(
                f"./{Paths.MODEL.relative_to(Paths.PROJECT_DIR)} not found. Starting the model \
building process."
            )
            data: pl.DataFrame = load_preprocessed_data().pipe(transform_data)
            model, metric = tune_model(data, build_model(data))
            data = create_model_metadata(data, model, metric)
            save_model(model)
            save_model_metadata(data)
        return model
    except Exception as e:
        raise e
