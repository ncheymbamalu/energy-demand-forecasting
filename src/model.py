"""This module provides functionality to train, validate, and tune select ML models."""

import pickle

from functools import partial
from pathlib import PosixPath

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import Apply
from lightgbm import LGBMRegressor
from omegaconf import DictConfig
from xgboost import XGBRegressor

from src.config import Paths, load_config
from src.data import DATA_CONFIG, fetch_data, transform_data
from src.database import DB_CONFIG
from src.logger import logger

MODEL_CONFIG: DictConfig = load_config().model


def compute_metrics(y: np.ndarray | pd.Series, yhat: np.ndarray | pd.Series) -> dict[str, float]:
    """Computes the root mean squared error, RMSE, and coefficient of
    determination, R², between y and yhat.

    Args:
        y (np.ndarray | pd.Series): Observations.
        yhat (np.ndarray | pd.Series): Predictions.

    Returns:
        dict[str, float]: RMSE and R²
    """
    try:
        t: np.ndarray | pd.Series = y - y.mean()
        sst: float = t.dot(t)
        e: np.ndarray | pd.Series = y - yhat
        sse: float = e.dot(e)
        rmse: float = np.sqrt(sse / e.shape[0])
        r_squared: float = 1 - (sse / sst)
        return {"rmse": round(rmse, 4), "r_squared": round(r_squared, 4)}
    except Exception as e:
        raise e


def get_time_series_splits(
    data: pd.DataFrame,
    timestamp_col: str = DATA_CONFIG.timestamp_column,
    n_splits: int = MODEL_CONFIG.n_splits,
    train_size: float = MODEL_CONFIG.train_size
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Returns two lists, one containing the train set splits, and the other containing the
    corresponding validation set splits.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.
        timestamp_col (str, opitonal): Name of the column that contains the timestamps.
        Defaults to DATA_CONFIG.timestamp_column.
        n_splits (int, optional): Number of train/validation splits to generate.
        Defaults to MODEL_CONFIG.n_splits.
        train_size (float, optional): Percentage of training data per split.
        Defaults to MODEL_CONFIG.train_size.

    Returns:
        tuple[list[pd.Timestamp], list[pd.Timestamp]]: Train set and validation set splits.
    """
    try:
        unique_timestamps: list[pd.Timestamp] = sorted(data[timestamp_col].unique())
        split_size: int = len(unique_timestamps) // n_splits
        indices: list[int] = [
            min(split_size*idx, len(unique_timestamps) - 1) for idx in range(1, n_splits + 1)
        ]
        train_splits: list[pd.Timestamp] = [
            unique_timestamps[int(train_size*idx)] for idx in indices
        ]
        val_splits: list[pd.Timestamp] = [unique_timestamps[idx] for idx in indices]
        return train_splits, val_splits
    except Exception as e:
        raise e


def train_and_validate_model(
    data: pd.DataFrame,
    model: CatBoostRegressor | LGBMRegressor | XGBRegressor,
    target_col: str = DATA_CONFIG.target_column,
    timestamp_col: str = DATA_CONFIG.timestamp_column
) -> tuple[CatBoostRegressor | LGBMRegressor | XGBRegressor, float]:
    """Trains an and returns an ML model along with its average validation RMSE.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.
        model (CatBoostRegressor | LGBMRegressor | XGBRegressor): Pre-trained ML model.
        target_col (str, optional): Name of the column that contains the target variable.
        Defaults to DATA_CONFIG.target_column.
        timestamp_col (str, optional): Name of the column that contains the timestamps.
        Defaults to DATA_CONFIG.timestamp_column.

    Returns:
        tuple[CatBoostRegressor | LGBMRegressor | XGBRegressor, float]: Trained ML model
        and its average validation RMSE.
    """
    try:
        # a list containing the names of the features
        features: list[str] = [col for col in data.columns if col not in DB_CONFIG.table.columns]

        # get the train and validation set splits
        train_splits, val_splits = data.pipe(get_time_series_splits)

        # an empty list to store the model's validation set metrics, one per split
        val_metrics: list[float] = []
        for train_split, val_split in zip(train_splits, val_splits):
            query: str = f"{timestamp_col} <= '{train_split}'"
            x_train: pd.DataFrame = data.query(query)[features]
            y_train: pd.Series = data.query(query)[target_col]
            query = f"{timestamp_col} > '{train_split}' & {timestamp_col} <= '{val_split}'"
            x_val: pd.DataFrame = data.query(query)[features]
            y_val: pd.Series = data.query(query)[target_col]

            # if the model is an object of type, 'CatBoostRegressor'
            if isinstance(model, CatBoostRegressor):
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
                metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

            # if the model is an object of type, 'LGBMRegressor'
            elif isinstance(model, LGBMRegressor):
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
                metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

            # if the model is an object of type, 'XGBRegressor'
            else:
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                metric: float = compute_metrics(y_val, model.predict(x_val)).get("rmse")

            # append the validation set metric to the 'val_metrics' list
            val_metrics.append(metric)
        return model, np.mean(val_metrics)
    except Exception as e:
        raise e


def save_model(model: CatBoostRegressor | LGBMRegressor | XGBRegressor) -> None:
    """Saves model to Paths.MODEL.

    Args:
        model (CatBoostRegressor | LGBMRegressor | XGBRegressor): ML model.
    """
    try:
        artifacts_dir: PosixPath = Paths.ARTIFACTS_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(Paths.MODEL, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise e


@logger.catch
def build_model(data: pd.DataFrame) -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Trains and evaluates select ML models and returns the one that produces
    the lowest average validation set RMSE.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.

    Returns:
        CatBoostRegressor | LGBMRegressor | NaiveForecast | XGBRegressor: Trained model
        that produced the lowest average validation RMSE.
    """
    try:
        # a dictionary of select pre-trained ML models
        models: dict[str, CatBoostRegressor | LGBMRegressor | XGBRegressor] = {
            model.__class__.__name__: model
            for model in [
                CatBoostRegressor(**MODEL_CONFIG.CatBoostRegressor),
                LGBMRegressor(**MODEL_CONFIG.LGBMRegressor),
                XGBRegressor(**MODEL_CONFIG.XGBRegressor)
            ]
        }

        # an empty dictionary to map each trained ML model to its corresponding metric
        report: dict[str, float] = {}

        for model_name, model in models.items():
            logger.info(f"Training initiated for the '{model_name}'.")

            # train and cross-validate the model
            model, metric = data.pipe(train_and_validate_model, model)

            # replace the pre-trained model in the 'models' dictionary with its trained version
            models[model_name] = model

            # map the trained model to its corresponding metric
            report[model_name] = metric

        # get the name of the trained model that produced the 'best' metric, that is, ...
        # the lowest average validation set RMSE
        best_model: str = (
            pd.DataFrame.from_dict(report, orient="index", columns=["rmse"])
            .sort_values("rmse")
            .index[0]
        )
        logger.info(
            f"Training complete. The '{best_model}' produced the lowest average validation RMSE."
        )
        return models.get(best_model)
    except Exception as e:
        raise e


def load_model() -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Loads Paths.MODEL, if it exists, otherwise the model building process is
    initiated, which trains and cross-validates select ML models, saves the one
    that produces the lowest validation set RMSE to Paths.MODEL, and returns it
    as a Python object.

    Returns:
        CatBoostRegressor | LGBMRegressor | XGBRegressor: Trained ML model.
    """
    try:
        # load Paths.MODEL, if it exits, otherwise start the model building process
        if Paths.MODEL.exists():
            with open(Paths.MODEL, "rb") as file:
                model: CatBoostRegressor | LGBMRegressor | XGBRegressor = pickle.load(file)
        else:
            logger.info(
                f"~/{Paths.MODEL.parent.name}/{Paths.MODEL.name} not found. Starting the model \
building process."
            )
            model = fetch_data().pipe(transform_data).pipe(build_model)
        return model
    except Exception as e:
        raise e


def hyperopt_objective(
    param_space: dict[str, Apply],
    data: pd.DataFrame
) -> dict[str, float | str]:
    """Loads the trained ML model, updates its hyperparameters with those specified in
    'param_space', and computes the corresponding validation metric.

    NOTE: The validation metric, i.e., the loss, is the objective that hyperopt seeks to
    optimize. By default, hyperopt performs minimization; however, if the validation metric
    is a value that needs to be maximized, like accuracy, f1-score, auc etc., then it should
    be negated prior to being returned as the output.

    Args:
        param_space (dict[str, Apply]): The ML model's hyperparameter search space.
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.

    Returns:
        dict[str, float | str]: A dictionary that contains the optimization objective for
        Bayesian hyperparameter optimization
    """
    try:
        model: CatBoostRegressor | LGBMRegressor | XGBRegressor = load_model()
        model = (
            CatBoostRegressor(**(model.get_params() | param_space))
            if isinstance(model, CatBoostRegressor)
            else model.set_params(**param_space)
        )
        metric: float = data.pipe(train_and_validate_model, model)[1]
        return {"loss": metric, "status": STATUS_OK}
    except Exception as e:
        raise e


def tune_model(
    data: pd.DataFrame,
    target_col: str = DATA_CONFIG.target_column,
    objective_function: hyperopt_objective.__class__ = hyperopt_objective
) -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Returns a trained ML model with Bayesian tuned/optimized hyperparameters.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.
        target_col (str, optional):  Name of the column that contains the target variable.
        Defaults to DATA_CONFIG.target_column.
        objective_function (hyperopt_objective.__class__, optional): User-defined objective
        function. Defaults to hyperopt_objective.

    Returns:
        CatBoostRegressor | LGBMRegressor | XGBRegressor: Trained ML model with Bayesian-tuned
        hyperparameters.
    """
    try:
        # load the trained model and get the features it was trained on
        model: CatBoostRegressor | LGBMRegressor | XGBRegressor = load_model()
        features: list[str] = (
            model.feature_names_ if isinstance(model, CatBoostRegressor)
            else model.feature_names_in_.tolist()
        )
        logger.info(f"Hyperparameter tuning initiated for the '{model.__class__.__name__}'.")
        param_space: dict[str, Apply] = (
            {  # CatBoostRegressor hyperparameters
                "iterations": hp.randint("iterations", 100, 300),
                "depth": hp.randint("depth", 3, 10),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": hp.uniform("l2_leaf_reg", 0, 10)
            } if isinstance(model, CatBoostRegressor)
            else {  # LGBMRegressor hyperparams
                "n_estimators": hp.randint("n_estimators", 100, 300),
                "max_depth": hp.randint("max_depth", 3, 10),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                "num_leaves": hp.randint("num_leaves", 8, 256),
                "min_data_in_leaf": hp.randint("min_data_in_leaf", 5, 300),
            } if isinstance(model, LGBMRegressor)
            else {  # XGBRegressor hyperparams
                "n_estimators": hp.randint("n_estimators", 100, 300),
                "max_depth": hp.randint("max_depth", 3, 10),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                "min_child_weight": hp.randint("min_child_weight", 0, 10)
            }
        )
        tuned_params: dict[str, float] = fmin(
            fn=partial(objective_function, data=data),
            space=param_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=Trials(),
            verbose=1
        )
        model = (
            CatBoostRegressor(**(tuned_params | {"silent": True}))
            if isinstance(model, CatBoostRegressor)
            else LGBMRegressor(**(tuned_params | {"verbosity": -1}))
            if isinstance(model, LGBMRegressor)
            else XGBRegressor(**tuned_params)
        )
        model.fit(data[features], data[target_col])
        logger.info("Hyperparameter tuning complete.")
        return model
    except Exception as e:
        raise e
