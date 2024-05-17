"""Model training, time series cross-validation, and Bayesian hyperparameter tuning"""

import random

from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import Apply
from xgboost import XGBRegressor

from src.ingest import ingest_data, query_ids
from src.logger import logging
from src.transform import transform_data


def get_rsquared(y: pd.Series | np.ndarray, yhat: pd.Series | np.ndarray) -> float:
    """
    Calculates and returns the R² between y and yhat

    Args:
        y: target vector
        yhat: prediction vector

    Returns:
        R² between y and yhat
    """
    try:
        # ensure that y and yhat are 1-D vectors
        y = y.ravel() if y.ndim > 1 else y
        yhat = yhat.ravel() if yhat.ndim > 1 else yhat

        # compute the R²
        t: np.ndarray = y - y.mean()
        sst: float = t.dot(t)
        e: np.ndarray = y - yhat
        sse: float = e.dot(e)
        return 1 - (sse / sst)
    except Exception as e:
        raise e


# pylint: disable=too-many-locals
def get_tss_metric(
    model: XGBRegressor,
    feature_matrix: pd.DataFrame,
    target_vector: pd.Series,
    step_percentage: float = 0.2,
    forecast_horizon: int = 24,
) -> float:
    """
    Computes the average train set R² from k time series splits

    Args:
        model (XGBRegressor): Regressor
        feature_matrix (pd.DataFrame): Matrix of lag features, window features,
        and datetime features
        target_vector (pd.Series): Target vector
        step_percentage (float): The percentage of data to add to each
        subsequent fold's train set
        forecast_horizon (int): Number of time steps to forecast into the future.
        Defaults to 24. NOTE: this is the size of each fold's validation set

    Returns:
        float: Average train set R² across all folds
    """
    try:
        n_records: int = feature_matrix.shape[0]
        step_size: int = int(step_percentage * n_records)
        train_indices: list[int] = np.arange(step_size, n_records + 1, step_size).tolist()
        val_indices: list[int] = [idx + forecast_horizon for idx in train_indices]
        idx_pairs: list[tuple[int, int]] = [
            idx_pair
            for idx_pair in zip(train_indices, val_indices)
            if n_records - idx_pair[0] > forecast_horizon / 2
        ]
        fold_metrics: list[float] = []
        for train_idx, val_idx in idx_pairs:
            x_train: pd.DataFrame = feature_matrix.iloc[:train_idx, :]
            y_train: pd.Series = target_vector.iloc[:train_idx]
            x_val: pd.DataFrame = feature_matrix.iloc[train_idx:val_idx, :]
            y_val: pd.Series = target_vector.iloc[train_idx:val_idx]
            model.fit(
                x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=False
            )
            metric: float = get_rsquared(y_train.values, model.predict(x_train))
            fold_metrics.append(max(0, metric))
        return np.mean(fold_metrics)
    except Exception as e:
        raise e


def objective(space: dict[str, Apply], x: pd.DataFrame, y: pd.Series) -> dict[str, float | str]:
    """Computes the time series split (tss) metric, which is (A) the average train
    set R², and (B) the objective for bayesian hyperparameter optimization

    Args:
        space (dict[str, Apply]): Hyperparameter search space
        model: (XGBRegressor | CatBoostRegressor): Regressor
        x (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector

    Returns:
        dict[str, float | str]: A dictionary that contains the objective (loss) to
        minimize during Bayesian hyperparameter tuning
    """
    try:
        model: XGBRegressor = XGBRegressor(
            objective="reg:squarederror",
            booster="gbtree",
            base_score=0.5,
            early_stopping_rounds=20,
            n_jobs=-1,
        ).set_params(**space)
        avg_metric: float = get_tss_metric(model, x, y)
        return {"loss": 1 - avg_metric, "status": STATUS_OK}
    except Exception as e:
        raise e


def train_model(
    feature_matrix: pd.DataFrame, target_vector: pd.Series, obj_func: Callable = objective
) -> XGBRegressor:
    """
    Returns an object of type, 'XGBRegressor', whose hyperparameters, which are defined
    in the 'param_space' dictionary, will have been tuned via Bayesian optimization

    Args:
        feature_matrix (pd.DataFrame): Matrix of lag features, window features,
        and datetime features
        target_vector (pd.Series): Target vector
        obj_func (Callable): Objective function. Defaults to 'objective', which
        is the user-defined objective function

    Returns:
        XGBRegressor: Model with optimized hyperparameters
    """
    try:
        logging.info("Initiating model training and hyperparameter tuning.")
        param_space: dict[str, Apply] = {
            "n_estimators": hp.randint("n_estimators", 100, 2000),
            "max_depth": hp.randint("max_depth", 3, 20),
            "learning_rate": hp.uniform("learning_rate", 0.05, 0.5),
            "gamma": hp.uniform("gamma", 0, 1),
            "min_child_weight": hp.uniform("min_child_weight", 0, 50),
        }
        tuned_params: dict[str, float] = fmin(
            fn=partial(obj_func, x=feature_matrix, y=target_vector),
            space=param_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=Trials(),
            verbose=1,
        )
        tuned_model: XGBRegressor = XGBRegressor(
            objective="reg:squarederror", booster="gbtree", base_score=0.5, n_jobs=-1
        ).set_params(**tuned_params)
        logging.info("Model training complete.")
        return tuned_model
    except Exception as e:
        raise e


if __name__ == "__main__":
    ids: list[str] = query_ids()
    df_stationary, target = ingest_data(random.choice(ids))
    x_matrix, y_vector = transform_data(df_stationary, target)
    reg: XGBRegressor = train_model(x_matrix, y_vector)
    print(
        {param: value for param, value in reg.get_params().items() if value and param != "missing"}
    )
