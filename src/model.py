"""This module provides functionality to train, validate, and tune select ML models."""

import pickle

from functools import partial

import pandas as pd

from catboost import CatBoostRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import Apply
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import Paths, data_config, model_config
from src.data import fetch_data, transform_data
from src.logger import logger
from src.utils import save_model, train_and_validate_model


# -------------------------------------------------------------------------------------------------
# Model building
# -------------------------------------------------------------------------------------------------
@logger.catch
def build_model(data: pd.DataFrame) -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Trains and evaluates select ML models and returns the one that produces
    the lowest average validation set RMSE.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.

    Returns:
        CatBoostRegressor | LGBMRegressor | XGBRegressor: Trained model
        that produced the lowest average validation RMSE.
    """
    try:
        # a dictionary of select pre-trained ML models
        models: dict[str, CatBoostRegressor | LGBMRegressor | XGBRegressor] = {
            model.__class__.__name__: model
            for model in [
                CatBoostRegressor(**model_config.CatBoostRegressor),
                LGBMRegressor(**model_config.LGBMRegressor),
                XGBRegressor(**model_config.XGBRegressor)
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
            .sort_values(by="rmse")
            .index[0]
        )
        logger.info(
            f"Training complete. The '{best_model}' produced the lowest average validation RMSE."
        )
        return models.get(best_model)
    except Exception as e:
        raise e


# -------------------------------------------------------------------------------------------------
# Hyperparameter tuning
# -------------------------------------------------------------------------------------------------
def load_model() -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Loads Paths.MODEL, if it exists, otherwise the model building process is
    initiated, which trains and cross-validates select ML models, saves the one
    that produces the lowest validation RMSE to Paths.MODEL, and returns it as a
    Python object.

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
            save_model(model)
        return model
    except Exception as e:
        raise e


def hyperopt_objective(
    param_space: dict[str, Apply],
    data: pd.DataFrame,
    model: CatBoostRegressor | LGBMRegressor | XGBRegressor
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
        model (CatBoostRegressor | LGBMRegressor | XGBRegressor): Trained ML model
        with default hyperparameters.

    Returns:
        dict[str, float | str]: A dictionary that contains the optimization objective for
        Bayesian hyperparameter optimization
    """
    try:
        model = (
            CatBoostRegressor(**(model.get_params() | param_space))
            if isinstance(model, CatBoostRegressor)
            else model.set_params(**param_space)
        )
        metric: float = data.pipe(train_and_validate_model, model)[1]
        return {"loss": metric, "status": STATUS_OK}
    except Exception as e:
        raise e


@logger.catch
def tune_model(
    data: pd.DataFrame,
    model: CatBoostRegressor | LGBMRegressor | XGBRegressor,
    target_col: str = data_config.target_column,
    objective_function: hyperopt_objective.__class__ = hyperopt_objective
) -> CatBoostRegressor | LGBMRegressor | XGBRegressor:
    """Returns a trained ML model with Bayesian tuned/optimized hyperparameters.

    Args:
        data (pd.DataFrame): ML-ready dataset containing datetime features, window
        features, lag features, and the corresponding target.
        model (CatBoostRegressor | LGBMRegressor | XGBRegressor): Trained ML model
        with default hyperparameters.
        target_col (str, optional): Name of the target variable.
        Defaults to data_config.target_column.
        objective_function (hyperopt_objective.__class__, optional): User-defined objective
        function. Defaults to hyperopt_objective.

    Returns:
        CatBoostRegressor | LGBMRegressor | XGBRegressor: Trained ML model with Bayesian-tuned
        hyperparameters.
    """
    try:
        # a list of features that the model was trained on
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
            fn=partial(objective_function, data=data, model=model),
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
