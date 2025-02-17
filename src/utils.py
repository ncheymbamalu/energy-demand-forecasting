"""This module contains utility/helper functions."""

import matplotlib.pyplot as plt
import pandas as pd

from src.data import DATA_CONFIG
from src.model import get_time_series_splits


def plot_time_series_splits(
    data: pd.DataFrame,
    company_id: str,
    target_col: str = DATA_CONFIG.target_column,
    timestamp_col: str = DATA_CONFIG.timestamp_column
) -> None:
    """Plots the time series splits, i.e., k-fold walk-forward validation, for a given company
    ID's hourly energy demand.

    Args:
        data (pd.DataFrame): DataFrame containing a 1-D time series of hourly energy demand.
        company_id (str): ID of the company that the time-series splits will be plotted for.
        target_col (str, optional): Name of the column that contains the 1-D time series,
        i.e., the target variable. Defaults to DATA_CONFIG.target_column.
        temporal_column (str, opitonal): Name of the column that contains the timestamps.
        Defaults to DATA_CONFIG.timestamp_column.
    """
    try:
        data = data.query(f"company_id == '{company_id}'")
        train_splits, val_splits = data.pipe(get_time_series_splits)
        k: int = len(train_splits)

        # plot the company ID's k-fold walk-forward validation
        _, ax = plt.subplots(k, figsize=(20, 14), sharex=True)
        for fold, (train_split, val_split) in enumerate(zip(train_splits, val_splits)):
            y_label: str = f"{company_id.title()} Energy Demand (MW)"
            if fold == 0:
                query: str = f"{timestamp_col} <= '{train_split}'"
                train: pd.Series = data.set_index(timestamp_col).query(query)[target_col]
                train.plot(ax=ax[fold], style="-", label="Train Set", color="black")
                query = f"{timestamp_col} > '{train_split}' & {timestamp_col} <= '{val_split}'"
                val: pd.Series = data.set_index(timestamp_col).query(query)[target_col]
                val.plot(ax=ax[fold], style="--", label="Validation Set", color="black")
                ax[fold].axvline(val.index.min(), color="red", lw=3, ls="--")
                ax[fold].set_title(f"Fold {fold+1}")
                ax[fold].set_ylabel(y_label)
                ax[fold].grid(which="both", alpha=0.3)
                ax[fold].legend(loc="best", frameon=True)
            else:
                query: str = f"{timestamp_col} <= '{train_split}'"
                train: pd.Series = data.set_index(timestamp_col).query(query)[target_col]
                train.plot(ax=ax[fold], style="-", label="Train Set", color="black")
                query = f"{timestamp_col} > '{train_split}' & {timestamp_col} <= '{val_split}'"
                val: pd.Series = data.set_index(timestamp_col).query(query)[target_col]
                val.plot(ax=ax[fold], style="--", label="Validation Set", color="black")
                ax[fold].axvline(val.index.min(), color="red", lw=3, ls="--")
                ax[fold].set_title(f"Fold {fold+1}")
                ax[fold].set_xlabel("Timestamp (UTC)")
                ax[fold].set_ylabel(y_label)
                ax[fold].grid(which="both", alpha=0.3)
        plt.tight_layout()
    except Exception as e:
        raise e
