"""Forecasting pipeline"""

import pandas as pd

from xgboost import XGBRegressor

from src.forecast import generate_forecast
from src.ingest import ingest_data
from src.train import train_model
from src.transform import transform_data


class ForecastingPipeline:
    """Class that encapsulates the forecasting pipeline"""

    def __init__(self, query_id: str):
        self.id: str = query_id
        self.stationary_data, self.target_name = ingest_data(query_id)
        self.features, self.labels = transform_data(self.stationary_data, self.target_name)
        self.model: XGBRegressor = train_model(self.features, self.labels)

    def run(self) -> pd.Series:
        """Returns the recursive forecast

        Returns:
            pd.Series: Recursive forecast
        """
        return generate_forecast(self.model, self.features, self.labels, self.stationary_data)
