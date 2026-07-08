import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.drop(
            columns=[
                "timestamp",
                "txn_id",
                "fraud_type",
                "sender_id",
                "receiver_id",
            ],
            inplace=True,
            errors="ignore"   # safer during inference in case of abssence of a column
        )
        X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)
        X["day_week_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["day_week_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        X.drop(
            columns=["hour_of_day", "day_of_week", "month"],
            inplace=True,
            errors="ignore"
        )
        return X