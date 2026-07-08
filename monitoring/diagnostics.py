import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any
from datetime import datetime
from utils import json_safe
from config import FEATURE_NAMES_PATH
'''
EXPECTED OUTPUT
{
  "total_anomalies": 3,
  "anomalies": [
    {
      "txn_id": "T101",
      "anomaly_type": "missing_value",
      "feature": "amount",
      "value": null,
      "severity": "high",
      "timestamp": "2026-07-03T10:12:00"
    },
    {
      "txn_id": "T205",
      "anomaly_type": "negative_value",
      "feature": "amount",
      "value": -500,
      "severity": "medium",
      "timestamp": "2026-07-03T10:12:00"
    }
  ]
}
'''


# Basically for batch inference. For online inference, for most of these anomalies, pydantic error is raised, except for
#unknown category 
class AnomalyEvent(BaseModel):
    txn_id: str
    anomaly_type: str
    feature: str
    value: Optional[Any] = None
    severity: str  # "low", "medium", "high"
    timestamp: datetime

#Helper function to create anomalies
def create_anomaly(txn_id, row_index,anomaly_type, feature, value, severity):
    return {
        "txn_id": json_safe(txn_id),#To convert nan to none
        "row_index": json_safe(row_index),
        "anomaly_type": anomaly_type,
        "feature": feature,
        "value": value,
        "severity": severity,
        "timestamp": datetime.utcnow().isoformat()
    }

#Missing values(Nan)
def detect_missing_values(X):
    anomalies = []
    for col in X.columns:
        mask = X[col].isna()
        if mask.any():
            for idx, row in X[mask].iterrows():
                anomalies.append(
                    create_anomaly(
                        txn_id=row.get("txn_id", None),
                        row_index=idx,
                        anomaly_type="missing_value",
                        feature=col,
                        value=None,
                        severity="high"
                    )
                )
    return anomalies

#Negative values in numeric columns
def detect_negative_values(X):
    anomalies = []
    numeric_cols = X.select_dtypes(include="number").columns
    for col in numeric_cols:
        mask = X[col] < 0
        for idx, row in X[mask].iterrows():
            anomalies.append(
                create_anomaly(
                    txn_id=row.get("txn_id", None),
                    row_index=idx,
                    anomaly_type="negative_value",
                    feature=col,
                    value=float(row[col]),
                    severity="medium"
                )
            )
    return anomalies

#Unknown categories
def detect_unknown_categories(X, ohe, cat_cols):
    anomalies = []
    for i, col in enumerate(cat_cols):
        valid = set(ohe.categories_[i])
        mask = ~X[col].isin(valid)
        for idx, row in X[mask].iterrows():
            anomalies.append(
                create_anomaly(
                    txn_id=row.get("txn_id", None),
                    row_index=idx,
                    anomaly_type="unknown_category",
                    feature=col,
                    value=row[col],
                    severity="low"
                )
            )
    return anomalies

#Missing columns  - batch level schema anomaly
def detect_missing_columns(X, expected_cols):
    anomalies = []
    received_cols = set(X.columns)
    expected_cols = set(expected_cols)
    missing = expected_cols - received_cols
    if "fraud_type" in missing:
        missing.remove("fraud_type")#This data will not be available in online inference. So, not a drift
    for col in missing:
        anomalies.append(
            create_anomaly(
                txn_id=None,  # batch-level anomaly
                row_index=None,
                anomaly_type="missing_column",
                feature=col,
                value=None,
                severity="high"
            )
        )
    return anomalies

#Extra columns
def detect_extra_columns(X, expected_cols):
    anomalies = []
    received_cols = set(X.columns)
    expected_cols = set(expected_cols)
    extra = received_cols - expected_cols
    for col in extra:
        anomalies.append(
            create_anomaly(
                txn_id=None,#Batch level anomaly
                row_index= None,
                anomaly_type="extra_column",
                feature=col,
                value=None,
                severity="low"
            )
        )
    return anomalies

#Binary value cols anomaly
def detect_binary_anomalies(X, cols):
    anomalies = []
    for col in cols:
        # invalid values (not 0 or 1)
        mask_invalid = ~X[col].isin([0, 1])
        for idx, row in X[mask_invalid].iterrows():
            anomalies.append(
            create_anomaly(
                txn_id=row.get("txn_id", None),
                row_index=idx,
                anomaly_type="invalid_binary_value",
                feature=col,
                value=row[col],
                severity="high"
            )
        )
        # missing values
        mask_null = X[col].isna()
        for idx, row in X[mask_null].iterrows():
            anomalies.append(
            create_anomaly(
                txn_id=row.get("txn_id", None),
                row_index=idx,
                anomaly_type="missing_value",
                feature=col,
                value=None,
                severity="high"
            )
        )
    return anomalies

#Temporal anomalies
def detect_time_anomalies(X):
    anomalies = []
    checks = {
        "month": (1, 12),
        "day_of_month": (1, 31),
        "hour_of_day": (0, 23),
        "day_of_week": (0, 6),
    }
    for col, (low, high) in checks.items():
        mask = ~X[col].between(low, high)
        for idx, row in X[mask].iterrows():
            anomalies.append(
            create_anomaly(
                txn_id=row.get("txn_id", None),
                row_index=idx,
                anomaly_type=f"invalid_{col}",
                feature=col,
                value=row[col],
                severity="high"
            )
        )
    return anomalies

#Main function
def run_diagnostics(X, MODEL):

    anomalies = []

    # schema drift
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    anomalies += detect_missing_columns(X, feature_names)
    anomalies += detect_extra_columns(X, feature_names)

    # numeric issues
    anomalies += detect_missing_values(X)
    anomalies += detect_negative_values(X)

    # categorical drift
    preprocessor = MODEL.named_steps["encode"]
    ohe = preprocessor.named_transformers_["cat"]
    cat_cols = [
        "sender_city",
        "receiver_city",
        "merchant_category",
        "sender_upi_domain",
        "receiver_upi_domain"
    ]
    anomalies += detect_unknown_categories(X, ohe, cat_cols)

    #binary validation
    binary_cols = [
        "device_changed",
        "ip_changed",
        "city_changed",
        "is_new_beneficiary",
        "is_festival"
    ]
    anomalies += detect_binary_anomalies(X, binary_cols)

    #time validation
    anomalies += detect_time_anomalies(X)

    #Result
    return {
        "total_anomalies": len(anomalies),
        "anomalies": anomalies
    }