import numpy as np
import pandas as pd

from db.logger import log_dirty_rows

#Converts category input case to the same as that used in training
def category_input_converter(X,MODEL):
    preprocessor = MODEL.named_steps["encode"]
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = preprocessor.transformers_[preprocessor.transformers_.index(
        next(t for t in preprocessor.transformers_ if t[0] == "cat"))][2]
    for col, categories in zip(cat_features, ohe.categories_):
        if col not in X.columns:#For example, fraud type will not be present in online inference data
            continue
        col_map = {str(c).lower(): c for c in categories}
        X[col] = (
            X[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(col_map)
        )
    
    return X

#Clean df
def sanitize_for_db(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M",errors="coerce")
    df["txn_id"] = df["txn_id"].where(pd.notna(df["txn_id"]), None)
    return df.where(pd.notna(df), None)

#To remove rows with missing values etc from batch inference input
def remove_error_rows_and_log(X,anomalies,batch_id):
    X=sanitize_for_db(X)#Cleaning df
    BLOCKING_ANOMALIES = {
    "missing_value",
    "negative_value",
    "invalid_binary_value",
    "invalid_month",
    "invalid_day_of_month",
    "invalid_hour_of_day",
    "invalid_day_of_week"
    }
    row_reasons = {}
    for a in anomalies:
        if (
            isinstance(a, dict)
            and a.get("anomaly_type") in BLOCKING_ANOMALIES
            and a.get("row_index") is not None
        ):
            idx = a["row_index"]
            row_reasons.setdefault(idx, []).append(
                a["anomaly_type"]
            )
    for idx, reasons in row_reasons.items():
        row = X.loc[idx]
        log_dirty_rows(
            row=row,
            batch_id=batch_id,
            idx=idx,
            reason=", ".join(reasons)
        )
    X_clean = X.drop(index=row_reasons.keys()).reset_index(drop=True)
    rows_to_remove = sorted(row_reasons.keys())
    return X_clean,rows_to_remove

