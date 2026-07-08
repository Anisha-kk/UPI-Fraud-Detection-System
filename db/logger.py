from datetime import datetime

import numpy as np
import pandas as pd
from utils import json_safe
from db.database import SessionLocal
from db.schema import Prediction,BatchRun, RemovedBatchRow, ValidationError

def log_prediction(data,probability,prediction,threshold,true_label=None,
                   is_correct=None,diagnostics={},has_anomalies=None,latency=None):
    db = SessionLocal()
    try:
        row = Prediction(
            txn_id = data["txn_id"],
            trxn_timestamp=data["timestamp"],
            pred_timestamp = datetime.now(),
            amount=data["amount"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            prediction_type="online",
            batch_id=None,
            threshold=float(threshold),
            probability=float(probability),
            prediction=int(prediction),
            true_label = true_label,
            is_correct = is_correct,
            latency=latency,
            diagnostics=diagnostics,
            drift_present = has_anomalies
        )

        db.add(row)
        db.commit()
    finally:
        db.close()

def log_batch_predictions(df,batch_id,threshold):
    db = SessionLocal()
    try:
        prediction_objects = []
        for _, row in df.iterrows():
            obj = Prediction(
                txn_id = row["txn_id"],
                trxn_timestamp=row["timestamp"],
                pred_timestamp = datetime.now(),
                amount=row["amount"],
                sender_id=row["sender_id"],
                receiver_id=row["receiver_id"],
                prediction_type="batch",
                batch_id=batch_id,
                threshold=float(threshold),
                probability=float(row["fraud_probability"]),
                prediction=int(row["prediction"]),
                true_label=row.get("true_label"),
                is_correct=row.get("is_correct")
            )
            prediction_objects.append(obj)
        db.bulk_save_objects(prediction_objects)
        db.commit()
    finally:
        db.close()

def log_batch_run(batch_id,roc_auc,pr_auc,report,cm,diagnostics={},latency=None,has_anomalies=None,report_path=None):
    db = SessionLocal()
    try:
        batch = BatchRun(
            batch_id=batch_id,
            run_time=datetime.now(),
            latency = latency,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            accuracy=report["accuracy"],
            confusion_matrix = cm,
            diagnostics=diagnostics,
            drift_present = has_anomalies,
            drift_report_path=report_path
        )
        db.add(batch)
        db.commit()
    finally:
        db.close()

def log_validation_error(txn_id: str | None,endpoint: str,client_ip: str | None,
                         field: str | None,error_type: str,message: str,payload: str | None):
    db = SessionLocal()
    error = ValidationError(
        txn_id=txn_id,
        endpoint=endpoint,
        client_ip=client_ip,
        field=field,
        error_type=error_type,
        message=message,
        payload=payload,
    )
    db.add(error)
    db.commit()
    db.refresh(error)


#logging dirty rows
def log_dirty_rows(row, batch_id, idx, reason):
    db = SessionLocal()
    try:
        txn_id = row.get("txn_id")
        txn_id = None if pd.isna(txn_id) else str(txn_id)
        payload = {
            k: (None if pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        payload = {
            k: json_safe(v)
            for k, v in row.items()
        }
        db.add(
            RemovedBatchRow(
                batch_id=batch_id,
                txn_id=txn_id,
                row_index=int(idx),
                reason=reason,
                payload=payload
            )
        )
        db.commit()
    except Exception as e:
        db.rollback()
        print("DB insert failed:", e)
    finally:
        db.close()