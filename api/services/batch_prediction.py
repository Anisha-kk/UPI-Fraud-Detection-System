import os
import time
import uuid
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from api.services.helper import category_input_converter, remove_error_rows_and_log
from db.logger import log_batch_predictions,log_batch_run
from monitoring.diagnostics import run_diagnostics
from monitoring.evidently_drift import drift_report

def process_batch(MODEL,THRESHOLD,file):
    #Reading production data
    df = pd.read_csv(file.file)

    #Creating a unique batch id
    batch_id = str(uuid.uuid4())

    #Copying the input file to batch folder
    os.makedirs("data/batches", exist_ok=True)
    file_path = f"data/batches/batch_{batch_id}.csv"
    df.to_csv(file_path, index=False)

    if "fraud_label" in df.columns:
        y_true = df["fraud_label"].copy()
        df = df.drop(columns=["fraud_label"])
    else:
        y_true = None
    X = df.copy()
    
    start = time.perf_counter()

    #Convert category inputs to the same case as used in training
    X = category_input_converter(X,MODEL)

    #Running diagnostic test for drift
    #the diagnostics should be computed once for the entire batch
    diagnostics = run_diagnostics(X, MODEL)
    drift_present = len(diagnostics.get("anomalies", [])) > 0
    drift_report_path = drift_report(df,batch_id)

    #Clean the df of rows with missing and incorrect values
    X,removed_rows = remove_error_rows_and_log(X,diagnostics.get("anomalies", []),batch_id)
    #Remove from y_true rows removed  from X
    if y_true is not None:
        y_true = (
            y_true
            .drop(index=removed_rows)
            .reset_index(drop=True)
        )

    #Prediction
    proba = MODEL.predict_proba(X)[:, 1]
    pred = (proba > THRESHOLD).astype(int)
    latency_ms = (time.perf_counter() - start) * 1000#Time to make the prediction
    
    #Metrics
    if y_true is not None:
        report = classification_report(y_true, pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        roc_auc = roc_auc_score(y_true, proba)
        pr_auc = average_precision_score(y_true, proba)
        cm = confusion_matrix(y_true, pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual_Normal", "Actual_Fraud"],
            columns=["Pred_Normal", "Pred_Fraud"]
        )
    else:
        report = None
        report_df = None
        roc_auc = None
        pr_auc = None
        cm_df = None

    X["fraud_probability"] = proba
    X["prediction"] = pred

    #Finding true label if present
    if y_true is not None:
        X["true_label"] = y_true.values
        X["is_correct"] = X["prediction"] == X["true_label"]
    else:
        X["true_label"] = None
        X["is_correct"] = None
    #--------------------
    #Logging predictions 
    #--------------------
    log_batch_predictions(X,batch_id,THRESHOLD)
    #--------------------
    #Logging metrics 
    #--------------------
    log_batch_run(batch_id,roc_auc,pr_auc,report,cm_df.to_dict(),diagnostics,latency_ms,drift_present,drift_report_path)