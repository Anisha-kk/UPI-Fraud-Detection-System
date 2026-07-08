import time
import pandas as pd
from api.services.helper import category_input_converter
from db.logger import log_prediction
from monitoring.diagnostics import run_diagnostics

def predict_transaction(model,threshold,transaction):
    start = time.perf_counter()
    df = pd.DataFrame([transaction.model_dump()])
    #Check for drift in model
    diagnostics = run_diagnostics(df,model)
    has_anomalies = len(diagnostics.get("anomalies", [])) > 0
    #Convert category inputs to the same case as used in training
    df = category_input_converter(df,model)
    #Prediction
    proba = model.predict_proba(df)[0, 1]
    pred = int(proba > threshold)
    latency_ms = (time.perf_counter() - start) * 1000#Time to make the prediction
    #Save in DB
    log_prediction(data=transaction.model_dump(),probability=proba,prediction=pred,threshold=threshold,
                   diagnostics=diagnostics,has_anomalies=has_anomalies,
                   latency=latency_ms)#prediction_type="online", batch_id=None are default values. So no need to pass them to the logs
    return {
        "fraud_probability": float(proba),
        "prediction": pred
    }