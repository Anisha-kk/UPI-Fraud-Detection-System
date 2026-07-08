import json
from evidently import Report
from evidently.presets import DataDriftPreset
import os
from config import TRAIN_DATA
import pandas as pd

def drift_report(X,batch_id):
    train_df = pd.read_csv(TRAIN_DATA)
    reference_data = train_df.drop(["fraud_label","fraud_type"],axis=1)
    if "fraud_type" in X.columns:
        X = X.drop("fraud_type",axis=1)#REmoving fraud_type from input data, if available
    report = Report(metrics=[
        DataDriftPreset()
    ])
    result =report.run(
        reference_data=reference_data,   # training or baseline data
        current_data=X
    )
    #Save the report
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/drift_report_{batch_id}.json"
    report_dict = result.dict()
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    return report_path