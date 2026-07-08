import pandas as pd
from sklearn.metrics import (
    average_precision_score,precision_score, recall_score,
    classification_report, roc_auc_score, f1_score,confusion_matrix)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from config import CLASSIFICATION_REPORT_FILE, CONFUSION_MATRIX_FILE, MODEL_METRICS_FILE, TEST_DATA

#Test data
test_df = pd.read_csv(TEST_DATA)
X_test = test_df.drop(columns=["fraud_label"])
y_test = test_df["fraud_label"]

def evaluate(model,threshold):
    test_proba = model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba > threshold).astype(int)

    roc_auc = roc_auc_score(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)
    
    # -------------------------
    # Print results
    # -------------------------
    print("Classification Report:\n")
    print(classification_report(y_test, test_preds))

    print("ROC_SCORE =", roc_auc)
    print("PR_AUC SCORE =", pr_auc)
    print("Threshold =", threshold)

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_preds,
        display_labels=["Normal", "Fraud"],
        cmap="Blues",
        values_format="d"
    )
    plt.title("Confusion Matrix")
    plt.show()

    #Store the classification report 
    report=classification_report(y_test, test_preds,output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_FILE, index=True)

    #Save model metrics
    metrics_df = pd.DataFrame([{
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "Threshold": threshold,
    "Accuracy": report["accuracy"]
    }])
    metrics_df.to_csv(MODEL_METRICS_FILE, index=False)

    #Store confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Normal", "Actual_Fraud"],
        columns=["Pred_Normal", "Pred_Fraud"]
    )
    cm_df.to_csv(CONFUSION_MATRIX_FILE,index=False)
