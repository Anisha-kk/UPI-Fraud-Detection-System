import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import (
    average_precision_score,precision_score, recall_score,
    classification_report, roc_auc_score, f1_score)
from sklearn.base import clone
from config import MODEL_PATH, THRESHOLD_PATH, TRAIN_DATA
from training.pipeline import build_pipeline
from training.evaluate import evaluate
import joblib
import os

train_df = pd.read_csv(TRAIN_DATA)
y = train_df["fraud_label"]
X = train_df.drop("fraud_label",axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

ratio = (y_train == 0).sum() / (y_train == 1).sum() #negatives/positives wrt fraud transactions

def objective(trial, X_train, y_train, X_valid, y_valid):
    model = build_pipeline(ratio)
    params = {
        "model__n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "model__max_depth": trial.suggest_int("max_depth", 3, 8),
        "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "model__gamma": trial.suggest_float("gamma", 0, 5),
        "model__reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "model__reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
    }
    model.set_params(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_valid)[:, 1]
    return average_precision_score(y_valid, preds)

def run_optuna(X_train, y_train, X_valid, y_valid):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=30
    )
    return study.best_params

def find_best_threshold(proba, y_true):
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_t = 0.5
    best_recall = 0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds)
        if precision >= 0.80 and recall > best_recall:
            best_recall = recall
            best_t = t
    return best_t

if __name__=="__main__":
    # Optuna tuning
    best_params = run_optuna(X_train, y_train, X_valid, y_valid)

    # Build and train model
    model = build_pipeline(ratio)
    model.set_params(**{f"model__{k}": v for k, v in best_params.items()})
    model.fit(X_train, y_train)
    
    # VALIDATION: threshold tuning
    valid_proba = model.predict_proba(X_valid)[:, 1]
    threshold = find_best_threshold(valid_proba, y_valid)

    #Evaluation
    evaluate(model,threshold)

    #Save the model and threshold
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)



