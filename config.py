MODEL_PATH = "artifacts/models/fraud_pipeline.pkl"
THRESHOLD_PATH = "artifacts/models/threshold.pkl"
FEATURE_NAMES_PATH = "artifacts/models/feature_names.pkl"

#Data files path
DATASET = "data/raw/synthetic_upi_fraud_dataset.csv"
TRAIN_DATA = "data/processed/train.csv"
TEST_DATA = "data/processed/test.csv"
PROD_DATA = "data/production/production.csv"

#Training eval results
CONFUSION_MATRIX_FILE = "artifacts/eval_results_from_training/confusion_matrix.csv"
MODEL_METRICS_FILE = "artifacts/eval_results_from_training/model_metrics.csv"
CLASSIFICATION_REPORT_FILE = "artifacts/eval_results_from_training/classification_report.csv"


import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:abc1234@localhost:5432/fraud_db"
)