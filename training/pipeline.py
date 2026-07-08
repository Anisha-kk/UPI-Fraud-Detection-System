from sklearn.pipeline import Pipeline as SkPipeline
from training.custom_transformers import DateFeatureExtractor
from training.preprocess import preprocessor
import xgboost as xgb

def build_pipeline(ratio):
        return SkPipeline([
        ("date", DateFeatureExtractor()),
        ("encode", preprocessor),
        ("model", xgb.XGBClassifier(eval_metric="aucpr",scale_pos_weight=ratio,random_state=42)) #scale_pos_weight is to deal with class imbalance
    ])