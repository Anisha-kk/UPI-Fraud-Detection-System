from sqlalchemy import (
    JSON,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    Text,
    func
)
from db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    txn_id = Column(String)
    trxn_timestamp = Column(DateTime)
    pred_timestamp = Column(DateTime)
    amount = Column(Float)
    sender_id = Column(String)
    receiver_id = Column(String)
    prediction_type = Column(String)
    batch_id = Column(String, nullable=True)
    threshold = Column(Float)
    probability = Column(Float)
    prediction = Column(Integer)
    true_label = Column(Integer, nullable=True)#Enter if available
    is_correct = Column(Boolean, nullable=True)#Prediction matches true label-add if available
    latency = Column(Float,nullable=True)
    diagnostics = Column(JSON, nullable=True)
    drift_present = Column(Boolean, nullable=True)

class BatchRun(Base):
    __tablename__ = "batch_run_metrics"
    batch_id = Column(String, primary_key=True)
    run_time = Column(DateTime)
    latency = Column(Float,nullable=True)
    roc_auc = Column(Float)
    pr_auc = Column(Float)
    accuracy = Column(Float)
    confusion_matrix = Column(JSON,nullable=True)
    diagnostics = Column(JSON, nullable=True)
    drift_present = Column(Boolean, nullable=True)
    drift_report_path = Column(String, nullable=True)
    
class ValidationError(Base):
    __tablename__ = "validation_errors"
    id = Column(Integer, primary_key=True, index=True)
    txn_id = Column(String, nullable=True)
    endpoint = Column(String, nullable=False)
    client_ip = Column(String, nullable=True)
    field = Column(String, nullable=True)
    error_type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    payload = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

class RemovedBatchRow(Base):
    __tablename__ = "removed_batch_rows"
    id = Column(Integer, primary_key=True)
    batch_id = Column(String, index=True)
    txn_id = Column(String)
    row_index = Column(Integer)
    reason = Column(String)
    payload = Column(JSON)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )