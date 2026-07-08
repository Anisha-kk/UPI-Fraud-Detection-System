from db.database import SessionLocal
from db.schema import Prediction, BatchRun, RemovedBatchRow

def orm_to_dict(obj):
    return {
        column.name: getattr(obj, column.name)
        for column in obj.__table__.columns
    }

def get_predictions():
    db = SessionLocal()
    try:
        predictions = (
            db.query(Prediction)
            .order_by(Prediction.pred_timestamp.desc())
            .all()
        )
        return predictions
    finally:
        db.close()


def get_batch_runs():
    db = SessionLocal()
    try:
        batch_runs = (
            db.query(BatchRun)
            .order_by(BatchRun.run_time.desc())
            .all()
        )
        return batch_runs
    finally:
        db.close()

def removed_rows():
    db = SessionLocal()
    rows_removed=None
    rows_removed= db.query(RemovedBatchRow).order_by(RemovedBatchRow.batch_id, RemovedBatchRow.row_index).all()
    return rows_removed
                