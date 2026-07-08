from fastapi import APIRouter
from api.pydantic_model import TransactionInput
from api.services.prediction import predict_transaction
from fastapi import Request

router = APIRouter(prefix="", tags=["Prediction"])

@router.post("/predict")
def predict(request: Request,transaction: TransactionInput):
    model = request.app.state.model
    threshold = request.app.state.threshold
    return predict_transaction(model,threshold,transaction)