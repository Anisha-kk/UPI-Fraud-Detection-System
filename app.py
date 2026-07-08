from fastapi import FastAPI
from api.routes.predict import router as predict_router
from api.routes.batch import router as batch_router
import joblib
import os
from config import MODEL_PATH, THRESHOLD_PATH
import json
from db.logger import log_validation_error
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request

app = FastAPI(title="UPI TRANSACTION FRAUD DETECTION SYSTEM")

MODEL = None
THRESHOLD = None

@app.on_event("startup")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"Threshold not found at {THRESHOLD_PATH}")
    app.state.model = joblib.load(MODEL_PATH)
    app.state.threshold = joblib.load(THRESHOLD_PATH)

"""
Request comes in
FastAPI tries to parse JSON → TransactionInput
If valid → enters predict()
If invalid → raises RequestValidationError
"""
#Logging pydantic validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if not request.url.path.startswith("/predict"):
        return JSONResponse(
        status_code=422,
        content={"message":"Validation error"}
    )
    else:
        body_bytes = await request.body()
        body_str = body_bytes.decode() if body_bytes else None
        txn_id = None
        body_json = None

        # Try to extract txn_id safely
        if body_str:
            try:
                body_json = json.loads(body_str)
                txn_id = body_json.get("txn_id")
            except Exception:
                txn_id = None
        for error in exc.errors():
            log_validation_error(
                txn_id=txn_id,
                endpoint=request.url.path,
                client_ip=request.client.host if request.client else None,
                field=".".join(map(str, error["loc"])),
                error_type=error["type"],
                message=error["msg"],
                payload=body_json
            )
        return JSONResponse(
            status_code=422,
            content={
                "message": "Validation error",
                "errors": exc.errors(),
            },
        )

app.include_router(predict_router)
app.include_router(batch_router)




