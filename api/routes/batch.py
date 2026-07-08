import traceback

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from api.services.batch_prediction import process_batch

router = APIRouter(prefix="", tags=["Batch"])

@router.post("/batch_predict")
def batch_predict(request: Request,file: UploadFile=File(...)):
    try:
        if file.content_type != "text/csv":    #Only allow csv files
            raise HTTPException(
                status_code=400,
                detail="Only CSV files allowed"
            )
        model = request.app.state.model
        threshold = request.app.state.threshold
        process_batch(model,threshold,file)
        return "Success"
    except HTTPException:
        raise
    except Exception as e:
        return {
            "error": "Batch inference failed",
            "exception_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
    