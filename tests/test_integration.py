from io import BytesIO

from fastapi.routing import APIRoute
import pytest
from fastapi.testclient import TestClient
from app import app
from tests.inputs import valid_input

client = TestClient(app)

#Test model and threshold are loaded during startup
def test_model_loaded():
    with TestClient(app):
        assert hasattr(app.state, "model")
        assert hasattr(app.state, "threshold")
        assert app.state.model is not None
        assert app.state.threshold is not None

#Testing that the app startup is successful
def test_app_startup():
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200

#Testing /predict with valid input
def test_predict_success():
    payload = valid_input
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body

#Testing invalid input on /predict
def test_predict_invalid_json():
    response = client.post(
        "/predict",
        data="{invalid json}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422

#Testing batch prediction
def test_batch_predict_success():
    csv = (
        "txn_id,timestamp,sender_id,receiver_id,sender_upi_domain,receiver_upi_domain,sender_account_age_days,receiver_account_age_days,amount,sender_city,receiver_city,device_changed,ip_changed,city_changed,failed_logins_last_24h,txn_count_last_1h,txn_count_last_24h,is_new_beneficiary,merchant_category,amount_to_average_ratio,distance_from_last_transaction,payday_intensity,month,day_of_month,hour_of_day,day_of_week,fraud_label,is_festival\n"
        "TXN00010034,01-10-2025 00:11,859,5896,@ybl,@oksbi,1536,78,1449.279942,Pune,Pune,0,0,0,0,3,1,0,Travel,1.19,79,1,10,1,0,2,0,1\n"
    )
    files = {
        "file": (
            "transactions.csv",
            BytesIO(csv.encode()),
            "text/csv",
        )
    }
    response = client.post("/batch_predict", files=files)
    assert response.status_code == 200

#Test /batch_predict without file
def test_batch_predict_missing_file():
    response = client.post("/batch_predict")
    assert response.status_code == 422

#Test wrong file type
def test_batch_predict_wrong_file():
    files = {
        "file": (
            "sample.txt",
            BytesIO(b"Hello"),
            "text/plain",
        )
    }
    response = client.post("/batch_predict", files=files)
    assert response.status_code in (400, 415, 422)



