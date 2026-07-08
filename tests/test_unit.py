import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from unittest.mock import MagicMock, patch
import json

import app

#########################################################
# Test Startup - Successful Model Loading
#########################################################
def test_load_model_success(monkeypatch):
    monkeypatch.setattr(app.os.path, "exists", lambda x: True)
    fake_model = MagicMock()
    fake_threshold = 0.65
    with patch("app.joblib.load") as mock_load:
        mock_load.side_effect = [fake_model, fake_threshold]
        test_fast_app = FastAPI()
        test_fast_app.state = MagicMock()
        with patch.object(app, "app", test_fast_app):
            app.load_model()
        assert test_fast_app.state.model == fake_model
        assert test_fast_app.state.threshold == fake_threshold

#########################################################
# Missing Model File
#########################################################
def test_load_model_missing_model(monkeypatch):
    def fake_exists(path):
        if path == app.MODEL_PATH:
            return False
        return True
    monkeypatch.setattr(app.os.path, "exists", fake_exists)
    with pytest.raises(FileNotFoundError):
        app.load_model()

#########################################################
# Missing Threshold File
#########################################################
def test_load_model_missing_threshold(monkeypatch):
    def fake_exists(path):
        if path == app.THRESHOLD_PATH:
            return False
        return True
    monkeypatch.setattr(app.os.path, "exists", fake_exists)
    with pytest.raises(FileNotFoundError):
        app.load_model()

#########################################################
# Validation Exception Handler
#########################################################
@pytest.mark.asyncio
async def test_validation_exception_handler():
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }
    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps({"txn_id": "TXN100"}).encode(),
            "more_body": False,
        }
    request = Request(scope, receive)
    errors = [
        {
            "loc": ("body", "amount"),
            "msg": "field required",
            "type": "missing",
        }
    ]
    exc = RequestValidationError(errors)
    with patch("app.log_validation_error") as mock_logger:
        response = await app.validation_exception_handler(request, exc)
        assert response.status_code == 422
        mock_logger.assert_called_once()

#########################################################
# Validation Handler Without txn_id
#########################################################
@pytest.mark.asyncio
async def test_validation_handler_no_txnid():
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [],
        "client": ("127.0.0.1", 8000),
    }
    async def receive():
        return {
            "type": "http.request",
            "body": b"",#No request body
            "more_body": False,
        }
    request = Request(scope, receive)
    errors = [
        {
            "loc": ("body", "amount"),
            "msg": "invalid",
            "type": "value_error",
        }
    ]
    exc = RequestValidationError(errors)
    with patch("app.log_validation_error") as mock_logger:
        response = await app.validation_exception_handler(request, exc)
        assert response.status_code == 422
        mock_logger.assert_called()

#########################################################
# Validation Handler For Non-Predict Route
#########################################################
@pytest.mark.asyncio
async def test_validation_handler_other_route():
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/batch_predict",
        "headers": [],
        "client": ("127.0.0.1", 8000),
    }
    async def receive():
        return {
            "type": "http.request",
            "body": b"",
            "more_body": False,
        }
    request = Request(scope, receive)
    errors = [
        {
            "loc": ("body", "file"),
            "msg": "required",
            "type": "missing",
        }
    ]
    exc = RequestValidationError(errors)
    response = await app.validation_exception_handler(request, exc)
    assert response.status_code == 422



