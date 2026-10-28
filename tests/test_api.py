"""API behavior tests use a stub, not a trained or substituted housing model."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main

EXAMPLE = json.loads((Path(__file__).resolve().parents[1] / "example_request.json").read_text())


@pytest.fixture
def client(tmp_path):
    with TestClient(main.create_app(tmp_path)) as client:
        yield client


def test_docs_and_liveness_without_model(client):
    assert client.get("/").status_code == 200
    assert client.get("/live").json() == {"status": "alive"}
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_missing_model_is_not_reported_as_ready(client):
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "model_unavailable"


@pytest.mark.parametrize("method,path,body", [("get", "/model-info", None), ("post", "/predict", EXAMPLE)])
def test_model_endpoints_unavailable_without_export(client, method, path, body):
    response = client.get(path) if method == "get" else client.post(path, json=body)
    assert response.status_code == 503


@pytest.mark.parametrize("patch", [
    {"total_rooms": -1}, {"ocean_proximity": "INVALID"}, {"unexpected": 1},
    {"latitude": 91}, {"longitude": -181}, {"households": 0},
    {"median_income": -1}, {"total_bedrooms": -1},
])
def test_invalid_input_is_rejected(client, patch):
    assert client.post("/predict", json={**EXAMPLE, **patch}).status_code == 422


def test_missing_required_fields(client):
    assert client.post("/predict", json={}).status_code == 422


class RecordingStub:
    """Deterministic test double; never exported or used by the application."""
    feature_names_in_ = list(reversed(main.FEATURES))

    def predict(self, frame):
        self.frame = frame
        return np.array([123456.0])


def test_prediction_preserves_model_column_order_and_converts_null_to_nan(monkeypatch, tmp_path):
    stub = RecordingStub()
    monkeypatch.setattr(main, "load_model", lambda path: (stub, {}))
    with TestClient(main.create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json={**EXAMPLE, "total_bedrooms": None})
    assert response.status_code == 200
    assert response.json() == {"predicted_median_house_value_usd": 123456.0}
    assert list(stub.frame.columns) == stub.feature_names_in_
    assert np.isnan(stub.frame.loc[0, "total_bedrooms"])
    assert stub.frame.loc[0, "median_income"] == EXAMPLE["median_income"]


def test_invalid_export_keeps_service_alive(monkeypatch, tmp_path):
    def fail(path):
        raise ValueError("Incompatible model version")
    monkeypatch.setattr(main, "load_model", fail)
    with TestClient(main.create_app(tmp_path)) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/health").status_code == 503


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_model_outputs_return_controlled_error(monkeypatch, tmp_path, value):
    stub = RecordingStub()
    stub.predict = lambda frame: [value]
    monkeypatch.setattr(main, "load_model", lambda path: (stub, {}))
    with TestClient(main.create_app(tmp_path)) as client:
        assert client.post("/predict", json=EXAMPLE).status_code == 500

