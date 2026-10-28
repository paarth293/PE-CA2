import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import ROOT, app, model

client = TestClient(app)
example = {
    "longitude": -122.23, "latitude": 37.88,
    "housing_median_age": 41, "total_rooms": 880,
    "total_bedrooms": 129, "population": 322,
    "households": 126, "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


def test_health():
    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "healthy"}


def test_prediction_matches_saved_model():
    expected = model.predict(pd.DataFrame([example], columns=model.feature_names_in_))[0]
    response = client.post("/predict", json=example)
    assert response.status_code == 200
    price = response.json()["predicted_median_house_value_usd"]
    assert math.isfinite(price) and price > 0
    assert price == pytest.approx(expected)


def test_missing_bedrooms():
    response = client.post("/predict", json={**example, "total_bedrooms": None})
    assert response.status_code == 200
    assert response.json()["predicted_median_house_value_usd"] > 0


def test_invalid_input():
    assert client.post("/predict", json={**example, "total_rooms": -1}).status_code == 422


def test_saved_parameters():
    metadata = json.loads((ROOT / "california_housing_rf_metadata.json").read_text())
    parameters = model.steps[-1][1].get_params()
    for name, value in metadata["parameters"].items():
        assert parameters[name] == value
