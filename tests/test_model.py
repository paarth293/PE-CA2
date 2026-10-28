"""Tests of the actual Colab export. Run after adding the .pkl to models/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import FEATURES, MODEL_DIR, create_app, load_model

if not (MODEL_DIR / "california_housing_rf_model.pkl").is_file():
    pytest.skip("Colab model has not been imported yet; real-model verification remains pending.", allow_module_level=True)


@pytest.fixture(scope="module")
def exported():
    return load_model(MODEL_DIR)


def test_exported_model_metadata_and_prediction(exported):
    model, metadata = exported
    assert metadata["parameters"]["n_estimators"] == model.steps[-1][1].n_estimators
    example = json.loads((MODEL_DIR.parent / "example_request.json").read_text())
    frame = pd.DataFrame([example], columns=model.feature_names_in_)
    expected = float(model.predict(frame)[0])
    assert np.isfinite(expected) and expected > 0
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model-info").json()["target"] == "median_house_value"
        result = client.post("/predict", json=example)
        assert result.status_code == 200
        assert result.json()["predicted_median_house_value_usd"] == pytest.approx(expected, rel=1e-9)
        reordered = dict(reversed(list(example.items())))
        assert client.post("/predict", json=reordered).json()["predicted_median_house_value_usd"] == pytest.approx(expected, rel=1e-9)
        assert client.post("/predict", json={**example, "total_bedrooms": None}).status_code == 200


def test_real_pipeline_matches_dataset_schema(exported):
    model, _ = exported
    assert set(model.feature_names_in_) == set(FEATURES)
    data = pd.read_csv(Path(__file__).resolve().parents[1] / "data/housing.csv")
    frame = data[FEATURES].head(20)
    actual = model.predict(frame)
    assert np.isfinite(actual).all()
    assert (actual > 0).all()
