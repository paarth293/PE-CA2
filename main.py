"""FastAPI service for the California housing pipeline exported from Colab."""
from contextlib import asynccontextmanager
import json
import logging
import math
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income", "ocean_proximity",
]
logger = logging.getLogger(__name__)


class HousingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float | None = Field(default=None, ge=0)
    population: float = Field(ge=0)
    households: float = Field(gt=0)
    median_income: float = Field(ge=0)
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


class PredictionOutput(BaseModel):
    predicted_median_house_value_usd: float


def load_model(model_dir: Path):
    """Load only the user's trusted Colab export; never train at API startup."""
    model_path = model_dir / "california_housing_rf_model.pkl"
    metadata_path = model_dir / "california_housing_rf_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Add the Colab model .pkl and metadata .json to models/, then restart the API.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = metadata.get("sklearn_version", metadata.get("versions", {}).get("scikit-learn"))
    if version and version != sklearn.__version__:
        raise ValueError(f"Model requires scikit-learn {version}; installed version is {sklearn.__version__}.")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline) or not isinstance(model.steps[-1][1], RandomForestRegressor):
        raise ValueError("Export the fitted preprocessing + RandomForestRegressor Pipeline from Colab.")
    if set(getattr(model, "feature_names_in_", [])) != set(FEATURES):
        raise ValueError("Exported model inputs do not match the nine California housing features.")
    for name, value in metadata.get("parameters", {}).items():
        if model.steps[-1][1].get_params().get(name) != value:
            raise ValueError(f"Exported model parameter {name} does not match its metadata.")
    return model, metadata


def create_app(model_dir: Path = MODEL_DIR):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = None
        app.state.metadata = {}
        app.state.model_error = None
        try:
            app.state.model, app.state.metadata = load_model(model_dir)
        except Exception as error:
            # Keep docs/liveness available; readiness and predictions remain unavailable.
            logger.warning("Model unavailable: %s", error)
            app.state.model_error = str(error)
        yield

    app = FastAPI(title="California House Price Prediction API", version="1.0.0", lifespan=lifespan)

    def require_model():
        if app.state.model is None:
            raise HTTPException(status_code=503, detail=app.state.model_error or "Model unavailable.")
        return app.state.model

    @app.get("/")
    def root():
        return {"message": "House Price Prediction API", "docs": "/docs"}

    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.get("/health")
    def health():
        if app.state.model is None:
            return JSONResponse(status_code=503, content={"status": "model_unavailable", "detail": app.state.model_error})
        return {"status": "healthy", "model": "RandomForestRegressor"}

    @app.get("/model-info")
    def model_info():
        model = require_model()
        metadata = app.state.metadata
        return {
            "target": "median_house_value", "currency": "USD",
            "features": list(model.feature_names_in_),
            "parameters": metadata.get("parameters", model.steps[-1][1].get_params()),
            "reported_test_metrics": metadata.get("test_metrics", {}),
        }

    @app.post("/predict", response_model=PredictionOutput)
    def predict(data: HousingInput):
        model = require_model()
        frame = pd.DataFrame([data.model_dump()], columns=model.feature_names_in_)
        # Explicit NaN keeps null bedroom values compatible with median imputation.
        frame["total_bedrooms"] = pd.to_numeric(frame["total_bedrooms"], errors="coerce")
        try:
            value = float(model.predict(frame)[0])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid model prediction")
        except Exception:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Model prediction failed. Check the exported pipeline.") from None
        return {"predicted_median_house_value_usd": value}

    return app


app = create_app()
