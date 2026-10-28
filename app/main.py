from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the complete preprocessing and Random Forest pipeline saved in Colab.
ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "california_housing_rf_model.pkl")
app = FastAPI(title="California Housing API")


class HousingInput(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float | None = Field(default=None, ge=0)
    population: float = Field(ge=0)
    households: float = Field(gt=0)
    median_income: float = Field(ge=0)
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


@app.get("/")
def home():
    return {"message": "California Housing API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: HousingInput):
    features = pd.DataFrame([data.model_dump()], columns=model.feature_names_in_)
    features["total_bedrooms"] = pd.to_numeric(features["total_bedrooms"], errors="coerce")
    prediction = float(model.predict(features)[0])
    return {"predicted_median_house_value_usd": prediction}
