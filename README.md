# California Housing Price Prediction

A simple Random Forest project using the [Kaggle California housing dataset](https://www.kaggle.com/datasets/hosammhmdali/house-price-dataset). The target is `median_house_value` in USD.

## Project structure

```text
PE-CA2/
├── app/
│   ├── __init__.py
│   └── main.py
├── california_housing_rf_model.pkl
├── california_housing_rf_metadata.json
├── test_api.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .github/workflows/ci.yml
└── README.md
```

The model was exported from Colab with its preprocessing pipeline. Its parameters and reported evaluation metrics are saved in the metadata file. It uses 200 trees and scikit-learn 1.6.1. The model file is losslessly compressed so it fits GitHub's file limit; `joblib.load` reads it directly.

## Run in VS Code

Use Python 3.11. From this folder in the PowerShell terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. Execute `GET /`, `GET /health`, and `POST /predict`. Use this prediction request:

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```

## Run through Docker

Start Docker Desktop and stop the local API with Ctrl+C first.

```powershell
docker build -t house-price-api .
docker run --rm -p 8000:8000 house-price-api
```

Open http://127.0.0.1:8000/docs and execute the same endpoints. Save screenshots of the prediction response and terminal for examination proof.

## GitHub Actions

Pushing to `main` runs `.github/workflows/ci.yml`: install dependencies, test the model/API, build Docker, start the container, and execute the HTTP endpoints. Open the repository's Actions tab to see the result.

Keep the executed Colab notebook and runtime screenshot separately as training evidence. Scikit-learn Random Forest uses the CPU even in a Colab GPU runtime.
