"""Verify the real exported model; exits unsuccessfully if the model is absent."""
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import MODEL_DIR, load_model

model, metadata = load_model(MODEL_DIR)
model_path = MODEL_DIR / "california_housing_rf_model.pkl"
print("Loaded complete fitted pipeline:", model_path.name)
print("Features:", list(model.feature_names_in_))
print("Trees:", model.steps[-1][1].n_estimators)
print("SHA256:", hashlib.sha256(model_path.read_bytes()).hexdigest())
print("Exported model and metadata are compatible. Run API tests to verify predictions.")
