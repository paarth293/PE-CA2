"""Real HTTP checks. --allow-missing-model explicitly checks the pending state."""
import argparse
from http.client import RemoteDisconnected
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
parser.add_argument("--allow-missing-model", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
example = json.loads((root / "example_request.json").read_text())


def request(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(args.base_url.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


for attempt in range(30):
    try:
        status, data = request("/live")
        if status == 200 and data["status"] == "alive":
            break
    except (URLError, TimeoutError, ConnectionError, RemoteDisconnected):
        pass
    if attempt == 29:
        raise RuntimeError("API did not become available within the startup timeout.")
    time.sleep(2)

assert request("/")[0] == 200
assert request("/predict", {**example, "total_rooms": -1})[0] == 422
status, health = request("/health")
if status == 503 and args.allow_missing_model:
    assert health["status"] == "model_unavailable"
    assert request("/predict", example)[0] == 503
    assert request("/model-info")[0] == 503
    print("PASS: HTTP startup, liveness, validation, and missing-model responses.")
    print("PENDING: real-model predictions; import the Colab model and restart.")
else:
    assert status == 200, f"Model is not ready: {health}"
    sys.path.insert(0, str(root))
    import pandas as pd
    from main import MODEL_DIR, load_model
    model, _ = load_model(MODEL_DIR)
    expected = float(model.predict(pd.DataFrame([example], columns=model.feature_names_in_))[0])
    assert request("/model-info")[1]["target"] == "median_house_value"
    status, output = request("/predict", example)
    assert status == 200
    actual = output["predicted_median_house_value_usd"]
    assert math.isfinite(actual) and actual > 0
    assert math.isclose(actual, expected, rel_tol=1e-9)
    print("PASS: real HTTP prediction agrees with exported model; invalid input rejected.")
    print("Prediction (USD):", actual)
