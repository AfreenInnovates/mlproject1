# app.py
import os, pickle, numpy as np, pandas as pd
from flask import Flask, request, jsonify

PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "artifact/preprocessor.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", "artifact/model.pkl")

with open(PREPROCESSOR_PATH, "rb") as f:  # load once
    pre = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

num_cols = list(pre.transformers_[0][2])
cat_cols = list(pre.transformers_[1][2])
required_cols = num_cols + cat_cols

def align(df: pd.DataFrame) -> pd.DataFrame:
    for c in required_cols:
        if c not in df:
            df[c] = np.nan
    return df[required_cols]

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def index():
    return {"status": "ok", "service": "mlproject1", "endpoints": ["/health", "/predict"]}

@app.post("/predict")
def predict():
    payload = request.get_json()
    if payload is None:
        return jsonify({"error": "Invalid JSON"}), 400
    rows = payload.get("instances") if isinstance(payload, dict) and "instances" in payload else payload
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return jsonify({"error": "Expect dict or list of dicts (or {'instances': [...]})"}), 400

    X = pd.DataFrame(rows)
    X = align(X)
    Xtr = pre.transform(X)
    preds = model.predict(Xtr)
    out = [int(p) for p in preds]
    resp = {"predictions": out}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xtr)
        classes = [int(c) for c in getattr(model, "classes_", [])]
        resp["probabilities"] = [dict(zip(classes, [float(x) for x in row])) for row in probs]
    return jsonify(resp)

application = app

if __name__ == "__main__":  # local dev only
    app.run(host="0.0.0.0", port=8080, debug=True)
