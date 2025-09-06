import argparse, json, os, pickle, sys
import numpy as np
import pandas as pd

CLASS_LABELS = {0: "F", 1: "D", 2: "C", 3: "B", 4: "A"}

DEFAULT_ROW = {
    "Age": 17, "StudyTimeWeekly": 8, "Absences": 3, "GPA": 3.4,
    "Gender": "F", "Ethnicity": "GroupA", "ParentalEducation": "Bachelors",
    "Tutoring": "Yes", "ParentalSupport": "High", "Extracurricular": "Yes",
    "Sports": "No", "Music": "Yes", "Volunteering": "No"
}

def parse_input(row_str: str | None, row_file: str | None):
    if row_file:
        with open(row_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif row_str:
        payload = json.loads(row_str)
    else:
        payload = {"instances": [DEFAULT_ROW]}
    rows = payload.get("instances", payload)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError("Expect dict, list of dicts, or {'instances': [...]} JSON.")
    return rows

def try_api(api_url: str, rows):
    try:
        import requests  # optional dependency for API mode
    except Exception:
        return None
    try:
        r = requests.post(f"{api_url.rstrip('/')}/predict",
                          json={"instances": rows},
                          timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def load_artifacts(pre_path: str, model_path: str):
    with open(pre_path, "rb") as f:
        pre = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    num_cols = list(pre.transformers_[0][2])
    cat_cols = list(pre.transformers_[1][2])
    required_cols = num_cols + cat_cols
    return pre, model, required_cols

def align_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    for c in required_cols:
        if c not in df:
            df[c] = np.nan
    return df[required_cols]

def offline_predict(rows, pre_path: str, model_path: str):
    pre, model, required_cols = load_artifacts(pre_path, model_path)
    X = pd.DataFrame(rows)
    X = align_columns(X, required_cols)
    Xtr = pre.transform(X)
    preds = model.predict(Xtr)
    out = [int(p) for p in preds]
    resp = {"predictions": out}
    if hasattr(model, "predict_proba"):
        classes = [int(c) for c in getattr(model, "classes_", [])]
        probs = model.predict_proba(Xtr)
        resp["probabilities"] = [dict(zip(classes, map(float, row))) for row in probs]
    missing = [c for c in required_cols if c not in X.columns]
    if missing:
        resp["_note"] = f"Missing columns imputed by pipeline: {missing}"
    return resp

def normalize_probas(prob_list):
    if not isinstance(prob_list, list):
        return None
    norm = []
    for d in prob_list:
        try:
            norm.append({int(k): float(v) for k, v in d.items()})
        except Exception:
            return None
    return norm

def pretty_print(resp: dict, topk: int):
    preds = resp.get("predictions", [])
    probas = normalize_probas(resp.get("probabilities"))
    width = 60
    print("═" * width)
    print(" Quick Test Results ".center(width, " "))
    print("═" * width)
    for i, pred in enumerate(preds, start=1):
        label = CLASS_LABELS.get(int(pred), str(pred))
        print(f"#{i}  Prediction: {pred} ({label})")
        if probas and i-1 < len(probas):
            items = list(probas[i-1].items())
            items.sort(key=lambda kv: kv[1], reverse=True)
            print("   Top probabilities:")
            for cls, p in items[:min(topk, len(items))]:
                lab = CLASS_LABELS.get(int(cls), str(cls))
                print(f"     - {cls} ({lab}): {p:.4f}")
        print("-" * width)
    note = resp.get("_note")
    if note:
        print(note)
    print("Done.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-url", default="http://127.0.0.1:8080", help="Flask API base URL")
    p.add_argument("--offline", action="store_true", help="Force offline mode (no server)")
    p.add_argument("--row", help="JSON string for a single row")
    p.add_argument("--row-file", help="Path to JSON file with one row/list/{'instances':[...]}")
    p.add_argument("--pre", default="artifact/preprocessor.pkl", help="Path to preprocessor.pkl")
    p.add_argument("--model", default="artifact/model.pkl", help="Path to model.pkl")
    p.add_argument("--format", choices=["pretty", "json"], default="pretty", help="Output format")
    p.add_argument("--topk", type=int, default=3, help="How many class probabilities to show in pretty mode")
    args = p.parse_args()

    try:
        rows = parse_input(args.row, args.row_file)

        resp = None
        if not args.offline:
            resp = try_api(args.api_url, rows)

        if resp is None:
            resp = offline_predict(rows, args.pre, args.model)

        if args.format == "json":
            print(json.dumps(resp, indent=2))
        else:
            pretty_print(resp, args.topk)

        sys.exit(0)
    except Exception as e:
        print("quick_test failed:", repr(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
