import json
r = json.load(open("artifact/model_report.json"))
print("Best:", r["best_model"])
print("CV:", r["best_model_cv"])
print("Test:", r["test_metrics"])
# Per-class metrics:
for k,v in r["classification_report"].items():
    if k.isdigit():  # class labels like "0", "1", ...
        print(k, {m: round(v[m],3) for m in ["precision","recall","f1-score","support"]})
