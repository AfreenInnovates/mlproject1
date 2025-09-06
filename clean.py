import json
import pathlib

for nb_path in pathlib.Path(".").rglob("*.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Cleaned: {nb_path}")
