# MLProject1 — Student Performance Classifier

A compact, end‑to‑end machine‑learning project that trains a classifier to predict **`GradeClass` (0–4)** from student features, and serves it as a **Flask API**. The repo includes reproducible training scripts, a persisted preprocessing pipeline, a model selection routine over 5 algorithms, and a production‑ready `/predict` endpoint.

---

## Highlights

* **Clean pipeline**: data ingestion → transformation (ColumnTransformer) → model selection (GridSearchCV) → persisted artifacts.
* **Strong baseline**: best model (in our run) = **GradientBoostingClassifier** with \~**0.916** accuracy and **0.869** F1‑macro on the held‑out test set.
* **Portable artifacts**: `artifact/preprocessor.pkl` + `artifact/model.pkl` load directly in the API.
* **Simple REST API**: `/health`, `/` and `/predict` with JSON input; returns predictions (and probabilities when supported).
---

## Project Structure

```
MLPROJECT1/
├─ artifact/
│  ├─ data.csv         
│  ├─ train.csv          # 80% split
│  ├─ test.csv           # 20% split
│  ├─ preprocessor.pkl   
│  └─ model.pkl          # best trained model
├─ app.py               
├─ requirements.txt 
├─ runtime.txt        
├─ .ebignore            
├─ src/
│  ├─ components/
│  │  ├─ data_ingestion.py
│  │  ├─ data_transformation.py
│  │  └─ model_trainer.py
│  ├─ pipeline/
│  │  └─ train_pipeline.py    
│  ├─ logger.py, exception.py, utils.py, __init__.py
│  └─ __init__.py
├─ notebook/            
├─ clean.py              # helper to clean notebook widget metadata
└─ README.md             # this file
```

---

## Problem & Data

The goal is to predict **`GradeClass`** ∈ {0,1,2,3,4}. Typical features used (as wired in the transformer):

* **Numeric**: `Age`, `StudyTimeWeekly`, `Absences`, `GPA`
* **Categorical**: `Gender`, `Ethnicity`, `ParentalEducation`, `Tutoring`, `ParentalSupport`, `Extracurricular`, `Sports`, `Music`, `Volunteering`
* **ID (dropped)**: `StudentID`
* **Target**: `GradeClass`

> If your CSV schema differs, update the lists inside `DataTransformation.get_data_transformer_object()`.

Class distribution example (train): 4.0≈55%, 3.0≈19%, 2.0≈18%, 1.0≈12%, 0.0≈5% (mild imbalance).

---

## Setup

### 1) Create & activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install minimal inference dependencies

(versions pinned to training env — edit as needed)

```text
pandas
numpy
seaborn
scikit-learn
catboost
xgboost
imblearn
```

```powershell
pip install -r requirements.txt
```


---

## Training Pipeline

You can run each component separately


1. **Data ingestion** (reads `notebook/data/student_data.csv`, writes splits under `artifact/`)

```powershell
python src/components/data_ingestion.py
```

2. **Data transformation** (fits preprocessor, saves `preprocessor.pkl`)

```powershell
python src/components/data_transformation.py
```

3. **Model training** (grid‑search over 5 models, saves best `model.pkl` and `model_report.json`)

```powershell
python src/components/model_trainer.py
```


### What the transformer does

* **Numeric pipeline**: median imputation → `StandardScaler`
* **Categorical pipeline**: most‑frequent imputation → `OneHotEncoder(handle_unknown="ignore")` → `StandardScaler(with_mean=False)`
* Drops `StudentID` and the target from features.

### Model selection

* Tries: Logistic Regression, Random Forest, Gradient Boosting (GBDT), SVC, KNN
* Uses `GridSearchCV(cv=5)`. The current config selects by **F1‑macro** (safer with imbalance) and also reports **accuracy**. Set `scoring="accuracy"` in `model_trainer.py` to select by accuracy instead.

### Inspect the report

```python
import json
r = json.load(open("artifact/model_report.json"))
print(r["best_model"])               
print(r["best_model_cv"])             # best params + CV score
print(r["test_metrics"])              # accuracy & f1_macro on held‑out test
```

---

## Local Inference (Flask API)

Start the dev server:

```powershell
python app.py
# Running on http://127.0.0.1:8080
```

### Health

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health -Method GET
```

### Predict (PowerShell example)

```powershell
$body = @{
  instances = @(@{
    Age=17; StudyTimeWeekly=8; Absences=3; GPA=3.4;
    Gender="F"; Ethnicity="GroupA"; ParentalEducation="Bachelors";
    Tutoring="Yes"; ParentalSupport="High"; Extracurricular="Yes";
    Sports="No"; Music="Yes"; Volunteering="No"
  })
} | ConvertTo-Json -Depth 5

Invoke-RestMethod http://127.0.0.1:8080/predict -Method POST -ContentType "application/json" -Body $body
```

**Response**

```json
{
  "predictions": [1],
  "probabilities": [{"0": 0.005, "1": 0.949, "2": 0.013, "3": 0.017, "4": 0.015}]
}
```

### Notes on inputs

* Order of keys doesn’t matter; missing keys are filled with `NaN` and imputed by the pipeline.
* Unknown categories are safely ignored (`OneHotEncoder(handle_unknown="ignore")`).
* The endpoint accepts **one object** or a list of objects, or `{ "instances": [...] }`.

### Python snippet for quick test

```python
python quick_test.py --offline
```

to test on a default data <br>

change: <br>

DEFAULT_ROW = {
    "Age": 17, "StudyTimeWeekly": 8, "Absences": 3, "GPA": 3.4,
    "Gender": "F", "Ethnicity": "GroupA", "ParentalEducation": "Bachelors",
    "Tutoring": "Yes", "ParentalSupport": "High", "Extracurricular": "Yes",
    "Sports": "No", "Music": "Yes", "Volunteering": "No"
}

to test on custom data

---