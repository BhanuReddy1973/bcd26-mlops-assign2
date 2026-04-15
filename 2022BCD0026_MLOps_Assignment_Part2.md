# MLOps Assignment Part (2)

> **2022 BCD 0026**  
> **Bhanu Reddy**

---

**GitHub Link:**  
[https://github.com/2022BCD0026-bhanu/2022bcd0026-assignment-mlops](https://github.com/2022BCD0026-bhanu/2022bcd0026-assignment-mlops)

---

## Table of Contents

1. [Folder Structure](#1-folder-structure)
2. [Add Dataset to data/raw](#2-add-dataset-to-dataraw)
3. [Data Loading — `src/data/load_data.py`](#3-data-loading)
4. [Data Preprocessing — `src/data/preprocess.py`](#4-data-preprocessing)
5. [Training Entry Point — `training/train.py` (v1)](#5-training-entry-point-v1)
6. [Feature Engineering — `src/features/build_features.py`](#6-feature-engineering)
7. [Update `training/train.py` (v2 — with Feature Engineering)](#7-update-trainingtrainpy-v2)
8. [Model Training — `src/models/train_model.py`](#8-model-training)
9. [Model Evaluation — `src/models/evaluate_model.py`](#9-model-evaluation)
10. [Update `training/train.py` (v3 — Full Pipeline)](#10-update-trainingtrainpy-v3)
11. [DVC — Data Versioning + Pipeline](#11-dvc--data-versioning--pipeline)
12. [Build DVC Pipeline — `dvc.yaml`](#12-build-dvc-pipeline)
13. [MLflow — Experiment Tracking](#13-mlflow--experiment-tracking)
14. [Model Registry + API](#14-model-registry--api)
15. [FastAPI Inference Endpoint](#15-fastapi-inference-endpoint)

---

## 1. Folder Structure

**Step:** Create the project folder structure for the churn-mlops project.

**Command:**
```bash
tree /f
```

**Expected Output:**
```
E:.
│   .gitignore
│   Dockerfile
│   dvc.yaml
│   params.yaml
│   README.md
│   requirements.txt
│
├───data
│   ├───external
│   ├───processed
│   └───raw
├───inference
│       app.py
│       schema.py
│
├───models
├───notebooks
├───reports
│   └───figures
├───src
│   ├───data
│   │       load_data.py
│   │       preprocess.py
│   │
│   ├───features
│   │       build_features.py
│   │
│   ├───models
│   │       evaluate_model.py
│   │       train_model.py
│   │
│   ├───pipeline
│   │       inference_pipeline.py
│   │       training_pipeline.py
│   │
│   └───utils
│           config.py
│           logger.py
│
└───training
        train.py
```

> **📸 Screenshot Placeholder — Folder Structure (tree /f output)**
>
> ![Folder Structure](./screenshots/01_folder_structure.png)

---

## 2. Add Dataset to `data/raw`

**Step:** Place the raw customer churn CSV dataset into the `data/raw/` directory.

- File name: `customer_churn.csv`
- Path: `data/raw/customer_churn.csv`

This dataset contains **7043 rows** and **21 columns** covering telecom customer attributes and churn labels.

---

## 3. Data Loading

**File:** `src/data/load_data.py`

**Purpose:** Load the raw CSV dataset into a pandas DataFrame with basic logging.

```python
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
```

**Key Points:**
- Accepts a file path string and returns a pandas DataFrame
- Logs the shape `(rows, columns)` on success
- Catches and re-raises any exceptions after logging

---

## 4. Data Preprocessing

**File:** `src/data/preprocess.py`

**Purpose:** Clean and normalize the raw dataset before feature engineering and model training.

```python
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess dataset
    """
    df = df.copy()

    # Drop customerID (not useful)
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges (it is string sometimes)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Handle missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    logging.info("Preprocessing completed")
    return df
```

**Steps performed:**
1. **Drop `customerID`** — non-predictive identifier column removed
2. **Fix `TotalCharges`** — coerced to numeric (handles string entries)
3. **Handle nulls** — median imputation for `TotalCharges`
4. **Binary encode target** — `"Yes"` → `1`, `"No"` → `0`

---

## 5. Training Entry Point (v1)

**File:** `training/train.py`

**Purpose:** Initial training script — load data, preprocess, and save processed CSV.

```python
import os
import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

DATA_PATH = "data/raw/customer_churn.csv"
OUTPUT_PATH = "data/processed/data.csv"

def main():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    df = preprocess_data(df)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Processed data saved!")

if __name__ == "__main__":
    main()
```

**Run command:**
```bash
python -m training.train
```

> **📸 Screenshot Placeholder — First Run Output (v1)**
>
> ![Train v1 Run](./screenshots/02_train_v1_run.png)

---

## 6. Feature Engineering

**File:** `src/features/build_features.py`

**Purpose:** Create new informative features from the preprocessed dataset to improve model performance.

```python
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum([1 for val in row if val in ["Yes", "DSL", "Fiber optic"]]),
        axis=1
    )

    # 2. Contract type flag
    df["is_monthly_contract"] = df["Contract"].apply(
        lambda x: 1 if x == "Month-to-month" else 0
    )

    # 3. Average charge per tenure month
    df["avg_charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 4. Tenure groups (categorical bucketing)
    def tenure_group(x):
        if x < 12:
            return "0-1yr"
        elif x < 24:
            return "1-2yr"
        elif x < 48:
            return "2-4yr"
        else:
            return "4+yr"

    df["tenure_group"] = df["tenure"].apply(tenure_group)

    # 5. Has support services
    df["has_support"] = df[["TechSupport", "OnlineSecurity"]].apply(
        lambda row: 1 if "Yes" in row.values else 0,
        axis=1
    )

    logging.info("Feature engineering completed")
    return df
```

**Features created:**

| Feature | Description |
|---|---|
| `num_services` | Count of active services per customer |
| `is_monthly_contract` | Binary flag: 1 if on month-to-month contract |
| `avg_charge_per_tenure` | Total charges divided by (tenure + 1) |
| `tenure_group` | Bucketed tenure: 0-1yr, 1-2yr, 2-4yr, 4+yr |
| `has_support` | Binary flag: 1 if TechSupport or OnlineSecurity is active |

---

## 7. Update `training/train.py` (v2)

**Purpose:** Add feature engineering step to the training pipeline.

```python
import os
import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

DATA_PATH = "data/raw/customer_churn.csv"
OUTPUT_PATH = "data/processed/data.csv"

def main():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    df = preprocess_data(df)

    # ✅ Feature Engineering Step Added
    df = build_features(df)

    # Save processed + feature-engineered data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Processed + feature-engineered data saved!")

if __name__ == "__main__":
    main()
```

**Run command:**
```bash
python -m training.train
```

> **📸 Screenshot Placeholder — Train v2 Run Output (with Feature Engineering)**
>
> ![Train v2 Run](./screenshots/03_train_v2_feature_engineering.png)

**Expected terminal output:**
```
INFO:root:Data loaded successfully with shape: (7043, 21)
INFO:root:Preprocessing completed
INFO:root:Feature engineering completed
Processed + feature-engineered data saved!
```

---

## 8. Model Training

**File:** `src/models/train_model.py`

**Purpose:** Build a scikit-learn pipeline with preprocessing and a Random Forest classifier. Train the model and evaluate on a held-out test set.

```python
import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)

def train_model(df: pd.DataFrame):
    df = df.copy()

    # Define target
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # Full model pipeline
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    logging.info(f"F1 Score: {f1}")
    logging.info(f"ROC-AUC: {roc}")

    return model, f1, roc
```

**Model Architecture:**

```
Pipeline
  ├── ColumnTransformer (preprocessing)
  │     ├── StandardScaler      → numerical columns
  │     └── OneHotEncoder       → categorical columns
  └── RandomForestClassifier
        ├── n_estimators = 100
        └── random_state = 42
```

---

## 9. Model Evaluation

**File:** `src/models/evaluate_model.py`

**Purpose:** Persist evaluation metrics to a JSON report file.

```python
import json
import os

def save_metrics(f1, roc):
    os.makedirs("reports", exist_ok=True)
    metrics = {
        "F1 Score": f1,
        "ROC-AUC": roc
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
```

**Output file:** `reports/metrics.json`
```json
{
    "F1 Score": 0.5533230293663061,
    "ROC-AUC": 0.8356576128023849
}
```

---

## 10. Update `training/train.py` (v3)

**Purpose:** Complete end-to-end training script — loads data, preprocesses, engineers features, trains, saves model artifact, and saves metrics.

```python
import os
import pandas as pd
import joblib
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate_model import save_metrics

DATA_PATH = "data/raw/customer_churn.csv"
OUTPUT_PATH = "data/processed/data.csv"

def main():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    df = preprocess_data(df)

    # Feature Engineering
    df = build_features(df)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Processed + feature-engineered data saved!")

    # Train model
    model, f1, roc = train_model(df)

    # Save model artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Save evaluation metrics
    save_metrics(f1, roc)

if __name__ == "__main__":
    main()
```

**Run command:**
```bash
python -m training.train
```

> **📸 Screenshot Placeholder — Full Pipeline Run Output**
>
> ![Full Pipeline Run](./screenshots/04_full_pipeline_run.png)

**Expected terminal output:**
```
INFO:root:Data loaded successfully with shape: (7043, 21)
INFO:root:Preprocessing completed
INFO:root:Feature engineering completed
Processed + feature-engineered data saved!
INFO:root:F1 Score: 0.5533230293663061
INFO:root:ROC-AUC: 0.8356576128023849
```

---

## 11. DVC — Data Versioning + Pipeline

**Purpose:** Use DVC (Data Version Control) to track datasets and define a reproducible pipeline.

### Step 1 — Set up virtual environment and install dependencies

```bash
python -m venv venv
.\venv\scripts\activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn fastapi uvicorn joblib mlflow dvc
pip freeze > requirements.txt
```

> **📸 Screenshot Placeholder — pip install output**
>
> ![pip install](./screenshots/05_pip_install.png)

### Step 2 — Initialize Git repository

```bash
git init
git add .
git commit -m "Initial commit before DVC"
```

> **📸 Screenshot Placeholder — git init and first commit**
>
> ![Git Init](./screenshots/06_git_init.png)

### Step 3 — Initialize DVC

```bash
dvc init
git add .
git commit -m "Initialize DVC"
```

> **📸 Screenshot Placeholder — dvc init output**
>
> ![DVC Init](./screenshots/07_dvc_init.png)

**DVC creates:**
- `.dvc/config` — DVC configuration
- `.dvc/.gitignore` — ignores DVC cache
- `.dvcignore` — patterns for DVC to ignore

### Step 4 — Track datasets with DVC

```bash
dvc add data/raw/customer_churn.csv
dvc add data/processed/data.csv
git add .
git commit -m "Track datasets with DVC"
```

> **📸 Screenshot Placeholder — dvc add datasets output**
>
> ![DVC Add Datasets](./screenshots/08_dvc_add_datasets.png)

**What this does:**
- Creates `.dvc` pointer files for each tracked file
- Moves actual data to DVC cache
- Only the small `.dvc` files are committed to Git

---

## 12. Build DVC Pipeline

**File:** `dvc.yaml`

**Purpose:** Define a reproducible ML pipeline as a DAG (Directed Acyclic Graph) with tracked inputs/outputs.

```yaml
stages:
  train_pipeline:
    cmd: python -m training.train
    deps:
      - training/train.py
      - src/data/load_data.py
      - src/data/preprocess.py
      - src/features/build_features.py
      - src/models/train_model.py
      - src/models/evaluate_model.py
      - data/raw/customer_churn.csv
    outs:
      - data/processed/data.csv
      - models/model.pkl
      - reports/metrics.json
```

> **📸 Screenshot Placeholder — dvc.yaml in editor**
>
> ![DVC yaml](./screenshots/09_dvc_yaml.png)

### Run the DVC Pipeline

```bash
dvc repro
```

> **📸 Screenshot Placeholder — dvc repro output**
>
> ![DVC Repro](./screenshots/10_dvc_repro.png)

**Expected output:**
```
'data/raw/customer_churn.csv.dvc' didn't change, skipping
Running stage 'train_pipeline':
> python -m training.train
INFO:root:Data loaded successfully with shape: (7043, 21)
INFO:root:Preprocessing completed
INFO:root:Feature engineering completed
Processed + feature-engineered data saved!
INFO:root:F1 Score: 0.5533230293663061
INFO:root:ROC-AUC: 0.8356576128023849
Generating lock file 'dvc.lock'
Updating lock file 'dvc.lock'
```

**DVC pipeline benefits:**
- Re-runs only changed stages (caching)
- Creates `dvc.lock` to record exact state of each run
- Fully reproducible pipeline across machines

---

## 13. MLflow — Experiment Tracking

**Purpose:** Integrate MLflow to automatically log parameters, metrics, and model artifacts for every training run.

### Update `src/models/train_model.py`

```python
import pandas as pd
import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)

def train_model(df: pd.DataFrame):
    df = df.copy()
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        model = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        # Log metrics to MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="churn-model"
        )

        logging.info(f"F1 Score: {f1}")
        logging.info(f"ROC-AUC: {roc}")

    return model, f1, roc
```

**What MLflow logs per run:**

| Category | Values |
|---|---|
| Parameters | `model=RandomForest`, `n_estimators=100` |
| Metrics | `f1_score`, `roc_auc` |
| Artifacts | Serialized sklearn pipeline model |

### Run pipeline with MLflow tracking

```bash
dvc repro --force
```

> **📸 Screenshot Placeholder — dvc repro with MLflow output**
>
> ![DVC Repro MLflow](./screenshots/11_dvc_repro_mlflow.png)

### Launch MLflow UI

```bash
mlflow ui
```
Open in browser: [http://localhost:5000](http://localhost:5000)

> **📸 Screenshot Placeholder — MLflow UI Recent Experiments**
>
> ![MLflow UI Experiments](./screenshots/12_mlflow_ui_experiments.png)

> **📸 Screenshot Placeholder — MLflow Run Detail (metrics, params, artifacts)**
>
> ![MLflow Run Detail](./screenshots/13_mlflow_run_detail.png)

**MLflow Run Summary:**
- Run name: `adaptable-goat-901` (auto-generated)
- Created: 04/15/2026, 02:33:14 PM
- Created by: bhanu
- Status: ✅ Finished
- Duration: 18.2s
- `f1_score`: 0.5533230293663061
- `roc_auc`: 0.8356576128023849

---

## 14. Model Registry + API

### Register Model in MLflow

The model is automatically registered to the MLflow Model Registry via `registered_model_name="churn-model"` in the `log_model()` call.

> **📸 Screenshot Placeholder — MLflow Model Registry Version 1 (no alias)**
>
> ![Model Registry v1 No Alias](./screenshots/14_model_registry_v1.png)

**Run again to register:**
```bash
dvc repro --force
```

**Terminal confirms:**
```
Successfully registered model 'churn-model'.
Created version '1' of model 'churn-model'.
```

> **📸 Screenshot Placeholder — dvc repro --force with registration output**
>
> ![DVC Repro Force Register](./screenshots/15_dvc_repro_force.png)

### Set Alias to Production

In the MLflow UI, navigate to:
`Registered Models → churn-model → Version 1 → Aliases → Add → production`

> **📸 Screenshot Placeholder — Model Version 1 with @production alias**
>
> ![Model Production Alias](./screenshots/16_model_production_alias.png)

> **📸 Screenshot Placeholder — Second MLflow Run (nosy-kit-776)**
>
> ![MLflow Second Run](./screenshots/17_mlflow_second_run.png)

---

## 15. FastAPI Inference Endpoint

### Step 1 — Ensure Model Is In Production

Confirm the model version has the `@production` alias set in MLflow Model Registry.

> **📸 Screenshot Placeholder — Model Registry showing @production alias**
>
> ![Model in Production](./screenshots/18_model_in_production.png)

### Step 2 — Create the API

**File:** `inference/app.py`

```python
from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load model from MLflow registry (production alias)
model = mlflow.pyfunc.load_model("models:/churn-model/Production")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {
        "prediction": int(prediction[0]),
        "churn": "Yes" if prediction[0] == 1 else "No"
    }
```

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns status message |
| `POST` | `/predict` | Accepts customer data dict, returns churn prediction |

**Response schema (`/predict`):**
```json
{
  "prediction": 0,
  "churn": "No"
}
```

### Step 3 — Run the API Server

```bash
uvicorn inference.app:app --reload
```

> **📸 Screenshot Placeholder — uvicorn startup logs**
>
> ![Uvicorn Run](./screenshots/19_uvicorn_run.png)

**Expected server logs:**
```
INFO:  Will watch for changes in these directories: ['E:\\MLOps\\churn-mlops']
INFO:  Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:  Started reloader process [11480] using StatReload
INFO:  Started server process [31444]
INFO:  Application startup complete.
INFO:  127.0.0.1:50566 - "GET /docs HTTP/1.1" 200 OK
INFO:  127.0.0.1:50566 - "GET /openapi.json HTTP/1.1" 200 OK
INFO:  127.0.0.1:59175 - "POST /predict HTTP/1.1" 200 OK
```

### Step 4 — Test the API

Access the auto-generated Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Request URL:**
```
http://127.0.0.1:8000/predict
```

**Sample response:**
```json
{
  "prediction": 0,
  "churn": "No"
}
```

**Response headers:**
```
content-length: 29
content-type: application/json
date: Wed, 15 Apr 2026 09:29:36 GMT
server: uvicorn
```

> **📸 Screenshot Placeholder — FastAPI Swagger UI /predict response (200 OK)**
>
> ![FastAPI Predict Response](./screenshots/20_fastapi_predict_response.png)

---

## End-to-End Pipeline Summary

```
Raw CSV (data/raw/)
      │
      ▼
load_data.py          ← Loads CSV into DataFrame
      │
      ▼
preprocess.py         ← Drops ID, fixes types, imputes nulls, encodes target
      │
      ▼
build_features.py     ← Engineers 5 new features (services, contract, tenure, etc.)
      │
      ▼
data/processed/data.csv  ← Saved via DVC
      │
      ▼
train_model.py        ← sklearn Pipeline (StandardScaler + OHE + RandomForest)
      │
      ├── models/model.pkl       ← Serialized artifact
      ├── reports/metrics.json   ← F1 + ROC-AUC
      └── MLflow Run             ← Params, Metrics, Model logged
                │
                ▼
        MLflow Registry
        churn-model @production
                │
                ▼
        FastAPI (inference/app.py)
        POST /predict → {"churn": "Yes"/"No"}
```

---

## Metrics Achieved

| Metric | Value |
|---|---|
| F1 Score | 0.5533 |
| ROC-AUC | 0.8357 |

---

*Assignment submitted by **Bhanu Reddy** — Roll No: **2022BCD0026***
