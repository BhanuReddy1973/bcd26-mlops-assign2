# Setup Guide - MLOps Assignment

## Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/BhanuReddy1973/bcd26-mlops-assign2.git
cd bcd26-mlops-assign2
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows (PowerShell):
```bash
.\venv\Scripts\activate
```

Windows (CMD):
```bash
venv\Scripts\activate.bat
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Add Dataset

Place your `customer_churn.csv` file in the `data/raw/` directory.

## Running the Project

### Initialize DVC

```bash
dvc init
```

### Track Data with DVC

```bash
dvc add data/raw/customer_churn.csv
git add data/raw/customer_churn.csv.dvc .gitignore
git commit -m "Track raw data with DVC"
```

### Run Training Pipeline

```bash
python -m training.train
```

Or use DVC:

```bash
dvc repro
```

### Start MLflow UI

```bash
mlflow ui
```

Access at: http://localhost:5000

### Start FastAPI Server

```bash
uvicorn inference.app:app --reload
```

Access at: http://127.0.0.1:8000
API Docs: http://127.0.0.1:8000/docs

## Testing the API

### Using curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 840.0
  }'
```

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw dataset
│   ├── processed/        # Processed data
│   └── external/         # External data
├── src/
│   ├── data/            # Data loading & preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training & evaluation
│   └── utils/           # Utility functions
├── training/            # Training scripts
├── inference/           # FastAPI application
├── models/              # Saved models
├── reports/             # Metrics and figures
├── notebooks/           # Jupyter notebooks
├── dvc.yaml            # DVC pipeline
└── requirements.txt    # Dependencies
```

## Troubleshooting

### Issue: Module not found

Solution: Ensure virtual environment is activated and dependencies are installed.

### Issue: Data file not found

Solution: Place `customer_churn.csv` in `data/raw/` directory.

### Issue: MLflow model not found

Solution: Run training pipeline first, then set model alias to "Production" in MLflow UI.
